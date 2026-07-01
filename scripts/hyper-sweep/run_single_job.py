"""
Worker script: runs inside a single Lightning AI job.

Downloads a dataset from HuggingFace (or uses a teamspace cache), trains one
Lightning Pose model via `litpose train`, and deletes model weights.
Results land in --output_dir (a path on teamspace storage).

Can also be called directly for local sweeps (no Lightning AI required):
    python run_single_job.py --dataset_repo paninski-lab/mirror-mouse-fused \\
        --backbone resnet50_animal_ap10k --train_frames 1 --seed 0 \\
        --output_dir /some/local/path
"""

import argparse
import os
import subprocess
from pathlib import Path

# must be set before huggingface_hub is imported
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

DEFAULT_DATASET_CACHE_DIR = "/teamspace/lightning_storage/datasets"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_repo", required=True,
        help="HuggingFace repo ID, e.g. paninski-lab/mirror-mouse-fused",
    )
    p.add_argument(
        "--dataset_cache_dir", default=DEFAULT_DATASET_CACHE_DIR,
        help="Directory where datasets are cached; set to a local path when running outside Lightning AI",
    )
    p.add_argument("--backbone", required=True)
    p.add_argument("--train_frames", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--losses_to_use", default="",
        help="Comma-separated loss names; empty string = supervised only",
    )
    p.add_argument("--model_type", default="heatmap")
    p.add_argument("--output_dir", required=True, help="Absolute path for training outputs")
    p.add_argument(
        "--download_videos", action="store_true",
        help="Download InD videos/ directory (required for unsupervised losses)",
    )
    p.add_argument(
        "--predict_vids", action="store_true",
        help="Run video prediction after training; also downloads videos_test/",
    )
    p.add_argument("--debug", action="store_true", help="Short smoke-test run (3 epochs)")
    return p.parse_args()


def _snapshot_worker(repo_id, repo_type, local_dir, ignore_patterns, tmp_dir):
    """Runs snapshot_download in a child process so it can be hard-killed on stall.

    Downloads to local disk (tmp_dir) first to avoid .incomplete file issues on
    network/distributed filesystems, then copies to the final local_dir.
    """
    import shutil
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(tmp_dir),
        ignore_patterns=ignore_patterns,
    )
    print(f"  Copying dataset from {tmp_dir} to {local_dir}...", flush=True)
    shutil.copytree(str(tmp_dir), str(local_dir), dirs_exist_ok=True)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)


def _watchdog(process, cache_path, stall_timeout, check_interval=30):
    """Monitors download progress; terminates process if no bytes land for stall_timeout seconds."""
    import time

    def dir_size():
        try:
            return sum(f.stat().st_size for f in Path(cache_path).rglob("*") if f.is_file())
        except Exception:
            return 0

    last_size = dir_size()
    last_progress = time.time()

    while process.is_alive():
        time.sleep(check_interval)
        size = dir_size()
        if size > last_size:
            last_size = size
            last_progress = time.time()
        elif time.time() - last_progress > stall_timeout:
            print(f"  No download progress for {stall_timeout}s — terminating stalled download.", flush=True)
            process.terminate()
            return


def _wait_for_filesystem_sync(cache_path, check_interval=15, timeout=300):
    """Poll until the file count in cache_path stabilizes across two consecutive checks.

    shutil.copytree can return before all writes are committed on distributed/network
    filesystems. Polling is more reliable than a fixed sleep.
    """
    import time

    def count_files():
        try:
            return sum(1 for f in cache_path.rglob("*") if f.is_file())
        except Exception:
            return 0

    print("Waiting for dataset to become fully accessible on teamspace filesystem...", flush=True)
    prev_count = -1
    start = time.time()
    deadline = start + timeout

    while True:
        count = count_files()
        elapsed = int(time.time() - start)
        if count > 0 and count == prev_count:
            print(f"  Filesystem sync complete after {elapsed}s: {count} files accessible.", flush=True)
            return
        prev_count = count
        if time.time() >= deadline:
            print(
                f"  Warning: filesystem may not be fully synced after {timeout}s; proceeding anyway.",
                flush=True,
            )
            return
        print(f"  [{elapsed}s elapsed] {count} files accessible so far; rechecking in {check_interval}s...", flush=True)
        time.sleep(check_interval)


def get_dataset(dataset_repo, cache_dir, download_videos, predict_vids):
    """Return path to dataset, downloading from HuggingFace only if not cached.

    Runs the download in a subprocess watched by a progress monitor. If no bytes
    land for stall_timeout seconds the download is killed and retried, allowing
    arbitrarily large/slow downloads while still catching rate-limit hangs.
    """
    import multiprocessing
    import threading
    import time

    dataset_name = dataset_repo.split("/")[-1]
    cache_path = Path(cache_dir) / dataset_name

    if cache_path.exists():
        print(f"Using cached dataset at {cache_path}", flush=True)
        (cache_path / "videos").mkdir(exist_ok=True)
        return cache_path

    # videos-for-each-labeled-frame is only needed for post-hoc smoothing, never for training
    ignore_patterns = ["videos-for-each-labeled-frame/*"]
    if not download_videos:
        ignore_patterns.append("videos/*")
    if not predict_vids:
        ignore_patterns.append("videos_test/*")

    cache_path.mkdir(parents=True, exist_ok=True)

    # download to local disk; .incomplete temp files on network storage cause FileNotFoundError
    tmp_dir = Path("/tmp") / f"hf_download_{dataset_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_repo} -> {cache_path}", flush=True)
    print(f"  ignore_patterns: {ignore_patterns}", flush=True)

    max_retries = 5
    stall_timeout = 5 * 60  # kill if no bytes received for 5 minutes
    retry_wait = 90          # pause before retrying to let rate limit window reset

    for attempt in range(1, max_retries + 1):
        p = multiprocessing.Process(
            target=_snapshot_worker,
            args=(dataset_repo, "dataset", str(cache_path), ignore_patterns, str(tmp_dir)),
        )
        p.start()

        watcher = threading.Thread(
            target=_watchdog, args=(p, tmp_dir, stall_timeout), daemon=True
        )
        watcher.start()
        p.join()

        if p.exitcode == 0:
            break  # success

        msg = "stalled" if not p.is_alive() else f"exited with code {p.exitcode}"
        if attempt == max_retries:
            raise RuntimeError(f"Failed to download {dataset_repo} after {max_retries} attempts ({msg})")

        print(f"  Download attempt {attempt}/{max_retries} {msg}; retrying in {retry_wait}s...", flush=True)
        time.sleep(retry_wait)

    # LP asserts video_dir exists even when not using videos
    (cache_path / "videos").mkdir(exist_ok=True)

    _wait_for_filesystem_sync(cache_path)

    return cache_path


def main():
    args = parse_args()
    dataset_name = args.dataset_repo.split("/")[-1]

    # -------------------------------------------------------------------------
    # 1. Get dataset (from cache or HuggingFace)
    # -------------------------------------------------------------------------
    data_dir = get_dataset(args.dataset_repo, args.dataset_cache_dir, args.download_videos, args.predict_vids)

    # -------------------------------------------------------------------------
    # 2. Build and run litpose train
    # -------------------------------------------------------------------------
    config_file = str(data_dir / f"config_{dataset_name}.yaml")

    losses = [l for l in args.losses_to_use.split(",") if l]
    losses_hydra = f"[{','.join(losses)}]"

    # ViT backbones need a lower learning rate
    lr = 5e-5 if "vit" in args.backbone else 1e-3

    overrides = [
        f"data.data_dir={data_dir}",
        f"model.model_type={args.model_type}",
        f"model.backbone={args.backbone}",
        f"model.losses_to_use={losses_hydra}",
        f"training.train_frames={args.train_frames}",
        f"training.rng_seed_data_pt={args.seed}",
        f"training.optimizer_params.learning_rate={lr}",
        f"eval.predict_vids_after_training={'true' if args.predict_vids else 'false'}",
    ]

    # vitb_sam OOMs above batch size 16 on T4
    if "vitb_sam" in args.backbone:
        overrides.append("training.train_batch_size=16")

    if args.debug:
        overrides += [
            "training.check_val_every_n_epoch=1",
            "training.max_epochs=3",
            "training.unfreezing_epoch=1",
            "eval.predict_vids_after_training=false",
        ]

    cmd = ["litpose", "train", config_file, "--output_dir", args.output_dir, "--overrides"] + overrides
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # -------------------------------------------------------------------------
    # 3. Delete model weights to save storage
    # -------------------------------------------------------------------------
    n_deleted = 0
    for ckpt in Path(args.output_dir).rglob("*.ckpt"):
        ckpt.unlink()
        n_deleted += 1
    print(f"Deleted {n_deleted} checkpoint file(s)", flush=True)
    print(f"Done. Results at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
