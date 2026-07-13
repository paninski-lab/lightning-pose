"""
Worker script: runs inside a single Lightning AI job.

Downloads a dataset's zip archive from HuggingFace, extracts it to local (ephemeral)
studio disk, trains one Lightning Pose model via `litpose train`, and deletes model
weights. Results land in --output_dir (a path on teamspace storage).

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

DEFAULT_LOCAL_DATA_DIR = "/tmp/lp_data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_repo", required=True,
        help="HuggingFace repo ID, e.g. paninski-lab/mirror-mouse-fused",
    )
    p.add_argument(
        "--local_data_dir", default=DEFAULT_LOCAL_DATA_DIR,
        help="Local (ephemeral) directory to download and extract dataset zips into; "
             "each dataset gets its own <local_data_dir>/<dataset_name> subdirectory",
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
        "--predict_vids", action="store_true",
        help="Run video prediction after training",
    )
    p.add_argument("--debug", action="store_true", help="Short smoke-test run (3 epochs)")
    return p.parse_args()


def get_dataset(dataset_repo, local_data_dir):
    """Download a dataset's zip archive from HuggingFace and extract it to local disk.

    Every job downloads and extracts its own copy fresh, rather than sharing a cache
    on teamspace storage. A single zip file is small enough that this is fast and
    sidesteps the rate-limit and distributed-filesystem sync issues that came from
    downloading thousands of individual files via snapshot_download onto network
    storage.

    The dataset is extracted to <local_data_dir>/<dataset_name>, so multiple datasets
    can coexist under the same local_data_dir without colliding.
    """
    import time
    import zipfile

    from huggingface_hub import hf_hub_download

    dataset_name = dataset_repo.split("/")[-1]
    zip_filename = f"{dataset_name}.zip"
    local_data_dir = Path(local_data_dir)
    extract_dir = local_data_dir / dataset_name

    print(f"Downloading {zip_filename} from {dataset_repo}...", flush=True)
    max_retries = 3
    retry_wait = 30
    zip_path = None
    for attempt in range(1, max_retries + 1):
        try:
            zip_path = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=zip_filename,
                local_dir=str(local_data_dir),
            )
            break
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {zip_filename} from {dataset_repo} "
                    f"after {max_retries} attempts"
                ) from e
            print(
                f"  Download attempt {attempt}/{max_retries} failed ({e}); "
                f"retrying in {retry_wait}s...",
                flush=True,
            )
            time.sleep(retry_wait)

    print(f"Extracting {zip_path} -> {extract_dir}", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    Path(zip_path).unlink()

    # LP asserts video_dir exists even for datasets with no video files
    (extract_dir / "videos").mkdir(exist_ok=True)

    return extract_dir


def main():
    args = parse_args()
    dataset_name = args.dataset_repo.split("/")[-1]

    # -------------------------------------------------------------------------
    # 1. Get dataset (download + extract zip from HuggingFace)
    # -------------------------------------------------------------------------
    data_dir = get_dataset(args.dataset_repo, args.local_data_dir)

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
        f"data.video_dir={data_dir}/videos",
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
