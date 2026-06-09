"""
Worker script: runs inside a single Lightning AI job.

Downloads a dataset from HuggingFace, trains one Lightning Pose model
via `litpose train`, deletes model weights, and cleans up the dataset.
Results land in --output_dir (a path on teamspace storage).

Can also be called directly for local sweeps (no Lightning AI required):
    python run_single_job.py --dataset_repo paninski-lab/mirror-mouse-fused \\
        --backbone resnet50_animal_ap10k --train_frames 1 --seed 0 \\
        --output_dir /some/local/path
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# must be set before huggingface_hub is imported
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_repo", required=True,
        help="HuggingFace repo ID, e.g. paninski-lab/mirror-mouse-fused",
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


def main():
    args = parse_args()
    dataset_name = args.dataset_repo.split("/")[-1]
    data_dir = f"/tmp/datasets/{dataset_name}"

    # -------------------------------------------------------------------------
    # 1. Download dataset from HuggingFace
    # -------------------------------------------------------------------------
    from huggingface_hub import snapshot_download

    # videos-for-each-labeled-frame is only needed for post-hoc smoothing, never for training
    ignore_patterns = ["videos-for-each-labeled-frame*"]
    if not args.download_videos:
        ignore_patterns.append("videos/*")
    if not args.predict_vids:
        ignore_patterns.append("videos_test/*")

    print(f"Downloading {args.dataset_repo} -> {data_dir}")
    print(f"  ignore_patterns: {ignore_patterns}")
    snapshot_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        local_dir=data_dir,
        ignore_patterns=ignore_patterns,
    )

    # -------------------------------------------------------------------------
    # 2. Build and run litpose train
    # -------------------------------------------------------------------------
    config_file = str(Path(data_dir) / f"config_{dataset_name}.yaml")

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
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # -------------------------------------------------------------------------
    # 3. Delete model weights to save storage
    # -------------------------------------------------------------------------
    n_deleted = 0
    for ckpt in Path(args.output_dir).rglob("*.ckpt"):
        ckpt.unlink()
        n_deleted += 1
    print(f"Deleted {n_deleted} checkpoint file(s)")

    # -------------------------------------------------------------------------
    # 4. Clean up downloaded dataset
    # -------------------------------------------------------------------------
    shutil.rmtree(data_dir, ignore_errors=True)
    print(f"Done. Results at {args.output_dir}")


if __name__ == "__main__":
    main()
