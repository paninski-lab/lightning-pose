"""
Orchestrate a Lightning Pose hyperparameter sweep on Lightning AI.

Run from within a Lightning AI studio:
    python run_sweep.py [--config sweep_config.yaml] [--dry_run]

Run from outside Lightning AI (set LIGHTNING_API_KEY env var first):
    LIGHTNING_API_KEY=<key> python run_sweep.py [--config sweep_config.yaml]

Output is written to:
    <output.base_dir>/<dataset>/<backbone>/<losses>/<tf{N}>/seed{N}/
"""

import argparse
import os
import time
from itertools import product
from pathlib import Path

import yaml

# must be set before huggingface_hub is imported
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def losses_str(losses: tuple) -> str:
    """Canonical directory name for a losses_to_use combination."""
    return "supervised" if not losses else "+".join(sorted(losses))


def sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(".", "_")


def dataset_shortname(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def make_job_name(dataset_repo, backbone, train_frames, seed, losses) -> str:
    return "__".join([
        sanitize(dataset_shortname(dataset_repo)),
        sanitize(backbone),
        losses_str(losses),
        f"tf{train_frames}",
        f"s{seed}",
    ])


def make_output_dir(base_dir, dataset_repo, backbone, train_frames, seed, losses) -> str:
    return str(
        Path(base_dir)
        / dataset_shortname(dataset_repo)
        / sanitize(backbone)
        / losses_str(losses)
        / f"tf{train_frames}"
        / f"seed{seed}"
    )


def make_worker_command(combo, cfg, worker_script) -> str:
    dataset_repo, backbone, train_frames, seed, losses = combo
    out_dir = make_output_dir(
        cfg["output"]["base_dir"], dataset_repo, backbone, train_frames, seed, losses
    )
    predict_vids = cfg["sweep"].get("predict_vids_after_training", False)
    download_videos = len(losses) > 0

    parts = [
        "python", str(worker_script),
        f"--dataset_repo={dataset_repo}",
        f"--dataset_cache_dir={cfg['output'].get('dataset_cache_dir', '/teamspace/lightning_storage/datasets')}",
        f"--backbone={backbone}",
        f"--train_frames={train_frames}",
        f"--seed={seed}",
        f"--losses_to_use={','.join(losses)}",
        f"--model_type={cfg['sweep'].get('model_type', 'heatmap')}",
        f"--output_dir={out_dir}",
    ]
    if download_videos:
        parts.append("--download_videos")
    if predict_vids:
        parts.append("--predict_vids")
    if cfg.get("debug", False):
        parts.append("--debug")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Launch a Lightning Pose hyperparameter sweep")
    parser.add_argument("--config", default="sweep_config.yaml", help="Path to sweep config YAML")
    parser.add_argument("--dry_run", action="store_true", help="Print jobs without launching")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip combos whose output directory already exists",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sweep = cfg["sweep"]
    li = cfg["lightning"]

    worker_script = Path(__file__).parent / "run_single_job.py"

    # build cartesian product; losses_to_use entries become tuples for hashing
    combos = list(product(
        sweep["datasets"],
        sweep["backbones"],
        sweep["train_frames"],
        sweep["seeds"],
        [tuple(l) for l in sweep.get("losses_to_use", [[]])],
    ))
    print(f"Total jobs: {len(combos)}")

    if args.skip_existing:
        base_dir = cfg["output"]["base_dir"]
        combos = [
            c for c in combos
            if not Path(make_output_dir(base_dir, *c)).exists()
        ]
        print(f"After skipping existing: {len(combos)} jobs remaining")

    if args.dry_run:
        print("\n--- Job list ---")
        for combo in combos:
            name = make_job_name(*combo)
            cmd = make_worker_command(combo, cfg, worker_script)
            print(f"\n{name}:\n  {cmd}")
        return

    # pre-download each unique dataset before launching jobs to avoid a race
    # condition where many jobs simultaneously attempt to populate an empty cache
    from run_single_job import get_dataset
    cache_dir = cfg["output"].get("dataset_cache_dir", "/teamspace/lightning_storage/datasets")
    download_videos = any(len(l) > 0 for l in sweep.get("losses_to_use", [[]]))
    predict_vids = sweep.get("predict_vids_after_training", False)
    for dataset_repo in set(sweep["datasets"]):
        get_dataset(dataset_repo, cache_dir, download_videos, predict_vids)

    from lightning_sdk import Job, Machine, Studio

    machine = getattr(Machine, li.get("machine", "T4_SMALL"))
    studio = Studio()

    jobs = {}
    for combo in combos:
        name = make_job_name(*combo)
        cmd = make_worker_command(combo, cfg, worker_script)
        print(f"Launching: {name}")
        job = Job.run(command=cmd, name=name, machine=machine, studio=studio)
        jobs[name] = job
        time.sleep(2)

    print(f"\nMonitoring {len(jobs)} jobs...")
    while True:
        statuses: dict[str, int] = {}
        for j in jobs.values():
            s = str(j.status)
            statuses[s] = statuses.get(s, 0) + 1
        print(f"  {statuses}")
        if not any(s in ("Running", "Pending") for s in statuses):
            break
        time.sleep(30)

    failed = [n for n, j in jobs.items() if str(j.status) == "Failed"]
    print(f"\nComplete: {len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print("Failed jobs:")
        for n in failed:
            print(f"  {n}")


if __name__ == "__main__":
    main()
