# Lightning Pose Hyperparameter Sweep

Scripts for running parallel Lightning Pose training sweeps, either on
[Lightning AI](https://lightning.ai) or locally.

## How it works

`run_sweep.py` builds the cartesian product of all sweep dimensions defined in
`sweep_config.yaml`. Before launching any jobs, it pre-downloads each unique
dataset into a shared teamspace cache — so the download happens exactly once,
regardless of how many jobs share that dataset. It then launches one Lightning AI
Job per combination. Each job runs `run_single_job.py`, which:

1. Uses the pre-cached dataset on teamspace (skips downloading if already present)
2. Calls `litpose train` with the appropriate config and overrides
3. Deletes model weights (`.ckpt`) to save storage

Results are written to teamspace storage and persist after each job ends.

## Output structure

```
<output.base_dir>/
└── <dataset>/
    └── <backbone>/
        └── <losses>/          # "supervised", "temporal", "pca_singleview+temporal", …
            └── tf<N>/         # train_frames
                └── seed<N>/   # rng seed
                    └── <LP training outputs, no .ckpt files>
```

## Running on Lightning AI

### Studio setup

Create a new studio, then install Lightning Pose from source following
[Option B in the installation guide](https://lightning-pose.readthedocs.io/en/latest/source/installation_guide.html#option-b-installation-from-source-development).
Also install `ffmpeg` as described in the docs. No conda environment is needed —
the studio is a self-contained environment.

### Teamspace storage setup

Sweep results and cached datasets are written to Lightning's managed teamspace
storage. Create the required folders once before running any sweeps:

1. In your teamspace, click the **Drive** tab (next to Studios)
2. Click **New Folder** and create a folder named `datasets`
3. Click **New Folder** again and create a folder named `sweep-results`

These names match the default paths in `sweep_config.yaml`, so no config edits
are needed.

### From within a studio

> **Note:** `run_sweep.py` must be run from the `hyper-sweep/` directory so it
> can import `run_single_job.py`. If you need to run it from elsewhere, add
> `hyper-sweep/` to your `PYTHONPATH` first.

```bash
cd lightning-pose/scripts/hyper-sweep
python run_sweep.py --config sweep_config.yaml
```

### Monitoring jobs

Once launched, each job appears in the **Jobs** panel. You can also click the
🚀 widget on the right-hand side of the studio — it shows a count of running
jobs and lets you inspect machine stats and logs for each one.

### From outside Lightning AI

Set your API key, then run the same command:

```bash
export LIGHTNING_API_KEY=<your_key>
python run_sweep.py --config sweep_config.yaml
```

The script uses `Studio()` to attach to the currently active studio.

### Dry run

Print all job names and commands without launching anything:

```bash
python run_sweep.py --dry_run
```

### Skip existing runs

Skip any combination whose output directory already exists (useful for
resuming a partially completed sweep):

```bash
python run_sweep.py --skip_existing
```

## Configuration

Before editing, copy `sweep_config.yaml` outside the LP repo so your changes
are not committed:

```bash
cp scripts/hyper-sweep/sweep_config.yaml /path/to/your/configs/sweep_config.yaml
```

Then pass your copy to the script:

```bash
python run_sweep.py --config /path/to/your/configs/sweep_config.yaml
```

Key fields:

| Field | Description |
|-------|-------------|
| `sweep.datasets` | HuggingFace repo IDs to train on |
| `sweep.backbones` | LP backbone names |
| `sweep.train_frames` | Labeled frame counts |
| `sweep.seeds` | RNG seeds (`training.rng_seed_data_pt`) |
| `sweep.losses_to_use` | List of loss combinations (each entry is a list) |
| `sweep.model_type` | `heatmap`, `context`, or `regression` |
| `sweep.predict_vids_after_training` | Run video prediction after training |
| `lightning.machine` | GPU type (`T4_SMALL`, `T4`, `A10G`, `A100`) |
| `output.base_dir` | Root path for results on teamspace storage |
| `debug` | `true` for a 3-epoch smoke test |

### Dataset caching

Downloaded datasets are cached in `output.dataset_cache_dir` (default:
`/teamspace/lightning_storage/datasets`), avoiding HuggingFace rate limits and
race conditions from simultaneous first-time downloads. A `.download_complete`
sentinel file is written only after the dataset is fully copied to the cache
directory. If a download fails partway through (e.g. after exhausting all
retries), the sentinel is absent and rerunning the sweep will resume the
download rather than skipping it. In-progress files are staged in
`/tmp/hf_download_<dataset>` and reused across runs, so HuggingFace will pick
up from where it left off rather than starting over.

For local runs, set `output.dataset_cache_dir` in `sweep_config.yaml` to a
local path, or pass `--dataset_cache_dir /your/path` directly to
`run_single_job.py`.

### Video download gating

To avoid downloading large video files unnecessarily:

- `videos/` (InD) — only downloaded when `losses_to_use` contains at least one
  unsupervised loss (i.e., any non-empty entry in the list)
- `videos_test/` (OOD) — only downloaded when `predict_vids_after_training: true`
- `videos-for-each-labeled-frame/` — never downloaded (used for post-hoc smoothing only)

## Running locally (no Lightning AI)

Call `run_single_job.py` directly. Example:

```bash
python run_single_job.py \
    --dataset_repo paninski-lab/mirror-mouse-fused \
    --backbone resnet50_animal_ap10k \
    --train_frames 1 \
    --seed 0 \
    --output_dir /path/to/results/mirror-mouse-fused/resnet50_animal_ap10k/supervised/tf1/seed0
```

To run a local sweep, call `run_single_job.py` in a loop or adapt
`run_sweep.py` to call it via `subprocess` instead of `Job.run()`.

## Plotting results

`plot_results.ipynb` visualizes pixel error vs. ensemble std dev for a completed
sweep. Before using it, **copy the notebook outside the LP repo** so that any
edits (dataset selection, colors, axis limits) are not committed back to the
repository:

```bash
cp scripts/hyper-sweep/plot_results.ipynb /path/to/your/notebooks/
```

Then install the one extra dependency:

```bash
pip install seaborn
```

Open the copy, edit the variables at the top of the configuration cell
(`base_dir`, `dataset`, `backbones`, `losses`, `train_frames`, `seeds`), and
run all cells.
