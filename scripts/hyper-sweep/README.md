# Lightning Pose Hyperparameter Sweep

Scripts for running parallel Lightning Pose training sweeps, either on
[Lightning AI](https://lightning.ai) or locally.

## How it works

`run_sweep.py` builds the cartesian product of all sweep dimensions defined in
`sweep_config.yaml` and launches one Lightning AI Job per combination.
Each job runs `run_single_job.py`, which:

1. Downloads the dataset from HuggingFace (selectively — videos only when needed)
2. Calls `litpose train` with the appropriate config and overrides
3. Deletes model weights (`.ckpt`) to save storage
4. Cleans up the downloaded dataset

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

> **Note:** `run_sweep.py` must be run from the `hyper-sweep/` directory so it
> can import `run_single_job.py`. If you need to run it from elsewhere, add
> `hyper-sweep/` to your `PYTHONPATH` first.

### From within a studio

```bash
cd lightning-pose/scripts/hyper-sweep
python run_sweep.py --config sweep_config.yaml
```

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
`/teamspace/lightning_storage/datasets`). Before launching any jobs,
`run_sweep.py` pre-downloads each unique dataset into the cache. This means
the download happens exactly once regardless of sweep size, and all jobs find
the cache already populated — avoiding both HuggingFace rate limits and race
conditions from simultaneous first-time downloads.

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

## Dependencies

```
lightning-pose    # must be installed; provides the litpose CLI
lightning_sdk     # pip install lightning_sdk  (orchestrator only)
huggingface_hub   # pip install huggingface_hub (worker)
pyyaml            # pip install pyyaml (orchestrator)
```

High-performance downloads are enabled automatically via `HF_XET_HIGH_PERFORMANCE=1`,
set at the top of `run_single_job.py` before `huggingface_hub` is imported.

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
