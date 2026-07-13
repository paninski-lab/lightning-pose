# Lightning Pose Hyperparameter Sweep

Scripts for running parallel Lightning Pose training sweeps, either on
[Lightning AI](https://lightning.ai) or locally.

## How it works

`run_sweep.py` builds the cartesian product of all sweep dimensions defined in
`sweep_config.yaml` and launches one Lightning AI Job per combination. Each job
runs `run_single_job.py`, which:

1. Downloads a single zip archive of the dataset from HuggingFace and extracts
   it to local (ephemeral) disk on the studio the job is running in
2. Calls `litpose train` with the appropriate config and overrides
3. Deletes model weights (`.ckpt`) to save storage

Results are written to teamspace storage and persist after each job ends.

Every job downloads its own copy of the dataset independently — there is no
shared cache. This trades a small amount of duplicate downloading (each job
fetches the same zip) for reliability: a single-file download is fast and
immune to the HuggingFace rate limits and distributed-filesystem sync issues
that come from downloading thousands of individual files onto shared teamspace
storage. See [Dataset zip archives](#dataset-zip-archives) for how to prepare
the zip file each dataset needs.

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

Sweep results are written to Lightning's managed teamspace storage. Create the
required folder once before running any sweeps:

1. In your teamspace, click the **Drive** tab (next to Studios)
2. Click **New Folder** and create a folder named `sweep-results`

This name matches the default `output.base_dir` in `sweep_config.yaml`, so no
config edits are needed. Datasets are no longer cached on teamspace storage —
each job downloads its own copy to local disk (see
[Dataset zip archives](#dataset-zip-archives)).

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
| `output.local_data_dir` | Local (ephemeral) dir each job extracts its dataset(s) into |
| `debug` | `true` for a 3-epoch smoke test |

## Dataset zip archives

Each job downloads a single zip file, named `<dataset_name>.zip` (e.g.
`mirror-mouse-fused.zip`), from the root of the dataset's HuggingFace repo. It
is extracted to `<output.local_data_dir>/<dataset_name>` (default
`/tmp/lp_data/<dataset_name>`) on the studio the job is running in — local
disk, not teamspace storage. Because extraction is namespaced by dataset name,
multiple datasets can coexist under the same `local_data_dir` without
colliding (useful for local runs that sweep over several datasets). No cleanup
of the extracted data is needed on Lightning AI: each Job runs in its own
fresh container, so the disk is reclaimed automatically when the job ends.

**This zip file is not created automatically** — it must be prepared and
uploaded to each dataset's HuggingFace repo once, before running a sweep
against that dataset. Since these published datasets are largely static, this
is a one-time (or rare) step per dataset, not something the sweep pipeline
does for you.

To prepare it, zip the *contents* of the dataset directory (not the directory
itself, so the zip has no wrapping folder — files like
`config_<dataset>.yaml` should sit at the top level of the archive):

```bash
cd /path/to/local/copy/of/mirror-mouse-fused
zip -r mirror-mouse-fused.zip .
```

Then upload it to the dataset's HuggingFace repo, e.g. with the
`huggingface_hub` Python API:

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="mirror-mouse-fused.zip",
    path_in_repo="mirror-mouse-fused.zip",
    repo_id="paninski-lab/mirror-mouse-fused",
    repo_type="dataset",
)
```

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
