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

Edit `sweep_config.yaml` before running. Key fields:

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
hf_transfer       # pip install hf_transfer    (worker; enables fast parallel downloads)
pyyaml            # pip install pyyaml (orchestrator)
```

`hf_transfer` is optional but strongly recommended — without it, HuggingFace downloads
are sequential and can be 10–20× slower. It is enabled automatically by `run_single_job.py`
via `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"`; the package just needs to be installed.
