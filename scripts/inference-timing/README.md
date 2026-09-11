# Inference timing benchmark

Reproduce the published Lightning Pose inference-speed figure — the one
comparing precision/runtime variants (eager fp32/fp16, `torch.compile`, ONNX
Runtime, TensorRT) and video-decoder backends (DALI vs PyNvVideoCodec) — on
**whatever single GPU you happen to have**. Useful both for reproducing the
docs figure and for benchmarking your own models/videos on your own
hardware.

This is the single-GPU sibling of `scripts/hyper-sweep/`: same overall
shape (a YAML config file you copy and edit, a worker script that does one
unit of work, an orchestrator script that loops over combinations, a
plotting script at the end), but built for timing inference on one machine
instead of launching a hyperparameter sweep across many Lightning jobs.

## Contents

- `benchmark_single_config.py` — worker script. Runs exactly ONE
  (model panel, precision/runtime variant, decoder) combination and appends
  timing rows to a CSV. Callable standalone for debugging a single
  combination.
- `run_benchmark.sh` — orchestrator. Reads a config file, loops over every
  (model panel x variant x decoder) combination, calling
  `benchmark_single_config.py` as a fresh process each time, then calls
  `plot_results.py` at the end.
- `load_config.py` — parses `timing_config.yaml` and prints the shell
  variable assignments `run_benchmark.sh` evals to drive its loop.
- `timing_config.yaml` — example config. Copy this outside the repo and
  edit your copy (same convention as `scripts/hyper-sweep/sweep_config.yaml`).
- `plot_results.py` — plotting script (plain `.py`, not a notebook, per
  request). Reads every `*.csv` in a results directory and draws one
  grouped-bar subplot per model panel.

Basic (GPU-free) tests for all of the above live in
`tests/scripts/inference_timing/`.

## Setup

You need a working Lightning Pose install with whichever of the following
you intend to benchmark: `torch.compile` support, ONNX Runtime (with a GPU
execution provider), TensorRT, NVIDIA DALI, and/or PyNvVideoCodec. You do
NOT need all of them — just skip the variants/decoders you can't install by
leaving them out of your config's `sweep.variants`/`sweep.decoders` lists.
The `opencv` decoder choice has no extra GPU-decode dependency and is a
useful first sanity check.

You also need at least one trained Lightning Pose model directory and at
least one video to run prediction on.

## Running

1. Copy the example config outside the repo and edit it to point at your
   model(s) and video(s):

   ```bash
   cp scripts/inference-timing/timing_config.yaml ~/my_timing_config.yaml
   # edit ~/my_timing_config.yaml
   ```

   See the comments in `timing_config.yaml` for the `panels` format (one
   entry per model/dataset you want a panel for) and for the
   `sweep`/`timing`/`output`/`export` knobs.

2. Do a dry run first to sanity-check the combinations it's about to run,
   without actually running anything:

   ```bash
   bash scripts/inference-timing/run_benchmark.sh --config ~/my_timing_config.yaml --dry_run
   ```

3. Run it for real:

   ```bash
   bash scripts/inference-timing/run_benchmark.sh --config ~/my_timing_config.yaml
   ```

   This writes one CSV per model panel plus a per-combination log file
   under `output.dir`, then generates `output.dir/inference_timing.png`.

   The GPU label used in the plot is auto-detected from
   `torch.cuda.get_device_name(0)` unless you set `output.gpu_label` in
   your config.

4. If you only want to (re)generate the plot from CSVs you already have
   (e.g. after manually re-running one failed combination), you can call
   the plotting script directly:

   ```bash
   python scripts/inference-timing/plot_results.py --input_dir ~/inference_timing_results --output ~/inference_timing_results/inference_timing.png
   ```

## Design notes

- **One process per combination.** Lightning Pose has a known history of a
  rare CUDA device-side assert that corrupts the whole CUDA context for the
  rest of the process (see upstream issue #483 and its fix, PR #498).
  Running each (model, variant, decoder) combination in its own fresh
  `python3` subprocess means a crash in one combination can't invalidate
  the timing already collected for others, and `run_benchmark.sh` logs the
  failure and continues to the next combination rather than aborting the
  whole sweep (it deliberately does not use `set -e` around the per-combo
  loop).
- **Config stays YAML, orchestration stays bash.** `load_config.py` is the
  only piece that knows about `timing_config.yaml`'s schema; it prints
  shell assignments that `run_benchmark.sh` evals, so the per-combination
  loop itself doesn't need to change.
- **Frame counts are auto-detected**, via `lightning_pose.data.utils.count_frames`,
  from whatever video you point the script at, rather than hardcoded — so
  this works with any video length, not just the one originally used to
  produce the published figure.
- **CSVs are appended to, not overwritten.** Re-running
  `run_benchmark.sh` against the same `output.dir` adds more rows rather
  than clobbering prior results; if you want a clean run, remove the old
  CSVs first.
- **The plot adapts to whatever data is present.** It doesn't assume a
  fixed number of model panels or a fixed set of variants/decoders — it
  draws one subplot per CSV it finds and one bar per decoder actually
  present in that CSV, marking any variant/decoder combination that's
  missing as "N/A" (matching the convention used in the published figure).

## Interpreting your results vs. the published figure

Exact numbers will differ from the published figure if you're on a
different GPU, a different model/video, or a different Lightning Pose
version — that's expected and is the point of this script. If you're trying
to reproduce the *published* figure specifically, use the same model
architecture/checkpoint and video the original figure used, and compare on
an L4 or A100 GPU (the two GPUs the published figure was benchmarked on).
