#!/usr/bin/env python3
"""Parse timing_config.yaml and print shell variable assignments for run_benchmark.sh.

Keeps the config file itself in YAML, matching the convention used by
scripts/hyper-sweep/sweep_config.yaml, while keeping run_benchmark.sh a bash
orchestrator: each (panel, variant, decoder) combination still runs
benchmark_single_config.py as its own subprocess, so a crash in one
combination can't corrupt the timing already collected for the others.

run_benchmark.sh calls this and evals the output:

    eval "$(python3 load_config.py --config timing_config.yaml)"
"""
import argparse
import os
import shlex

import yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to a timing_config.yaml file.")
    return p.parse_args(argv)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def panel_lines(panels):
    """Render each panel as the pipe-delimited `label|model_dir|dataset_dir|video_paths`
    string run_benchmark.sh's per-combination loop already expects."""
    lines = []
    for panel in panels:
        label = panel["label"]
        model_dir = panel["model_dir"]
        dataset_dir = panel.get("dataset_dir") or ""
        video_paths = ",".join(panel["video_paths"])
        lines.append(f"{label}|{model_dir}|{dataset_dir}|{video_paths}")
    return lines


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    panels = panel_lines(cfg["panels"])
    variants = cfg["sweep"]["variants"]
    decoders = cfg["sweep"]["decoders"]
    timing = cfg.get("timing", {})
    output = cfg.get("output", {})
    export = cfg.get("export", {})

    print("MODEL_PANELS=(" + " ".join(shlex.quote(p) for p in panels) + ")")
    print("VARIANTS=(" + " ".join(shlex.quote(v) for v in variants) + ")")
    print("DECODERS=(" + " ".join(shlex.quote(d) for d in decoders) + ")")
    print(f"NUM_WARMUP={shlex.quote(str(timing.get('num_warmup', 1)))}")
    print(f"NUM_REPEATS={shlex.quote(str(timing.get('num_repeats', 3)))}")
    default_output_dir = os.path.expanduser("~/inference_timing_results")
    print(f"OUTPUT_DIR={shlex.quote(str(output.get('dir') or default_output_dir))}")
    print(f"GPU_LABEL={shlex.quote(str(output.get('gpu_label') or ''))}")
    print(f"MAX_BATCH_SIZE={shlex.quote(str(export.get('max_batch_size', 8)))}")
    print(f"OPT_BATCH_SIZE={shlex.quote(str(export.get('opt_batch_size', 1)))}")


if __name__ == "__main__":
    main()
