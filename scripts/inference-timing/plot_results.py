#!/usr/bin/env python3
"""Plot inference-speed results produced by run_benchmark.sh / benchmark_single_config.py.

Globs every *.csv in --input_dir (one CSV per model panel, as written by
run_benchmark.sh) and draws one subplot per panel: grouped bars of fps,
grouped by variant, with one bar per decoder actually present in that
panel's data. This is deliberately NOT hardcoded to a fixed number of panels
or a fixed set of variants/decoders -- it plots whatever combinations of
data it actually finds, so it works for any model panel/variant/decoder
selection a user's config produced (unlike some of the earlier one-off
regen scripts in this project, which hardcoded exactly 3 panels and fixed
per-GPU filename maps).

Usage:
    python plot_results.py --input_dir results/ --output results/inference_timing.png
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Preferred variant ordering left-to-right within each panel, when present.
VARIANT_ORDER = ["eager_fp32", "eager_fp16", "compile_fp16", "onnx_fp16", "tensorrt_fp16"]

# Consistent decoder colors, reused from earlier plots in this project.
DECODER_COLORS = {
    "dali": "#4B2E83",
    "pynvvc": "#B19CD9",
    "opencv": "#888888",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing one *.csv per model panel (as written by "
            "run_benchmark.sh)."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output image path, e.g. results/inference_timing.png",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Overall figure title. Defaults to the GPU label(s) found in the data.",
    )
    return p.parse_args(argv)


def load_panels(input_dir):
    csv_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    panels = {}  # label -> DataFrame
    for path in csv_paths:
        label = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        if df.empty:
            continue
        panels[label] = df
    return panels


def ordered_variants(present):
    ordered = [v for v in VARIANT_ORDER if v in present]
    extra = [v for v in present if v not in VARIANT_ORDER]
    return ordered + sorted(extra)


def plot_panel(ax, df, label):
    variants = ordered_variants(sorted(df["variant"].unique()))
    decoders = sorted(
        df["decoder"].unique(),
        key=lambda d: list(DECODER_COLORS).index(d) if d in DECODER_COLORS else 99,
    )

    # mean fps per (variant, decoder), across repeats (is_warmup rows are
    # excluded by benchmark_single_config.py unless --keep_warmup_rows was
    # passed, in which case drop them here too).
    if "is_warmup" in df.columns:
        df = df[df["is_warmup"] == 0]
    means = df.groupby(["variant", "decoder"])["fps"].mean()

    n_variants = len(variants)
    n_decoders = len(decoders)
    bar_width = 0.8 / max(n_decoders, 1)
    x = range(n_variants)

    for i, decoder in enumerate(decoders):
        heights = []
        for v in variants:
            heights.append(means.get((v, decoder), float("nan")))
        offsets = [xi + (i - (n_decoders - 1) / 2) * bar_width for xi in x]
        color = DECODER_COLORS.get(decoder, None)
        ax.bar(offsets, heights, width=bar_width, label=decoder, color=color)
        for xi, h in zip(offsets, heights, strict=True):
            if h != h:  # NaN check
                ax.text(
                    xi, 0, "N/A", ha="center", va="bottom", rotation=90, fontsize=7, color="gray"
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel("fps")
    ax.set_title(label)


def main(argv=None):
    args = parse_args(argv)
    panels = load_panels(args.input_dir)
    if not panels:
        raise SystemExit(f"No non-empty *.csv files found in {args.input_dir}")

    labels = sorted(panels.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    gpu_labels = set()
    for _label, df in panels.items():
        if "gpu_label" in df.columns:
            gpu_labels.update(df["gpu_label"].dropna().unique().tolist())

    for ax, label in zip(axes, labels, strict=True):
        plot_panel(ax, panels[label], label)

    # Single shared legend for decoder colors.
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 1.05),
        )

    if args.title:
        title = args.title
    elif gpu_labels:
        title = f"Inference speed — {', '.join(sorted(gpu_labels))}"
    else:
        title = "Inference speed"
    fig.suptitle(title, y=1.1)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
