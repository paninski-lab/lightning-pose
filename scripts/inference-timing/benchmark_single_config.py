#!/usr/bin/env python3
"""Time a single (model, variant, decoder) inference configuration.

This script measures end-to-end video-prediction speed for exactly ONE
combination of model panel / precision-runtime "variant" / video-decoder
backend, and appends one CSV row per repeat to an output file.

It is intentionally scoped to a single combination and run as its own OS
process (see run_benchmark.sh): Lightning Pose has a known history of a rare
CUDA device-side assert corrupting the whole CUDA context (see upstream issue
#483 / PR #498). Running every combination in a single long-lived Python
process means one hard crash can silently invalidate every measurement taken
afterwards in that process. Running one process per combination means a
crash only takes down the one row we were trying to collect; the
orchestrating shell script logs it and moves on to the next combination in a
fresh process.

Usage (see run_benchmark.sh for how this is normally invoked):

    python benchmark_single_config.py \\
        --model_dir /path/to/model_dir \\
        --model_label resnet50 \\
        --video_paths /path/to/vid.mp4 \\
        --variant eager_fp32 \\
        --decoder dali \\
        --num_repeats 3 \\
        --num_warmup 1 \\
        --output_csv results/resnet50.csv

For a multiview model, pass --video_paths as a comma-separated list of one
video per view, in the same order the model's config expects.
"""
import argparse
import csv
import os
import sys
import time
import traceback

VARIANT_CHOICES = ["eager_fp32", "eager_fp16", "compile_fp16", "onnx_fp16", "tensorrt_fp16"]
DECODER_CHOICES = ["dali", "pynvvc", "opencv"]

CSV_FIELDS = [
    "gpu_label",
    "model_label",
    "variant",
    "decoder",
    "multiview",
    "run_idx",
    "is_warmup",
    "num_frames",
    "elapsed_s",
    "fps",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model_dir",
        required=True,
        help="Path to a trained Lightning Pose model directory.",
    )
    p.add_argument(
        "--model_label",
        required=True,
        help=(
            "Short label for this model panel, e.g. 'resnet50'. Used as a "
            "plotting group and in the output CSV."
        ),
    )
    p.add_argument(
        "--dataset_dir",
        default=None,
        help=(
            "Optional dataset dir override (data.data_dir / data.video_dir hydra "
            "overrides). Omit to use the model's own config as-is."
        ),
    )
    p.add_argument(
        "--video_paths",
        required=True,
        help=(
            "Comma-separated video file path(s) to run prediction on. One path "
            "for single-view models; one path per view, in config order, for "
            "multiview models."
        ),
    )
    p.add_argument("--variant", required=True, choices=VARIANT_CHOICES)
    p.add_argument(
        "--decoder",
        required=True,
        choices=DECODER_CHOICES,
        help=(
            "Video reader/decoder backend passed as the `reader=` kwarg to "
            "predict_on_video_file(_multiview)."
        ),
    )
    p.add_argument(
        "--num_repeats",
        type=int,
        default=3,
        help="Number of timed repeats (rows) to record, after warmup.",
    )
    p.add_argument(
        "--num_warmup",
        type=int,
        default=1,
        help=(
            "Number of untimed warmup runs before timed repeats. First-ever run "
            "through a fresh export/compile path is often much slower than "
            "steady-state, so warmup runs are excluded from the CSV by default "
            "unless --keep_warmup_rows is set."
        ),
    )
    p.add_argument(
        "--keep_warmup_rows",
        action="store_true",
        help=(
            "Also write warmup runs to the CSV (marked is_warmup=1) instead of "
            "discarding them."
        ),
    )
    p.add_argument(
        "--output_csv",
        required=True,
        help=(
            "CSV file to append results to. Created with a header if it doesn't "
            "exist yet; rows are appended if it does, so re-running a sweep "
            "with --skip_existing style bookkeeping in the caller is safe."
        ),
    )
    p.add_argument(
        "--gpu_label",
        default=None,
        help=(
            "Label for the GPU this was run on, e.g. 'L4' or 'A100'. If "
            "omitted, auto-detected from torch.cuda.get_device_name(0)."
        ),
    )
    p.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Passed through to model.export() for onnx/tensorrt variants.",
    )
    p.add_argument(
        "--opt_batch_size",
        type=int,
        default=1,
        help="Passed through to model.export() for onnx/tensorrt variants.",
    )
    return p.parse_args(argv)


def hydra_overrides_for(dataset_dir):
    overrides = ["++training.imgaug_hflip=false"]
    if dataset_dir is not None:
        overrides.append(f"data.data_dir={dataset_dir}")
        overrides.append(f"data.video_dir={dataset_dir}/videos")
    return overrides


def build_model(model_dir, variant, dataset_dir, max_batch_size, opt_batch_size):
    from lightning_pose.api import Model

    overrides = hydra_overrides_for(dataset_dir)

    if variant == "eager_fp32":
        model = Model.from_dir2(
            model_dir, precision="fp32", runtime="eager", hydra_overrides=overrides
        )
    elif variant == "eager_fp16":
        model = Model.from_dir2(
            model_dir, precision="fp16", runtime="eager", hydra_overrides=overrides
        )
    elif variant == "compile_fp16":
        model = Model.from_dir2(
            model_dir, precision="fp16", runtime="eager", hydra_overrides=overrides
        )
        model.compile()
    elif variant == "onnx_fp16":
        # export() is safe to call every time: it skips re-exporting if a
        # matching cached export already exists in the model dir.
        export_model = Model.from_dir2(
            model_dir, precision="fp16", runtime="eager", hydra_overrides=overrides
        )
        export_model.export(
            runtime="onnx",
            onnx_precision="fp16",
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )
        model = Model.from_dir2(
            model_dir, runtime="onnx", onnx_precision="fp16", hydra_overrides=overrides
        )
    elif variant == "tensorrt_fp16":
        export_model = Model.from_dir2(
            model_dir, precision="fp16", runtime="eager", hydra_overrides=overrides
        )
        export_model.export(
            runtime="tensorrt",
            onnx_precision="fp16",
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )
        model = Model.from_dir2(
            model_dir, runtime="tensorrt", onnx_precision="fp16", hydra_overrides=overrides
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    return model


def time_predict_runs(model, video_paths, decoder, multiview, num_repeats, num_warmup):
    import torch

    timings = []  # list of (elapsed_s, is_warmup)
    for run_idx in range(num_warmup + num_repeats):
        is_warmup = run_idx < num_warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if multiview:
            model.predict_on_video_file_multiview(
                video_paths, generate_labeled_video=False, reader=decoder
            )
        else:
            model.predict_on_video_file(
                video_file=video_paths[0], generate_labeled_video=False, reader=decoder
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timings.append((elapsed, is_warmup))
        kind = "warmup" if is_warmup else "timed"
        print(
            f"  [{kind} run {run_idx + 1}/{num_warmup + num_repeats}] {elapsed:.2f}s",
            flush=True,
        )
    return timings


def detect_gpu_label():
    import torch

    return torch.cuda.get_device_name(0)


def write_rows(output_csv, rows):
    file_exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    # Imported lazily, and from lightning_pose itself rather than
    # reimplemented here, so this script can't drift out of sync with the
    # rest of the repo's frame-counting logic.
    from lightning_pose.data.utils import count_frames

    args = parse_args()
    video_paths = [v.strip() for v in args.video_paths.split(",") if v.strip()]
    if not video_paths:
        raise ValueError("--video_paths did not contain any paths")

    gpu_label = args.gpu_label or detect_gpu_label()

    print(
        f"=== {args.model_label} | {args.variant} | {args.decoder} | GPU={gpu_label} ===",
        flush=True,
    )
    print(f"model_dir={args.model_dir}", flush=True)
    print(f"video_paths={video_paths}", flush=True)

    model = build_model(
        args.model_dir,
        args.variant,
        args.dataset_dir,
        args.max_batch_size,
        args.opt_batch_size,
    )
    multiview = bool(model.config.is_multi_view())

    if multiview and len(video_paths) < 2:
        print(
            "WARNING: model reports multiview but only 1 video path was given; "
            "using it for all views is NOT what you want. Check --video_paths.",
            flush=True,
        )

    num_frames = count_frames(video_paths[0])

    timings = time_predict_runs(
        model, video_paths, args.decoder, multiview, args.num_repeats, args.num_warmup
    )

    rows = []
    run_idx = 0
    for elapsed, is_warmup in timings:
        if is_warmup and not args.keep_warmup_rows:
            run_idx += 1
            continue
        fps = num_frames / elapsed if elapsed > 0 else float("nan")
        rows.append({
            "gpu_label": gpu_label,
            "model_label": args.model_label,
            "variant": args.variant,
            "decoder": args.decoder,
            "multiview": int(multiview),
            "run_idx": run_idx,
            "is_warmup": int(is_warmup),
            "num_frames": num_frames,
            "elapsed_s": f"{elapsed:.4f}",
            "fps": f"{fps:.3f}",
        })
        run_idx += 1

    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Fail loudly and exit non-zero so the orchestrating shell script can
        # detect and log this combination as failed, then move on to the next
        # one in a fresh process, rather than silently producing no output.
        traceback.print_exc()
        sys.exit(1)
