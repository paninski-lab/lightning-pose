"""Export command for the lightning-pose CLI."""

from __future__ import annotations

import argparse
import logging
import textwrap
from typing import Any

from .. import types

logger = logging.getLogger(__name__)

# Export targets. TensorRT builds its engine from the same .onnx artifact
# "onnx" produces, through a different execution provider.
_RUNTIME_CHOICES = ("onnx", "tensorrt")

# Weight precisions the exporter supports. Deliberately narrower than
# --precision on `litpose predict`: this is the precision baked into the
# exported file, not the autocast precision used for eager inference. For
# --runtime tensorrt, this selects which existing ONNX export to build from.
_ONNX_PRECISION_CHOICES = ("fp32", "fp16")


def register_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the export command parser."""
    export_parser = subparsers.add_parser(
        "export",
        description=textwrap.dedent(
            """\
        Exports a trained model to an optimized inference format.

          Exports are written to a fixed location inside the model directory::

            <model_dir>/
            |-- exports_onnx/
            |   `-- <checkpoint_stem>_<onnx_precision>.onnx
            `-- exports_trt/
                `-- <checkpoint_stem>_<onnx_precision>/
                    `-- trt_metadata.json

          The export path is not configurable. `litpose predict --runtime onnx`
          or `--runtime tensorrt` reads from these same locations.

          TensorRT builds its engine from an existing ONNX export rather than
          re-tracing the model -- run `--runtime onnx` first for the same
          --onnx-precision.
        """
        ),
        usage="litpose export <model_dir> [OPTIONS]",
    )
    export_parser.add_argument(
        "model_dir", type=types.existing_model_dir, help="path to a model directory"
    )
    export_parser.add_argument(
        "--runtime",
        choices=sorted(_RUNTIME_CHOICES),
        default="onnx",
        help="inference format to export to. Default: onnx.",
    )
    export_parser.add_argument(
        "--onnx-precision",
        choices=sorted(_ONNX_PRECISION_CHOICES),
        default="fp16",
        help=(
            "weight precision baked into the exported file. Unlike "
            "`litpose predict --precision`, this changes the exported weights "
            "themselves rather than the precision of a forward pass. For "
            "--runtime tensorrt, selects which existing ONNX export to build "
            "the engine from. Default: fp16."
        ),
    )
    export_parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help=(
            "only used with --runtime tensorrt. Upper bound of the "
            "dynamic-shape batch profile the engine is built for; inference "
            "at a batch size above this fails. Default: 8."
        ),
    )
    export_parser.add_argument(
        "--opt-batch-size",
        type=int,
        default=None,
        help=(
            "only used with --runtime tensorrt. Batch size the engine is "
            "optimized for; defaults to --max-batch-size. Correctness holds "
            "for any batch size in [1, max-batch-size], but performance is "
            "best near this value."
        ),
    )
    return export_parser


def get_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser for the `litpose export` subcommand (for docs)."""
    parser = argparse.ArgumentParser(prog="litpose")
    subparsers = parser.add_subparsers(dest="command")
    return register_parser(subparsers)


def handle(args: argparse.Namespace) -> None:
    """Handle the export command."""
    # Delay this import because it's slow.
    from lightning_pose.api import Model

    model = Model.from_dir(args.model_dir)
    output_path = model.export(
        args.runtime,
        onnx_precision=args.onnx_precision,
        max_batch_size=args.max_batch_size,
        opt_batch_size=args.opt_batch_size,
    )
    logger.info(f'export written to {output_path}')
