"""Remap command for the lightning-pose CLI."""

from pathlib import Path
from textwrap import dedent


def register_parser(subparsers):
    """Register the remap command parser."""
    remap_parser = subparsers.add_parser(
        "remap",
        description=dedent(
            """\
            Remaps predictions from cropped to original coordinate space.
            Requires model predictions to already have been generated using `litpose predict`.

            Remapped predictions are saved as "remapped_{preds_file}" in the same folder as
            preds_file.
            """
        ),
        usage="litpose remap <preds_file> <bbox_file>",
    )
    remap_parser.add_argument("preds_file", type=Path, help="path to a prediction file")
    remap_parser.add_argument("bbox_file", type=Path, help="path to a bbox file")


def handle(args):
    """Handle the remap command."""
    import lightning_pose.utils.cropzoom as cz

    output_file = args.preds_file.with_name("remapped_" + args.preds_file.name)

    cz.generate_cropped_csv_file(
        input_csv_file=args.preds_file,
        input_bbox_file=args.bbox_file,
        output_csv_file=output_file,
        mode="add",
    )
