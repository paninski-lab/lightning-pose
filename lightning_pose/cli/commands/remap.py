"""Remap command for the lightning-pose CLI."""

from pathlib import Path
from textwrap import dedent


def register_parser(subparsers):
    """Register the remap command parser."""
    # Choose documentation link depending on whether we're being imported by Sphinx
    import sys

    is_building_docs = "sphinx" in sys.modules
    _doc_link = (
        ":doc:`user_guide_advanced/cropzoom_pipeline`"
        if is_building_docs
        else "https://lightning-pose.readthedocs.io/en/latest/source/user_guide_advanced/cropzoom_pipeline.html"
    )

    description_text = dedent(
        f"""\
            Remaps predictions from cropped to original coordinate space.
            Requires model predictions to already have been generated using ``litpose predict``.

            Remapped predictions are saved as ``remapped_{{preds_file}}`` in the same folder as
            preds_file.

            For an end-to-end usage example of the Cropzoom workflow, see the user guide:
            {_doc_link}.
            """
    )

    remap_parser = subparsers.add_parser(
        "remap",
        description=description_text,
        usage="litpose remap <preds_file> <bbox_file>",
    )
    remap_parser.add_argument("preds_file", type=Path, help="path to a prediction file")
    remap_parser.add_argument("bbox_file", type=Path, help="path to a bbox file")
    return remap_parser


def get_parser():
    """Return an ArgumentParser for the `litpose remap` subcommand (for docs)."""
    import argparse

    parser = argparse.ArgumentParser(prog="litpose")
    subparsers = parser.add_subparsers(dest="command")
    return register_parser(subparsers)


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
