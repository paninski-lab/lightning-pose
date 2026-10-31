"""Bounding-box sizing arguments shared by ``litpose create_bbox`` and ``litpose predict``.

Both commands turn detector keypoints into per-frame boxes with the same three options
(``--crop_ratio``, ``--crop_size``, ``--anchor_keypoints``); this module owns their
definition and validation so the two commands cannot drift apart.
"""

import argparse
from typing import Any, TypedDict


class BboxSizing(TypedDict, total=False):
    """Resolved bbox size: either ``crop_ratio`` or both ``crop_height`` and ``crop_width``."""

    crop_ratio: float
    crop_height: int
    crop_width: int


def add_bbox_sizing_args(parser: argparse.ArgumentParser | Any) -> None:
    """Add ``--crop_ratio``, ``--crop_size`` and ``--anchor_keypoints`` to a parser or group.

    Args:
        parser: an ``ArgumentParser`` or argument group to add the options to.
    """
    parser.add_argument(
        '--crop_ratio',
        type=float,
        default=None,
        help=(
            'size the bounding box this many times larger than the animal keypoint span'
            ' (default 2.0 when neither --crop_ratio nor --crop_size is given).'
            ' Mutually exclusive with --crop_size.'
        ),
    )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=None,
        help=(
            'fixed square bounding box side length in pixels, centred on the per-frame mean'
            ' of the anchor keypoints. Mutually exclusive with --crop_ratio.'
        ),
    )
    parser.add_argument(
        '--anchor_keypoints',
        type=str,
        default='',
        help='comma-separated list of anchor keypoint names; defaults to all keypoints',
    )


def resolve_bbox_sizing(
    crop_ratio: float | None,
    crop_size: int | None,
    anchor_keypoints: str,
) -> tuple[BboxSizing, list[str]]:
    """Validate the bbox sizing options and resolve defaults.

    Args:
        crop_ratio: value of ``--crop_ratio``, or None if not given.
        crop_size: value of ``--crop_size``, or None if not given.
        anchor_keypoints: value of ``--anchor_keypoints`` (comma-separated, may be empty).

    Returns:
        Tuple of (sizing, anchor keypoint names). ``sizing`` has ``crop_ratio`` or both
        ``crop_height`` and ``crop_width`` set, ready to be splatted into keyword arguments
        (e.g. of ``CropZoomPipeline.from_dirs``); the names list is empty for "all keypoints".

    Raises:
        ValueError: if both options are given, ``crop_size`` is not positive, or
            ``crop_ratio`` is not greater than 1.
    """
    if crop_ratio is not None and crop_size is not None:
        raise ValueError('--crop_ratio and --crop_size are mutually exclusive.')
    if crop_ratio is None and crop_size is None:
        crop_ratio = 2.0

    names = anchor_keypoints.split(',') if anchor_keypoints else []

    if crop_size is not None:
        if crop_size <= 0:
            raise ValueError(f'--crop_size must be a positive integer, got {crop_size}.')
        return BboxSizing(crop_height=crop_size, crop_width=crop_size), names
    assert crop_ratio is not None
    if crop_ratio <= 1:
        raise ValueError(f'--crop_ratio must be greater than 1, got {crop_ratio}.')
    return BboxSizing(crop_ratio=crop_ratio), names
