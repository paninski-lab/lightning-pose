"""Path handling, label-file parsing, and checkpoint discovery.

Video-specific helpers live in :mod:`lightning_pose.utils.io_video`.
"""

from __future__ import annotations  # python 3.8 compatibility for sphinx

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)

# filenames of the per-label-CSV outputs, written to ``image_preds/<csv name>/``
PREDICTIONS_CSV_NAME = 'predictions.csv'
BBOX_CSV_NAME = 'bbox.csv'

# to ignore imports for sphinx-autoapidoc
__all__ = [
    "BBOX_CSV_NAME",
    "PREDICTIONS_CSV_NAME",
    "ckpt_path_from_base_path",
    "get_keypoint_names",
    "LabeledData",
    "parse_label_csv",
    "return_absolute_path",
    "resolve_path",
    "return_absolute_data_paths",
    "get_context_img_paths",
]


def ckpt_path_from_base_path(
    base_path: str,
    model_name: str,
    logging_dir_name: str = "tb_logs/",
) -> str | None:
    """Given a path to a hydra output with trained model, extract the model .ckpt file.

    Prioritizes the checkpoint marked with '-best.ckpt' in the latest version directory.
    If no 'best' checkpoint is found, falls back to the latest checkpoint (highest step count).
    If only one checkpoint exists, returns it directly.

    Args:
        base_path (str): path to a folder with logs and checkpoint. for example,
            function will search base_path/logging_dir_name/model_name...
        model_name (str): the name you gave your model before training it; appears as
            model_name in lightning-pose/scripts/config/model_params.yaml
        logging_dir_name (str, optional): name of the folder in logs, controlled in
            train_hydra.py Defaults to "tb_logs/".

    Returns:
        str: path to model checkpoint, or None if none found.

    """
    # Find all checkpoint files; model_name is part of the base (not the pattern), so any glob
    # characters in it are matched literally
    model_dir = Path(base_path) / logging_dir_name / model_name
    all_checkpoint_files = [
        str(f) for f in model_dir.glob('version_*/checkpoints/*.ckpt')
    ]
    if not all_checkpoint_files:
        return None

    # Group checkpoints by version
    ckpt_files_by_version = {}
    for f in all_checkpoint_files:
        match = re.search(r"version_(\d+)", f)
        if match:
            version = int(match.group(1))
            if version not in ckpt_files_by_version:
                ckpt_files_by_version[version] = []
            ckpt_files_by_version[version].append(f)

    if not ckpt_files_by_version:
        # Should not happen if all_checkpoint_files is not empty and pattern matches
        return None

    # Get the latest version's checkpoint files
    latest_version = max(ckpt_files_by_version.keys())
    latest_version_files = ckpt_files_by_version[latest_version]

    # Find all "best" checkpoints in the latest version
    best_ckpt_files = []
    for f in latest_version_files:
        if "-best.ckpt" in Path(f).name:
            best_ckpt_files.append(f)

    if len(best_ckpt_files) == 1:
        # Found exactly one 'best' checkpoint
        return best_ckpt_files[0]
    elif len(best_ckpt_files) > 1:
        # Multiple 'best' checkpoints found (e.g., stale file on a network filesystem).
        # Pick the one with the highest step count.
        logger.warning(
            f"Multiple 'best' checkpoint files found: {best_ckpt_files}. "
            "Selecting the one with the highest step count."
        )
        def _step(f):
            m = re.search(r"step=(\d+)", f)
            return int(m.group(1)) if m else -1

        best_ckpt_files.sort(key=_step)
        return best_ckpt_files[-1]
    else:
        # No 'best' checkpoint found
        logger.warning("No 'best' checkpoint found, falling back to latest checkpoint.")
        if len(latest_version_files) == 1:
            # Only one checkpoint file exists, return it
            return latest_version_files[0]
        else:
            # Multiple checkpoints exist, but none are marked 'best'.
            # Try to find the one with the highest step count.
            max_step = -1
            latest_ckpt = None

            for f in latest_version_files:
                match = re.search(r"step=(\d+)", f)
                if match:
                    step = int(match.group(1))
                    if step > max_step:
                        max_step = step
                        latest_ckpt = f

            if latest_ckpt is not None:
                return latest_ckpt
            else:
                # Could not determine which checkpoint to use
                raise ValueError(
                    "Multiple checkpoint files found but cannot determine which "
                    f"to use: {latest_version_files}. "
                    "None are marked as 'best' and cannot parse step counts to determine latest. "
                    "Please manually select the appropriate checkpoint."
                )


def get_keypoint_names(
    cfg: DictConfig | ListConfig | None = None,
    csv_file: str | None = None,
    header_rows: list | None = [0, 1, 2],
) -> list[str]:
    """Return keypoint names from a label CSV file or from the config.

    Args:
        cfg: hydra config; used as a fallback when ``csv_file`` is not provided or does not exist.
        csv_file: path to a labeled CSV file in DLC format; takes precedence over ``cfg``.
        header_rows: row indices in the CSV that form the MultiIndex header. Defaults to
            ``[0, 1, 2]`` (DLC format with scorer/bodypart/coords rows).

    Returns:
        List of keypoint name strings.
    """
    if csv_file is not None and Path(csv_file).exists():
        if header_rows is None:
            assert cfg is not None
            if "header_rows" in cfg.data:
                header_rows = list(cfg.data.header_rows)
            else:
                # assume dlc format
                header_rows = [0, 1, 2]
        # We're just reading to parse the column structure, so let's only
        # read a few rows (nrows=...). Unsure if this includes header rows.
        csv_data = pd.read_csv(csv_file, header=header_rows, nrows=5)
        # collect marker names from multiindex header
        if header_rows == [1, 2] or header_rows == [0, 1]:
            # self.keypoint_names = csv_data.columns.levels[0]
            # ^this returns a sorted list for some reason, don't want that
            keypoint_names = [b[0] for b in csv_data.columns if b[1] == "x"]
        elif header_rows == [0, 1, 2]:
            # self.keypoint_names = csv_data.columns.levels[1]
            keypoint_names = [b[1] for b in csv_data.columns if b[2] == "x"]
    else:
        assert cfg is not None, 'cfg must be provided when csv_file is not given'
        keypoint_names = [f"bp_{n}" for n in range(cfg.data.num_targets // 2)]
    return keypoint_names


@dataclass
class LabeledData:
    """Result of parsing a label CSV file.

    Attributes:
        keypoint_names: ordered list of keypoint name strings
        image_names: ordered list of image path strings (relative to the project root)
        keypoints: ``(N, K, 2)`` float tensor of ``(x, y)`` coordinates; NaN where unlabeled
        visibility: ``(N, K)`` int64 tensor of per-keypoint visibility flags (0, 1, or 2),
            or ``None`` when the CSV does not contain a ``visible`` column
    """

    keypoint_names: list[str]
    image_names: list[str]
    keypoints: Float[torch.Tensor, 'num_frames num_keypoints 2']
    visibility: Int[torch.Tensor, 'num_frames num_keypoints'] | None


def parse_label_csv(
    csv_file: str,
    header_rows: list[int] | None = None,
) -> LabeledData:
    """Parse a label CSV file into a :class:`LabeledData` object in a single read.

    Reads the file exactly once and returns keypoint names, image names, coordinates,
    and per-keypoint visibility flags.  When the file contains a ``visible`` column per
    keypoint, visibility flags are returned as an int64 tensor; otherwise ``visibility``
    is ``None``.

    This function is the single entry point for reading label files.  A future
    ``parse_label_json`` function for COCO ``.json`` format should return the same
    :class:`LabeledData` type, allowing callers to remain format-agnostic.

    Args:
        csv_file: path to the label CSV file
        header_rows: row indices that form the MultiIndex header; defaults to ``[0, 1, 2]``

    Returns:
        :class:`LabeledData` containing all label information

    Raises:
        FileNotFoundError: if ``csv_file`` does not exist
        ValueError: if any visibility value is outside ``{0, 1, 2}``, or if any keypoint has
            exactly one of its x, y coordinates missing (NaN)
    """
    if header_rows is None:
        header_rows = [0, 1, 2]
    if not Path(csv_file).exists():
        raise FileNotFoundError(f'could not find csv file at {csv_file}')

    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    csv_data = fix_empty_first_row(csv_data)

    # extract keypoint names from the MultiIndex header
    if header_rows in ([1, 2], [0, 1]):
        keypoint_names = [b[0] for b in csv_data.columns if b[1] == 'x']
    else:
        keypoint_names = [b[1] for b in csv_data.columns if b[2] == 'x']

    image_names = list(csv_data.index)

    # detect visibility column and split raw data accordingly
    has_vis = (
        header_rows == [0, 1, 2]
        and any(b[2] == 'visible' for b in csv_data.columns)
    )
    if has_vis:
        raw = torch.tensor(csv_data.to_numpy(), dtype=torch.float32)
        raw = raw.reshape(raw.shape[0], -1, 3)
        keypoints = raw[:, :, :2].contiguous()
        vis_float = raw[:, :, 2]
        valid_vals = {0.0, 1.0, 2.0}
        unique_vals = set(vis_float[~torch.isnan(vis_float)].unique().tolist())
        invalid_vals = unique_vals - valid_vals
        if invalid_vals:
            raise ValueError(
                f'visibility column contains invalid values {invalid_vals}; '
                'expected values in {{0, 1, 2}}'
            )
        visibility: Int[torch.Tensor, 'num_frames num_keypoints'] | None = vis_float.long()
    else:
        raw = torch.tensor(csv_data.to_numpy(), dtype=torch.float32)
        keypoints = raw.reshape(raw.shape[0], -1, 2)
        visibility = None

    # a keypoint is either fully labeled or fully missing; exactly one NaN coordinate is
    # a labeling/export error that would otherwise surface as a shape error mid-training
    nan_xy = torch.isnan(keypoints)
    idx_half_nan = nan_xy[:, :, 0] != nan_xy[:, :, 1]
    if idx_half_nan.any():
        examples = []
        for idx_img, idx_kp in idx_half_nan.nonzero()[:5].tolist():
            missing = 'x' if nan_xy[idx_img, idx_kp, 0] else 'y'
            examples.append(
                f'{image_names[idx_img]} / {keypoint_names[idx_kp]} ({missing} is NaN)'
            )
        raise ValueError(
            f'{csv_file} contains {int(idx_half_nan.sum())} keypoint(s) with only one of x, y '
            'missing; both coordinates must be present or both must be NaN. '
            f'First offending entries: {"; ".join(examples)}'
        )

    return LabeledData(
        keypoint_names=keypoint_names,
        image_names=image_names,
        keypoints=keypoints,
        visibility=visibility,
    )


# --------------------------------------------------------------------------------------
# Path handling functions for running toy dataset
# --------------------------------------------------------------------------------------


def return_absolute_path(possibly_relative_path: str, n_dirs_back: int = 3) -> str:
    """Return absolute path from possibly relative path."""
    path = Path(possibly_relative_path)
    if path.is_absolute():
        # absolute path already; do nothing
        abs_path = path
    else:
        # our toy_dataset in relative path
        # pathlib (not manual os.path.sep splitting/joining) -- splitting a Windows path
        # on the separator loses the drive letter's root marker (str.split gives 'C:', not
        # 'C:\\'), and os.path.join then silently drops the separator after it.
        base_dir = Path.cwd().parents[n_dirs_back - 1]
        if base_dir.name == "multirun":
            # hydra multirun, go one dir back
            base_dir = base_dir.parent
        abs_path = base_dir / possibly_relative_path
    if not abs_path.exists():
        raise OSError(f"{abs_path} is not a valid path")
    return str(abs_path)


def resolve_path(path: str | Path, root_directory: str | Path) -> Path:
    """Resolve a path that is relative to a project root directory, or absolute.

    Candidates are tried in order: ``root_directory / path`` (which is just ``path`` when it is
    absolute), then ``path`` as given (relative to the current working directory).

    Args:
        path: file path, relative to ``root_directory`` or absolute.
        root_directory: directory that relative paths are anchored to.

    Returns:
        path to the first existing candidate.

    Raises:
        FileNotFoundError: if no candidate exists; the message lists the locations searched.
    """
    candidates = [Path(root_directory) / path, Path(path)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ', '.join(str(c) for c in dict.fromkeys(candidates))
    raise FileNotFoundError(f'could not find {path}; looked in: {searched}')


def return_absolute_data_paths(
    data_cfg: DictConfig, n_dirs_back: int = 3
) -> tuple[str, str | None]:
    """Generate absolute path for our example toy data.

    @hydra.main decorator switches the cwd when executing the decorated function, e.g.,
    our train(). so we're in some /outputs/YYYY-MM-DD/HH-MM-SS folder.

    Args:
        data_cfg (DictConfig): data config file with paths to data and video folders.
        n_dirs_back (int):

    Returns:
        Tuple[str, str | None]: absolute paths to data and video folders. The video path is
        ``None`` if ``data.video_dir`` is unset; otherwise it is resolved but not validated,
        since it does not need to exist for purely supervised models; use
        :func:`~lightning_pose.utils.io_video.has_videos` or
        :func:`~lightning_pose.utils.io_video.require_videos` to check it.

    """
    data_dir = return_absolute_path(data_cfg.data_dir, n_dirs_back=n_dirs_back)
    # video_dir is optional (only semi-supervised models need it)
    video_dir_cfg = data_cfg.get('video_dir')
    video_dir = None
    if video_dir_cfg is not None:
        # an absolute video_dir overrides data_dir; a relative one is anchored to it
        video_dir = str(Path(data_dir) / video_dir_cfg)
    # only the data directory is required here; videos are needed solely for semi-supervised
    # training and are validated on demand by `require_videos`
    assert Path(data_dir).is_dir()
    return data_dir, video_dir


























def get_context_img_paths(center_img_path: Path) -> list[Path]:
    """Given the path to a center image frame, return paths of 5 context frames
    (n-2, n-1, n, n+1, n+2).

    Negative indices are floored at 0.
    """
    match = re.search(r"(\d+)", center_img_path.stem)
    assert match is not None, (
        f"No frame index in filename, can't get context frames: {center_img_path.name}"
    )

    center_index_string = match.group()
    center_index = int(center_index_string)

    context_img_paths = []
    for index in range(
        center_index - 2, center_index + 3
    ):  # End at n+3 exclusive, n+2 inclusive.
        # Negative indices are floored at 0.
        index = max(index, 0)

        # Add leading zeros to match center_index_string length.
        index_string = str(index).zfill(len(center_index_string))

        stem = center_img_path.stem.replace(center_index_string, index_string)
        path = center_img_path.with_stem(stem)

        context_img_paths.append(path)

    return context_img_paths


def fix_empty_first_row(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes a problem with `pd.read_csv` where if the first row is all NaN
    it gets dropped.

    Pandas uses the first row after a multiline header for the df.index.name.
    It would look just like a data row where all values are NaN. Pandas has no
    way to distinguish between an index name row, and a NaN data row.

    Pandas gh issue: https://github.com/pandas-dev/pandas/issues/21995

    Our fix detects if there's an index name, and if so it adds a NaN data row
    with index=df.index.name.
    """
    if df.index.name is not None:
        new_row = {col: np.nan for col in df.columns}
        prepend_df = pd.DataFrame(
            new_row,
            index=pd.Index([df.index.name]),
            columns=df.columns,
            dtype="float64",
        )
        fixed_df = pd.concat([prepend_df, df])
        assert fixed_df.index.name is None
        return fixed_df

    return df
