"""Path handling functions."""

from __future__ import annotations  # python 3.8 compatibility for sphinx

import collections
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)

# video formats accepted for training and inference; compared case-insensitively
VIDEO_SUFFIXES: tuple[str, ...] = ('.mp4',)

# to ignore imports for sphinx-autoapidoc
__all__ = [
    "ckpt_path_from_base_path",
    "get_keypoint_names",
    "LabeledData",
    "parse_label_csv",
    "return_absolute_path",
    "resolve_path",
    "return_absolute_data_paths",
    "extract_session_name_from_video",
    "group_videos_in_dir_by_session",
    "filename_matches_view",
    "get_videos_in_dir",
    "normalize_video_paths",
    "VIDEO_SUFFIXES",
    "is_video_file",
    "list_video_files",
    "has_videos",
    "require_videos",
    "get_context_img_paths",
    "group_videos_by_session",
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
        since it does not need to exist for purely supervised models; use :func:`has_videos` or
        :func:`require_videos` to check it.

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


def filename_matches_view(filename: str | Path, view_name: str) -> bool:
    """Return whether a video filename belongs to a view.

    The view name must appear in the filename stem delimited by non-alphanumeric characters (or
    the start/end of the stem). For example, ``mouse_top_3.mp4`` matches view ``top``, but
    ``mousetop3.mp4`` does not, and ``sess_cam10.mp4`` matches ``cam10`` but not ``cam1``. This is
    the single matching rule used for all multiview video-filename parsing.

    Args:
        filename: video filename or path.
        view_name: name of the view.

    Returns:
        ``True`` if the filename contains the view name as a delimited token.
    """
    pattern = rf'(?<![0-9a-zA-Z]){re.escape(view_name)}(?![0-9a-zA-Z])'
    return re.search(pattern, Path(filename).stem) is not None


@overload
def get_videos_in_dir(
    video_dir: str,
    view_names: None = None,
    return_mp4_only: bool = True,
) -> list[str]: ...


@overload
def get_videos_in_dir(
    video_dir: str,
    view_names: list[str],
    return_mp4_only: bool = True,
) -> list[list[str]]: ...


def get_videos_in_dir(
    video_dir: str, view_names: list[str] | None = None, return_mp4_only: bool = True
) -> list[str] | list[list[str]]:
    """Gather the videos in a directory (strict; compare :func:`group_videos_in_dir_by_session`).

    Args:
        video_dir: directory containing the videos.
        view_names: for multiview models, the view names; filenames must contain their view's
            name delimited by non-alphanumeric characters (see :func:`filename_matches_view`).
        return_mp4_only: if ``False``, also accept ``.avi`` and ``.mov`` files.

    Returns:
        Single view (``view_names`` is ``None``): flat list of video paths. Multiview: nested
        list with one inner list per view (in ``view_names`` order), each holding that view's
        videos across sessions, i.e. views x sessions.

    Raises:
        OSError: if no videos are found, or a view has no videos.
        RuntimeError: if the views do not have the same set of videos.
    """
    assert Path(video_dir).is_dir()
    suffixes = VIDEO_SUFFIXES if return_mp4_only else ('.mp4', '.avi', '.mov')
    all_video_files = list_video_files(video_dir, suffixes=suffixes)
    video_files: list[str] | list[list[str]]
    if view_names:
        # make a list of lists, where the outer list is over views, each inner list is over videos/
        # sessions
        video_files = [
            [str(f) for f in all_video_files if filename_matches_view(f.name, view)]
            for view in view_names
        ]
        # check to make sure we have the same set of videos for each view
        # naming convention is <vid_name>_<view_name[0]>, <vid_name>_<view_name[1]>, etc.
        vid_names = [
            [vid_name.split(f"_{view_names[v]}")[0] for vid_name in video_files_]
            for v, video_files_ in enumerate(video_files)
        ]
        for view, view_files in zip(view_names, video_files, strict=True):
            if len(view_files) == 0:
                raise OSError(
                    f"Did not find any video files for view '{view}' in {video_dir}. "
                    "Video filenames must contain the view name delimited by "
                    "non-alphanumeric characters, e.g. <vid_name>_<view_name>.mp4 "
                    "or <vid_name>_<view_name>.short.mp4."
                )
        for vids_view in vid_names:
            if set(vids_view) != set(vid_names[0]):
                raise RuntimeError(
                    "Mismatched video names across views! "
                    "Please check your videos are in the format "
                    "<vid_name>_<view_name[0]>, <vid_name>_<view_name[1]>, etc., "
                    "where the `view_name` variable is defined in the config file."
                )
    else:
        video_files = [str(f) for f in all_video_files]

    if len(video_files) == 0:
        raise OSError(f"Did not find any valid video files in {video_dir}")

    return video_files


def is_video_file(path: str | Path, suffixes: tuple[str, ...] = VIDEO_SUFFIXES) -> bool:
    """Return whether ``path`` has a supported video suffix (case-insensitive).

    This only inspects the name; it does not check that the file exists.

    Args:
        path: path to test.
        suffixes: accepted suffixes, including the leading dot.

    Returns:
        ``True`` if the suffix is in ``suffixes``.
    """
    return Path(path).suffix.lower() in suffixes


def list_video_files(
    path: str | Path,
    suffixes: tuple[str, ...] = VIDEO_SUFFIXES,
) -> list[Path]:
    """List the video files at ``path``, which may be a single file or a directory.

    Args:
        path: a video file or a directory of videos (not searched recursively).
        suffixes: accepted suffixes, including the leading dot.

    Returns:
        sorted list of video files; ``[path]`` for a video file, and an empty list if the path
        is missing, is not a video file, or is a directory without videos.
    """
    path = Path(path)
    if path.is_file():
        return [path] if is_video_file(path, suffixes) else []
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.is_file() and is_video_file(f, suffixes))
    return []


def has_videos(video_dir: str | Path | None) -> bool:
    """Return whether ``video_dir`` is a video file or a directory containing videos.

    Args:
        video_dir: path to a video file or a directory of videos; may be ``None``.

    Returns:
        ``True`` if the path is an existing video file or a directory with at least one video;
        ``False`` otherwise.
    """
    return video_dir is not None and len(list_video_files(video_dir)) > 0


def require_videos(video_dir: str | Path | None, purpose: str) -> None:
    """Raise a descriptive error unless ``video_dir`` contains videos.

    Args:
        video_dir: path to a video file or a directory of videos; may be ``None``.
        purpose: what the videos are needed for, used in the error message.

    Raises:
        FileNotFoundError: if ``video_dir`` is unset, missing, or contains no videos.
    """
    if not has_videos(video_dir):
        raise FileNotFoundError(
            f'{purpose} requires a non-empty video directory (data.video_dir); '
            f'found {video_dir}. Add videos to this directory or use only supervised losses.'
        )


@overload
def normalize_video_paths(
    video_paths: list[str] | str,
    view_names: None = None,
) -> list[str]: ...


@overload
def normalize_video_paths(
    video_paths: list[str] | str,
    view_names: list[str],
) -> list[list[str]]: ...


def normalize_video_paths(
    video_paths: list[str] | str,
    view_names: list[str] | None = None,
) -> list[str] | list[list[str]]:
    """Validate and normalise video paths, returning a flat or nested list of mp4 paths.

    Accepts a single file path, a list of file paths, or a directory and returns a flat list of
    video paths for single-view models, or a nested list ``(views × sessions)`` for multi-view
    models.

    Args:
        video_paths: single video file path, list of video file paths, or a directory.
        view_names: if provided, organise the paths by view name for a multi-view model.

    Returns:
        For single-view: ``list[str]`` of mp4 file paths.
        For multi-view: ``list[list[str]]`` where outer list indexes views.

    Raises:
        ValueError: if ``video_paths`` is not a valid file, list, or directory, or if any of the
            resulting files does not have a supported video suffix (see :data:`VIDEO_SUFFIXES`).
    """
    # get input data
    if isinstance(video_paths, list):
        # presumably a list of files
        filenames = video_paths
    elif isinstance(video_paths, str) and Path(video_paths).is_file():
        # single video file
        filenames = [video_paths]
    elif isinstance(video_paths, str) and Path(video_paths).is_dir():
        # directory of videos
        filenames = get_videos_in_dir(video_paths, view_names=view_names)
    else:
        raise ValueError(
            "`video_paths_list` must be a list of files, a single file, or a directory name"
        )
    unsupported = [
        f
        for filename in filenames
        for f in ([filename] if isinstance(filename, str) else filename)
        if not is_video_file(f)
    ]
    if unsupported:
        raise ValueError(
            f'video files must be one of {VIDEO_SUFFIXES} (h.264 codec); '
            f'unsupported files: {unsupported}'
        )

    return filenames


def match_videos_to_views(
    video_files: list[Path], view_names: list[str]
) -> dict[str, Path]:
    """Match the videos of a single session to views based on their filenames.

    Filenames must contain their corresponding view's name, separated from the rest of the
    filename by some non-alphanumeric delimiter (see :func:`filename_matches_view`). For example,
    mouse_top_3.mp4 is allowed, but mousetop3.mp4 is not allowed.

    Args:
        video_files: one video per view.
        view_names: names of the views.

    Returns:
        Mapping from view name to video file.

    Raises:
        ValueError: if a file matches multiple views, or no file is found for a view.
    """
    assert len(video_files) == len(view_names), (
        f"{len(video_files)} != {len(view_names)}"
    )
    video_files_by_view: dict[str, Path] = {}
    for view_name in view_names:
        # Search all the video_files for a match.
        for video_file in video_files:
            if filename_matches_view(video_file.name, view_name):
                if view_name not in video_files_by_view:
                    video_files_by_view[view_name] = video_file
                else:
                    raise ValueError(f"File matches multiple views: {video_file}")
        # After the search if nothing was added to dict, there is a problem.
        if view_name not in video_files_by_view:
            raise ValueError(f"File not found for view: {view_name}")

    return video_files_by_view


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


def extract_session_name_from_video(video_filename: str, view_names: list[str]) -> str:
    """Extract the session name from a video filename by removing the view name.

    Args:
        video_filename: name of the video file (with or without extension)
        view_names: list of possible view names to remove

    Returns:
        Session name with ``_<view_name>`` removed; the filename stem if no view name matches
        (see :func:`filename_matches_view`).
    """
    name_without_ext = Path(video_filename).stem

    view_name = extract_view_name_from_video(video_filename, view_names)
    if view_name is None:
        return name_without_ext

    session_name = name_without_ext.replace(f"_{view_name}", "")
    if session_name == name_without_ext:
        # view name was found but not in the expected "_<view>" form (e.g. a hyphen was used
        # instead); the session name below still contains the view name, so this video likely
        # won't be grouped with its other views
        logger.warning(
            f"video filename '{video_filename}' contains view name '{view_name}' but "
            f"not as a '_{view_name}' suffix; expected the pattern "
            f"'<session>_{view_name}'. This video may not be grouped correctly with "
            "its other views for multiview prediction."
        )
    return session_name


def extract_view_name_from_video(
    video_filename: str, view_names: list[str]
) -> str | None:
    """Return the view name that a video filename matches, or ``None`` if there is none.

    Args:
        video_filename: name of the video file (with or without extension)
        view_names: list of possible view names

    Returns:
        The first view in ``view_names`` that matches (see :func:`filename_matches_view`).
    """
    for view_name in view_names:
        if filename_matches_view(video_filename, view_name):
            return view_name
    return None


def group_videos_by_session(
    video_paths: list[Path],
    view_names: list[str],
) -> list[list[Path]]:
    """Group a list of videos from different sessions and views by session.

    Returns a list of lists like
    `[[sess0_view0.mp4, sess0_view1.mp4, ...], [sess1_view0.mp4, sess1_view1.mp4, ...], ...]`,
    i.e. sessions x views. Sessions that are missing a view are skipped.

    Args:
        video_paths: List of paths to video files to split
        view_names: List of view names to find videos for

    Returns:
        List for each session, each containing a sub-list with videos for each view for
        that session
    """
    # map of session -> view -> video
    session_view_video_map = collections.defaultdict(dict[str, Path])

    for video_path in video_paths:
        view = extract_view_name_from_video(video_path.name, view_names)

        if view is not None:
            session = extract_session_name_from_video(video_path.name, view_names)

            session_view_video_map[session][view] = video_path

    video_views_per_session = []
    for view_to_video_path in session_view_video_map.values():
        # skip sessions with any missing view
        if any(view_name not in view_to_video_path for view_name in view_names):
            continue

        view_list = []
        for view_name in view_names:
            view_list.append(view_to_video_path[view_name])
        video_views_per_session.append(view_list)
    return video_views_per_session


def group_videos_in_dir_by_session(
    video_dir: str, view_names: list[str]
) -> list[list[Path]]:
    """Group the videos in a directory by session (lenient; compare :func:`get_videos_in_dir`).

    Returns a list of lists like
    `[[sess0_view0.mp4, sess0_view1.mp4, ...], [sess1_view0.mp4, sess1_view1.mp4, ...], ...]`,
    i.e. sessions x views. Sessions that are missing a view are skipped.

    Args:
        video_dir: Directory containing video files
        view_names: List of view names to find videos for

    Returns:
        List for each session, each containing a sub-list with videos for each view for
        that session

    Raises:
        FileNotFoundError: if the directory does not exist or contains no videos.
    """
    video_dir_path = Path(video_dir)

    if not video_dir_path.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    # Get all video files in the directory
    all_video_files = list_video_files(video_dir_path)

    if not all_video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    return group_videos_by_session(all_video_files, view_names)
