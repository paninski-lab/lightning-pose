"""Path handling functions."""
from __future__ import annotations  # python 3.8 compatibility for sphinx

import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
from typeguard import typechecked

# to ignore imports for sphix-autoapidoc
__all__ = [
    "ckpt_path_from_base_path",
    "check_if_semi_supervised",
    "get_keypoint_names",
    "return_absolute_path",
    "return_absolute_data_paths",
    "extract_session_name_from_video",
    "find_video_files_for_views",
    "get_videos_in_dir",
    "check_video_paths",
    "get_context_img_paths",
]


@typechecked
def ckpt_path_from_base_path(
    base_path: str,
    model_name: str,
    logging_dir_name: str = "tb_logs/",
) -> str | None:
    """Given a path to a hydra output with trained model, extract the model .ckpt file.

    Args:
        base_path (str): path to a folder with logs and checkpoint. for example,
            function will search base_path/logging_dir_name/model_name...
        model_name (str): the name you gave your model before training it; appears as
            model_name in lightning-pose/scripts/config/model_params.yaml
        logging_dir_name (str, optional): name of the folder in logs, controlled in
            train_hydra.py Defaults to "tb_logs/".
        version (int. optional):

    Returns:
        str: path to model checkpoint, or None if none found.

    """
    import glob

    model_search_path = os.path.join(
        base_path,
        logging_dir_name,  # may change when we switch from Tensorboard
        model_name,  # get the name string of the model (determined pre-training)
        "version_*",  # TensorBoardLogger increments versions if retraining same model.
        "checkpoints",
        "*.ckpt",
    )

    # Find all checkpoint files
    checkpoint_files = glob.glob(model_search_path)
    # Return None if none were found.
    if not checkpoint_files:
        return None

    # Get the latest version's checkpoint files.
    ckpt_file_by_version = {}
    for f in checkpoint_files:
        version = re.search(r"version_(\d)", f).group(1)
        version = int(version)
        if version in ckpt_file_by_version:
            raise NotImplementedError(
                f"Multiple checkpoint files found in version directory for {f}. "
                "Logic to select among multiple checkpoints is not yet implemented."
            )
        ckpt_file_by_version[version] = f

    latest_version = max(ckpt_file_by_version.keys())
    return ckpt_file_by_version[latest_version]


@typechecked
def check_if_semi_supervised(losses_to_use: ListConfig | list | None = None) -> bool:
    """Use config file to determine if model is semi-supervised.

    Take the entry of the hydra cfg that specifies losses_to_use. If it contains
    meaningful entries, infer that we want a semi_supervised model.

    Args:
        losses_to_use (ListConfig, list | None, optional): the cfg entry
            specifying semisupervised losses to use. Defaults to None.

    Returns:
        bool: True if the model is semi_supervised. False otherwise.

    """
    if losses_to_use is None:  # null
        semi_supervised = False
    elif len(losses_to_use) == 0:  # empty list
        semi_supervised = False
    elif len(losses_to_use) == 1 and losses_to_use[0] == "":  # list with an empty string
        semi_supervised = False
    else:
        semi_supervised = True
    return semi_supervised


@typechecked
def get_keypoint_names(
    cfg: DictConfig | None = None,
    csv_file: str | None = None,
    header_rows: list | None = [0, 1, 2],
) -> list[str]:
    if os.path.exists(csv_file):
        if header_rows is None:
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
        keypoint_names = ["bp_%i" % n for n in range(cfg.data.num_targets // 2)]
    return keypoint_names


# --------------------------------------------------------------------------------------
# Path handling functions for running toy dataset
# --------------------------------------------------------------------------------------

@typechecked
def return_absolute_path(possibly_relative_path: str, n_dirs_back: int = 3) -> str:
    """Return absolute path from possibly relative path."""
    if os.path.isabs(possibly_relative_path):
        # absolute path already; do nothing
        abs_path = possibly_relative_path
    else:
        # our toy_dataset in relative path
        cwd_split = os.getcwd().split(os.path.sep)
        desired_path_list = cwd_split[:-n_dirs_back]
        if desired_path_list[-1] == "multirun":
            # hydra multirun, go one dir back
            desired_path_list = desired_path_list[:-1]
        abs_path = os.path.join(os.path.sep, *desired_path_list, possibly_relative_path)
    if not os.path.exists(abs_path):
        raise IOError("%s is not a valid path" % abs_path)
    return abs_path


@typechecked
def return_absolute_data_paths(data_cfg: DictConfig, n_dirs_back: int = 3) -> Tuple[str, str]:
    """Generate absolute path for our example toy data.

    @hydra.main decorator switches the cwd when executing the decorated function, e.g.,
    our train(). so we're in some /outputs/YYYY-MM-DD/HH-MM-SS folder.

    Args:
        data_cfg (DictConfig): data config file with paths to data and video folders.
        n_dirs_back (int):

    Returns:
        Tuple[str, str]: absolute paths to data and video folders.

    """
    data_dir = return_absolute_path(data_cfg.data_dir, n_dirs_back=n_dirs_back)
    if os.path.isabs(data_cfg.video_dir):
        video_dir = data_cfg.video_dir
    else:
        video_dir = os.path.join(data_dir, data_cfg.video_dir)
    # assert that those paths exist and in the proper format
    assert os.path.isdir(data_dir)
    assert os.path.isdir(video_dir) or os.path.isfile(video_dir)
    return data_dir, video_dir


@typechecked
def get_videos_in_dir(
    video_dir: str,
    view_names: list[str] | None = None,
    return_mp4_only: bool = True
) -> list[str] | list[list[str]]:
    """Gather videos to process from a single directory."""
    assert os.path.isdir(video_dir)
    # get all video files in directory, from allowed formats
    allowed_formats = (".mp4", ".avi", ".mov")
    if return_mp4_only:
        allowed_formats = ".mp4"
    if view_names:
        # make a list of lists, where the outer list is over views, each inner list is over videos/
        # sessions
        all_video_files = sorted(os.listdir(video_dir))
        video_files = [
            [
                os.path.join(video_dir, f)
                for f in all_video_files
                if (
                    f.endswith(allowed_formats)
                    and (
                        f.split(".")[-2].endswith(view)
                        or f"_{view}_" in f
                    )
                )
            ]
            for view in view_names
        ]
        # check to make sure we have the same set of videos for each view
        # naming convention is <vid_name>_<view_name[0]>, <vid_name>_<view_name[1]>, etc.
        vid_names = [
            [vid_name.split(f'_{view_names[v]}')[0] for vid_name in video_files_]
            for v, video_files_ in enumerate(video_files)
        ]
        for vids_view in vid_names:
            if set(vids_view) != set(vid_names[0]):
                raise RuntimeError(
                    "Mismatched video names across views! "
                    "Please check your videos are in the format "
                    "<vid_name>_<view_name[0]>, <vid_name>_<view_name[1]>, etc., "
                    "where the `view_name` variable is defined in the config file."
                )
    else:
        video_files = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith(allowed_formats)
        ]

    if len(video_files) == 0:
        raise IOError("Did not find any valid video files in %s" % video_dir)

    return video_files


@typechecked
def check_video_paths(
    video_paths: list[str] | str,
    view_names: list[str] | None = None,
) -> list[str] | list[list[str]]:
    # get input data
    if isinstance(video_paths, list):
        # presumably a list of files
        filenames = video_paths
    elif isinstance(video_paths, str) and os.path.isfile(video_paths):
        # single video file
        filenames = [video_paths]
    elif isinstance(video_paths, str) and os.path.isdir(video_paths):
        # directory of videos
        filenames = get_videos_in_dir(video_paths, view_names=view_names)
    else:
        raise ValueError(
            "`video_paths_list` must be a list of files, a single file, or a directory name"
        )
    for filename in filenames:
        if isinstance(filename, str):
            filename = [filename]
        for f in filename:
            assert f.endswith(".mp4"), "video files must be mp4 format!"

    return filenames


def collect_video_files_by_view(video_files: list[Path], view_names: list[str]) -> dict[str, Path]:
    """Given a list of video files, matches them to views based on their filenames.

    Filenames must contain their corresponding view's name, separated by the rest of the filename
    by some non-alphanumeric delimiter. For example, mouse_top_3.mp4 is allowed, but mousetop3.mp4
    is not allowed.
    """
    assert len(video_files) == len(view_names), f"{len(video_files)} != {len(view_names)}"
    video_files_by_view: dict[str, Path] = {}
    for view_name in view_names:
        # Search all the video_files for a match.
        for video_file in video_files:
            if re.search(rf"(?<!0-9a-zA-Z){re.escape(view_name)}(?![0-9a-zA-Z])", video_file.stem):
                if view_name not in video_files_by_view:
                    video_files_by_view[view_name] = video_file
                else:
                    raise ValueError(f"File matches multiple views: {video_file}")
        # After the search if nothing was added to dict, there is a problem.
        if view_name not in video_files_by_view:
            raise ValueError(f"File not found for view: {view_name}")

    return video_files_by_view


@typechecked
def get_context_img_paths(center_img_path: Path) -> list[Path]:
    """Given the path to a center image frame, return paths of 5 context frames
    (n-2, n-1, n, n+1, n+2).

    Negative indices are floored at 0.
    """
    match = re.search(r"(\d+)", center_img_path.stem)
    assert (
        match is not None
    ), f"No frame index in filename, can't get context frames: {center_img_path.name}"

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
            new_row, index=[df.index.name], columns=df.columns, dtype="float64"
        )
        fixed_df = pd.concat([prepend_df, df])
        assert fixed_df.index.name is None
        return fixed_df

    return df


def extract_session_name_from_video(video_filename: str, view_names: list[str]) -> str:
    """
    Extract session name from video filename by removing the view name.

    Simple approach: remove the underscore and view name from the filename.

    Args:
        video_filename: Name of the video file (with or without extension)
        view_names: List of possible view names to remove

    Returns:
        Session name with view name removed
    """
    # Remove file extension if present
    name_without_ext = Path(video_filename).stem

    # Try to remove each view name (with underscore before it)
    for view_name in view_names:
        if view_name in name_without_ext:
            # Remove the underscore and view name
            session_name = name_without_ext.replace(f"_{view_name}", "")
            return session_name

    # If no view name found, return the original name
    return name_without_ext


def find_video_files_for_views(video_dir: str, view_names: list[str]) -> list[str]:
    """
    Find video files for each view by looking for files that contain the view name.

    Args:
        video_dir: Directory containing video files
        view_names: List of view names to find videos for

    Returns:
        List of video file paths, one for each view
    """
    video_dir_path = Path(video_dir)

    if not video_dir_path.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    # Get all video files in the directory
    all_video_files = list(video_dir_path.glob("*.mp4"))

    if not all_video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    video_files_per_view = []

    # Find videos for each view
    for view_name in view_names:
        video_file = None

        # Look for videos that contain this view name
        for video_path in all_video_files:
            if view_name in video_path.name:
                video_file = str(video_path)
                break

        if video_file is None:
            # Fallback: use the first available video file
            video_file = str(all_video_files[0])
            print(f"Warning: No specific video found for view '{view_name}', using: {video_file}")

        video_files_per_view.append(video_file)
        print(f"Found video for view '{view_name}': {video_file}")

    return video_files_per_view
