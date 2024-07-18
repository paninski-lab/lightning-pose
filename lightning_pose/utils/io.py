"""Path handling functions."""

import os
from typing import List, Optional, Tuple, Union

import pandas as pd
from omegaconf import DictConfig, ListConfig
from typeguard import typechecked

# to ignore imports for sphix-autoapidoc
__all__ = [
    "ckpt_path_from_base_path",
    "check_if_semi_supervised",
    "load_label_csv_from_cfg",
    "get_keypoint_names",
    "return_absolute_path",
    "return_absolute_data_paths",
    "get_videos_in_dir",
    "check_video_paths",
]


@typechecked
def ckpt_path_from_base_path(
    base_path: str,
    model_name: str,
    logging_dir_name: str = "tb_logs/",
    version: int = 0,
) -> str:
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
        str: path to model checkpoint

    """
    import glob

    model_search_path = os.path.join(
        base_path,
        logging_dir_name,        # may change when we switch from Tensorboard
        model_name,              # get the name string of the model (determined pre-training)
        "version_%i" % version,  # always version_0 because ptl starts a version_0 dir
        "checkpoints",
        "*.ckpt",
    )
    # TODO: we're taking the last ckpt. make sure that with multiple checkpoints, this
    # is what we want
    model_ckpt_path = glob.glob(model_search_path)[-1]
    return model_ckpt_path


@typechecked
def check_if_semi_supervised(losses_to_use: Union[ListConfig, list, None] = None) -> bool:
    """Use config file to determine if model is semi-supervised.

    Take the entry of the hydra cfg that specifies losses_to_use. If it contains
    meaningful entries, infer that we want a semi_supervised model.

    Args:
        losses_to_use (Union[ListConfig, list, None], optional): the cfg entry
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
def load_label_csv_from_cfg(cfg: Union[DictConfig, dict]) -> pd.DataFrame:
    """Helper function for easy loading.

    Args:
        cfg: DictConfig

    Returns:
        pd.DataFrame
    """

    csv_file = os.path.join(cfg["data"]["data_dir"], cfg["data"]["csv_file"])
    labels_df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
    return labels_df


@typechecked
def get_keypoint_names(
    cfg: Optional[DictConfig] = None,
    csv_file: Optional[str] = None,
    header_rows: Optional[list] = [0, 1, 2],
) -> List[str]:
    if os.path.exists(csv_file):
        if header_rows is None:
            if "header_rows" in cfg.data:
                header_rows = list(cfg.data.header_rows)
            else:
                # assume dlc format
                header_rows = [0, 1, 2]
        csv_data = pd.read_csv(csv_file, header=header_rows)
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
    view_names: Optional[List[str]] = None,
    return_mp4_only: bool = True
) -> Union[List[str], List[List[str]]]:
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
                os.path.join(video_dir, f) for f in all_video_files if
                (f.endswith(allowed_formats) and f.split(".")[-2].endswith(view))
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
    video_paths: Union[List[str], str],
    view_names: Optional[List[str]] = None,
) -> Union[List[str], List[List[str]]]:
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
