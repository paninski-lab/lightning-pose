"""Run inference on a list of models and videos."""

import glob
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import torch

from lightning_pose.utils.plotting_utils import get_videos_in_dir, predict_videos
from lightning_pose.utils.io import (
    ckpt_path_from_base_path,
    return_absolute_path,
)

""" this script will get two imporant args. model to use and video folder to process.
hydra will orchestrate both. advanatages -- in the future we could parallelize to new machines.
no need to loop over models or folders. we do need to loop over videos within a folder.
however, keeping cfg.eval.hydra_paths is useful for the fiftyone image plotting. so keep"""


@hydra.main(config_path="configs", config_name="config")
def make_predictions(cfg: DictConfig):
    """
    This script will work with a path to a trained model's hydra folder

    From that folder it'll read the info about the model, get the checkpoint, and
    predict on a new vid

    NOTE: by decorating with hydra, the current working directory will be become the new
    folder os.path.join(os.getcwd(), "/outputs/YYYY-MM-DD/hour-info")

    """
    # loop over models
    for i, hydra_relative_path in enumerate(cfg.eval.hydra_paths):

        # cfg.eval.hydra_paths defines a list of relative paths to hydra folders
        # "YYYY-MM-DD/HH-MM-SS", and we extract an absolute path below
        absolute_cfg_path = return_absolute_path(hydra_relative_path, n_dirs_back=2)
        # absolute_cfg_path will be the output path of the trained model we're using for predictions
        model_cfg = OmegaConf.load(
            os.path.join(absolute_cfg_path, ".hydra/config.yaml")
        )  # path for the cfg file saved from the current trained model
        ckpt_file = ckpt_path_from_base_path(
            base_path=absolute_cfg_path, model_name=model_cfg.model.model_name
        )

        # determine a directory in which to save video prediction csv files
        if cfg.eval.saved_video_predictions_directory is None:
            # save to where the videos are. may get an exception
            saved_video_predictions_directory = cfg.eval.path_to_test_videos
        else:
            saved_video_predictions_directory = return_absolute_path(
                cfg.eval.saved_video_predictions_directory, n_dirs_back=3
            )

        # loop over videos in a provided directory
        video_files = get_videos_in_dir(cfg.eval.path_to_test_videos)

        for test_videos_directory in cfg.eval.test_videos_directory:
            absolute_path_to_test_videos = return_absolute_path(path_to_test_videos)
            predict_videos(
                video_path=absolute_path_to_test_videos,
                ckpt_file=ckpt_file,
                cfg_file=model_cfg,
                save_file=saved_csv_path,
                sequence_length=cfg.eval.dali_parameters.sequence_length,
            )


if __name__ == "__main__":
    make_predictions()
