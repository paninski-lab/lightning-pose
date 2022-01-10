"""Run inference on a list of models and videos."""

import glob
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import torch

from lightning_pose.utils.plotting_utils import predict_videos
from lightning_pose.utils.io import (
    ckpt_path_from_base_path,
    return_absolute_path,
)


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
        model_cfg = OmegaConf.load(
            os.path.join(absolute_cfg_path, ".hydra/config.yaml")
        )  # path for the cfg file saved from the current trained model
        ckpt_file = ckpt_path_from_base_path(
            base_path=absolute_cfg_path,
            model_name=model_cfg.model.model_name
        )

        # determine where to save video prediction csv files
        if cfg.eval.path_to_save_predictions is None:
            saved_csv_path = absolute_cfg_path
        else:
            saved_csv_path = return_absolute_path(
                cfg.eval.path_to_save_predictions[i],  n_dirs_back=3)

        # loop over video directories
        for path_to_test_videos in cfg.eval.path_to_test_videos:
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
