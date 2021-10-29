import glob
import os
import numpy as np
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pose_est_nets.utils.plotting_utils import (
    predict_videos,
)
from pose_est_nets.utils.io import (
    get_absolute_hydra_path_from_hydra_str,
    ckpt_path_from_base_path,
)


@hydra.main(config_path="configs", config_name="config")
def make_predictions(cfg: DictConfig):
    """this script will work with a path to a trained model's hydra folder
    from that folder it'll read the info about the model, get the checkpoint, and predict on a new vid"""
    """note, by decorating with hydra, the current working directory will be become the new folder os.path.join(os.getcwd(), "/outputs/YYYY-MM-DD/hour-info")"""
    # go to folders up to the "outputs" folder, and search for hydra_path from cfg
    for hydra_relative_path in cfg.eval.hydra_paths:
        # cfg.eval.hydra_paths defines a list of relative paths to hydra folders "YYYY-MM-DD/HH-MM-SS", and we extract an absolute path below
        absolute_cfg_path = get_absolute_hydra_path_from_hydra_str(hydra_relative_path)
        model_cfg = OmegaConf.load(
            os.path.join(absolute_cfg_path, ".hydra/config.yaml")
        )  # path for the cfg file saved from the current trained model
        ckpt_file = ckpt_path_from_base_path(
            base_path=absolute_cfg_path, model_name=model_cfg.model.model_name
        )

        predict_videos(
            video_path=cfg.eval.path_to_test_videos,
            ckpt_file=ckpt_file,
            cfg_file=model_cfg,
            save_file=cfg.eval.path_to_save_predictions,
            sequence_length=cfg.eval.dali_parameters.sequence_length,
        )


if __name__ == "__main__":
    make_predictions()
