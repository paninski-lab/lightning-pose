import glob
import os
import numpy as np
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pose_est_nets.utils.plotting_utils import predict_videos


@hydra.main(config_path="configs", config_name="config")
def make_predictions(cfg: DictConfig):
    """note, by decorating with hydra, the current working directory will be become the new folder os.path.join(os.getcwd(), "/outputs/YYYY-MM-DD/hour-info")"""

    # go to folders up to the "outputs" folder, and search for hydra_path from cfg
    model_config_path = os.path.join(
        "../..", cfg.eval.hydra_paths[0], ".hydra/config.yaml"
    )
    model_config_file = OmegaConf.load(model_config_path)

    model_search_path = os.path.join(
        "../..",  # go two folders up to the outputs/ folder
        cfg.eval.hydra_paths[
            0
        ],  # TODO: this is a list of paths, right now we're supporting one
        "tb_logs",  # TODO: may change when we switch from Tensorboard
        model_config_file.model.model_name,  # get the name string of the model (determined pre-training)
        "version_0",  # always version_0 because ptl starts a version_0 folder in a new hydra outputs folder
        "checkpoints",
        "*.ckpt",
    )

    # TODO: we're taking the last ckpt. make sure that with multiple checkpoints, this is what we want
    model_ckpt_path = glob.glob(model_search_path)[
        -1
    ]  # the output of glob.glob is a list with all files that match the search.
    predict_videos(
        video_path=cfg.eval.path_to_test_videos,
        model_file=model_ckpt_path,
        config_file=model_config_path,
        save_file=cfg.eval.path_to_save_predictions,
        sequence_length=cfg.eval.dali_parameters.sequence_length,
    )


if __name__ == "__main__":
    make_predictions()
