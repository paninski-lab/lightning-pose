"""Run inference on a list of models and videos."""

import hydra
from omegaconf import DictConfig, OmegaConf
import os

from typeguard import typechecked

from lightning_pose.utils.plotting_utils import get_videos_in_dir, predict_single_video
from lightning_pose.utils.io import (
    check_if_semi_supervised,
    ckpt_path_from_base_path,
    return_absolute_path,
    VideoPredPathHandler,
)

""" this script will get two imporant args. model to use and video folder to process.
hydra will orchestrate both. advanatages -- in the future we could parallelize to new machines.
no need to loop over models or folders. we do need to loop over videos within a folder.
however, keeping cfg.eval.hydra_paths is useful for the fiftyone image plotting. so keep"""


@hydra.main(config_path="configs", config_name="config")
def predict_videos_in_dir(cfg: DictConfig):
    """
    This script will work with a path to a trained model's hydra folder

    From that folder it'll read the info about the model, get the checkpoint, and
    predict on a new vid

    If you need to predict multiple folders (each with one or more videos), define a --multirun and pass these directories as
    cfg.eval.test_videos_directory='dir/1','dir/2'...

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
        if cfg.eval.saved_vid_preds_dir is None:
            # save to where the videos are. may get an exception
            save_preds_dir = cfg.eval.test_videos_directory
        else:
            save_preds_dir = return_absolute_path(
                cfg.eval.saved_vid_preds_dir, n_dirs_back=3
            )
        # save_preds_dir is checked below in VideoPredPathHandler

        # loop over videos in a provided directory
        video_files = get_videos_in_dir(
            return_absolute_path(cfg.eval.test_videos_directory)
        )

        for video_file in video_files:
            video_pred_path_handler = VideoPredPathHandler(
                save_preds_dir=save_preds_dir,
                video_file=video_file,
                model_cfg=model_cfg,
            )
            preds_file = video_pred_path_handler()

            preds_df, heatmaps_np = predict_single_video(
                video_file=video_file,
                ckpt_file=ckpt_file,
                cfg_file=model_cfg,
                preds_file=preds_file,
            )
            # this script is not doing anything with preds_df and heatmaps_np


if __name__ == "__main__":
    predict_videos_in_dir()
