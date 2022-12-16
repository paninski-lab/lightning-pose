"""Run inference on a list of models and videos."""

import hydra
from moviepy.editor import VideoFileClip
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pytorch_lightning as pl
from typeguard import typechecked

# from lightning_pose.utils.predictions import predict_single_video
# from lightning_pose.utils.predictions import create_labeled_video
from lightning_pose.utils import get_gpu_list_from_cfg
from lightning_pose.utils.io import (
    check_if_semi_supervised,
    ckpt_path_from_base_path,
    get_videos_in_dir,
    return_absolute_path,
    return_absolute_data_paths,
    VideoPredPathHandler,
)
from lightning_pose.utils.predictions import load_model_from_checkpoint
from lightning_pose.utils.scripts import get_imgaug_transform, get_dataset, get_data_module
from lightning_pose.utils.scripts import export_predictions_and_labeled_video

""" this script will get two imporant args. model to use and video folder to process.
hydra will orchestrate both. advanatages -- in the future we could parallelize to new machines.
no need to loop over models or folders. we do need to loop over videos within a folder.
however, keeping cfg.eval.hydra_paths is useful for the fiftyone image plotting. so keep"""


@hydra.main(config_path="configs", config_name="config_toy-dataset")
def predict_videos_in_dir(cfg: DictConfig):
    """
    This script will work with a path to a trained model's hydra folder

    From that folder it'll read the info about the model, get the checkpoint, and
    predict on a new vid

    If you need to predict multiple folders (each with one or more videos), define a
    --multirun and pass these directories as
    cfg.eval.test_videos_directory='dir/1','dir/2'...

    NOTE: by decorating with hydra, the current working directory will be become the new
    folder os.path.join(os.getcwd(), "/outputs/YYYY-MM-DD/hour-info")

    """

    # get pl trainer for prediction
    gpus = get_gpu_list_from_cfg(cfg)
    trainer = pl.Trainer(gpus=gpus)

    # loop over models
    for i, hydra_relative_path in enumerate(cfg.eval.hydra_paths):

        # cfg.eval.hydra_paths defines a list of relative paths to hydra folders
        # "YYYY-MM-DD/HH-MM-SS", and we extract an absolute path below

        # absolute_cfg_path will be the path of the trained model we're using for predictions
        absolute_cfg_path = return_absolute_path(hydra_relative_path, n_dirs_back=2)

        # load model
        model_cfg = OmegaConf.load(os.path.join(absolute_cfg_path, ".hydra/config.yaml"))
        ckpt_file = ckpt_path_from_base_path(
            base_path=absolute_cfg_path, model_name=model_cfg.model.model_name
        )
        model = load_model_from_checkpoint(cfg=cfg, ckpt_file=ckpt_file, eval=True)

        # load data module, which contains info about keypoint names, etc.
        data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
        imgaug_transform = get_imgaug_transform(cfg=cfg)
        dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
        data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

        # determine a directory in which to save video prediction csv files
        if cfg.eval.saved_vid_preds_dir is None:
            # save to where the videos are. may get an exception
            save_preds_dir = cfg.eval.test_videos_directory
        else:
            save_preds_dir = return_absolute_path(cfg.eval.saved_vid_preds_dir, n_dirs_back=3)

        # loop over videos in a provided directory
        video_files = get_videos_in_dir(return_absolute_path(cfg.eval.test_videos_directory))

        for video_file in video_files:

            video_pred_path_handler = VideoPredPathHandler(
                save_preds_dir=save_preds_dir,
                video_file=video_file,
                model_cfg=model_cfg,
            )
            prediction_csv_file = video_pred_path_handler()

            if cfg.eval.get("create_labeled_video", False):
                labeled_mp4_file = prediction_csv_file.replace(".csv", "_labeled.mp4")
            else:
                labeled_mp4_file = None

            export_predictions_and_labeled_video(
                video_file=video_file,
                cfg=cfg,
                ckpt_file=ckpt_file,
                prediction_csv_file=prediction_csv_file,
                labeled_mp4_file=labeled_mp4_file,
                trainer=trainer,
                model=model,
                data_module=data_module,
            )


if __name__ == "__main__":
    predict_videos_in_dir()
