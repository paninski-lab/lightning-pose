"""Run inference on a list of models and videos."""

import os

import hydra
import lightning.pytorch as pl
import numpy as np
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked

from lightning_pose.utils import get_gpu_list_from_cfg
from lightning_pose.utils.io import (
    check_if_semi_supervised,
    ckpt_path_from_base_path,
    get_videos_in_dir,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import load_model_from_checkpoint
from lightning_pose.utils.scripts import (
    export_predictions_and_labeled_video,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)

""" this script will get two imporant args. model to use and video folder to process.
hydra will orchestrate both. advanatages -- in the future we could parallelize to new machines.
no need to loop over models or folders. we do need to loop over videos within a folder.
however, keeping cfg.eval.hydra_paths is useful for the fiftyone image plotting. so keep"""


@typechecked
class VideoPredPathHandler:
    """class that defines filename for a predictions .csv file, given video file and
    model specs.
    """

    def __init__(
        self, save_preds_dir: str, video_file: str, model_cfg: DictConfig
    ) -> None:
        self.video_file = video_file
        self.save_preds_dir = save_preds_dir
        self.model_cfg = model_cfg
        self.check_input_paths()

    @property
    def video_basename(self) -> str:
        return os.path.basename(self.video_file).split(".")[0]

    @property
    def loss_str(self) -> str:
        semi_supervised = check_if_semi_supervised(self.model_cfg.model.losses_to_use)
        loss_names = []
        loss_weights = []
        loss_str = ""
        if semi_supervised:  # add the loss names and weights
            loss_str = ""
            if len(self.model_cfg.model.losses_to_use) > 0:
                loss_names = list(self.model_cfg.model.losses_to_use)
                for loss in loss_names:
                    loss_weights.append(self.model_cfg.losses[loss]["log_weight"])

                loss_str = ""
                for loss, weight in zip(loss_names, loss_weights):
                    loss_str += "_" + loss + "_" + str(weight)

            else:  # fully supervised, return empty string
                loss_str = ""
        return loss_str

    def check_input_paths(self) -> None:
        assert os.path.isfile(self.video_file)
        assert os.path.isdir(self.save_preds_dir)

    def build_pred_file_basename(self, extra_str="") -> str:
        return "%s_%s%s%s.csv" % (
            self.video_basename,
            self.model_cfg.model.model_type,
            self.loss_str,
            extra_str,
        )

    def __call__(self, extra_str="") -> str:
        pred_file_basename = self.build_pred_file_basename(extra_str=extra_str)
        return os.path.join(self.save_preds_dir, pred_file_basename)


@hydra.main(config_path="configs", config_name="config_mirror-mouse-example")
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
        print("getting imgaug transform...")
        imgaug_transform = get_imgaug_transform(cfg=cfg)
        print("getting dataset...")
        dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
        print("getting data module...")
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

            if cfg.eval.get("save_vids_after_training", False):
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
