"""Run inference on a list of models and videos."""

import os

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from lightning_pose.utils.io import (
    ckpt_path_from_base_path,
    get_videos_in_dir,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import load_model_from_checkpoint
from lightning_pose.utils.scripts import (
    compute_metrics,
    export_predictions_and_labeled_video,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)


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
    trainer = pl.Trainer(accelerator="gpu", devices=1)

    # load data module, which contains info about keypoint names, etc.
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
    print("getting imgaug transform...")
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    print("getting dataset...")
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
    print("getting data module...")
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

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
        model = load_model_from_checkpoint(
            cfg=cfg,
            ckpt_file=ckpt_file,
            eval=True,
            data_module=data_module,
        )

        # loop over videos in a provided directory
        video_files = get_videos_in_dir(return_absolute_path(cfg.eval.test_videos_directory))

        for video_file in video_files:

            # prediction_csv_file = video_pred_path_handler()
            prediction_csv_file = os.path.join(
                absolute_cfg_path,
                "video_preds",
                os.path.basename(video_file).replace(".mp4", ".csv")
            )

            if cfg.eval.get("save_vids_after_training", False):
                labeled_mp4_file = prediction_csv_file.replace(".csv", "_labeled.mp4")
            else:
                labeled_mp4_file = None

            # debug
            print(f"\n\n{prediction_csv_file = }\n\n")

            export_predictions_and_labeled_video(
                video_file=video_file,
                cfg=cfg,
                ckpt_file=ckpt_file,
                prediction_csv_file=prediction_csv_file,
                labeled_mp4_file=labeled_mp4_file,
                trainer=trainer,
                model=model,
                data_module=data_module,
                save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
            )

            # compute and save various metrics
            try:
                compute_metrics(
                    cfg=cfg,
                    preds_file=prediction_csv_file,
                    data_module=data_module,
                )
            except Exception as e:
                print(f"Error predicting on video {video_file}:\n{e}")
                continue


if __name__ == "__main__":
    predict_videos_in_dir()
