"""Example model training script."""

import hydra
from omegaconf import DictConfig, ListConfig
import os
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

from lightning_pose.callbacks.callbacks import AnnealWeight
from lightning_pose.utils.io import check_video_paths, return_absolute_data_paths, \
    return_absolute_path, get_keypoint_names
from lightning_pose.utils.predictions import create_labeled_video, predict_dataset, make_pred_arr_undo_resize, get_csv_file
from lightning_pose.utils.scripts import (
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
    pretty_print_str,
)
from lightning_pose.data.utils import count_frames
# attempting to predict a video here
from lightning_pose.data.dali import PrepareDALI, video_pipe, LightningWrapper, ContextLightningWrapper
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from typing import List, Tuple
from torchtyping import TensorType, patch_typeguard
from lightning_pose.utils.predictions_new import PredictionHandler


_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="configs", config_name="config_mm_cont")
def train(cfg: DictConfig):
    """Main fitting function, accessed from command line."""

    print("Our Hydra config file:")
    pretty_print(cfg)

    # path handling for toy data
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

    # ----------------------------------------------------------------------------------
    # Set up data/model objects
    # ----------------------------------------------------------------------------------

    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg)

    # dataset
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)
    
    if ("temporal" in cfg.model.losses_to_use) \
            and model.do_context \
            and not data_module.unlabeled_dataloader.context_sequences_successive:
        raise ValueError(
            f"Temporal loss is not compatible with non-successive context sequences. "
            f"Please change context_sequences_successive in data_module.py to True.")
    
    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_supervised_loss",
        patience=cfg.training.early_stop_patience,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_supervised_loss"
    )
    transfer_unfreeze_callback = pl.callbacks.BackboneFinetuning(
        unfreeze_backbone_at_epoch=cfg.training.unfreezing_epoch,
        lambda_func=lambda epoch: 1.5,
        backbone_initial_ratio_lr=0.1,
        should_align=True,
        train_bn=True,
    )
    anneal_weight_callback = AnnealWeight(**cfg.callbacks.anneal_weight)
    # TODO: add wandb?
    # determine gpu setup
    if _TORCH_DEVICE == "cpu":
        gpus = 0
    elif isinstance(cfg.training.gpu_id, list):
        gpus = cfg.training.gpu_id
    elif isinstance(cfg.training.gpu_id, ListConfig):
        gpus = list(cfg.training.gpu_id)
    elif isinstance(cfg.training.gpu_id, int):
        gpus = [cfg.training.gpu_id]
    else:
        raise NotImplementedError(
            "training.gpu_id must be list or int, not {}".format(
                type(cfg.training.gpu_id)
            )
        )
    # # calculate limit_train_batches
    # if cfg.training.limit_train_batches is None:
    #     if type(data_module.train_dataloader()) is torch.utils.data.DataLoader:
    #         num_batches = len(data_module.train_dataloader())
    #     elif type(data_module.train_dataloader()) is dict:
    #         num_batches = len(data_module.train_dataloader["labeled"])
    #     print(num_batches)
    #     exit()
    #     limit_train_batches = np.ceil(len(data_module.train_dataset) / cfg.training.train_batch_size)
    #     limit_train_batches = np.max([limit_train_batches, 10]) # 10 is minimum
    #     print("limit_train_batches: {}".format(limit_train_batches))
    # else: 
    #     limit_train_batches = cfg.training.limit_train_batches
    limit_train_batches = cfg.training.limit_train_batches
    print("limit_train_batches={}".format(limit_train_batches))
    # cannot access the data_module because train_dataset hasn't been setup yet.
    # set up trainer
    trainer = pl.Trainer(  # TODO: be careful with devices when scaling to multiple gpus
        gpus=gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=[
            early_stopping,
            lr_monitor,
            ckpt_callback,
            transfer_unfreeze_callback,
            anneal_weight_callback,
        ],
        logger=logger,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        multiple_trainloader_mode=cfg.training.multiple_trainloader_mode,
        profiler=cfg.training.profiler,
    )
    trainer.fit(model=model, datamodule=data_module)

    # ----------------------------------------------------------------------------------
    # Post-training analysis
    # ----------------------------------------------------------------------------------
    hydra_output_directory = os.getcwd()
    print("Hydra output directory: {}".format(hydra_output_directory))
    # get best ckpt
    best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)
    # check if best_ckpt is a file 
    if not os.path.isfile(best_ckpt):
        raise FileNotFoundError(
            "Cannot find model checkpoint. Have you trained for too few epochs?"
        )
    
    # ----------------------------------------------------------------------------------
    # predict full dataloader
    # ----------------------------------------------------------------------------------
    pretty_print_str("Predicting train/val/test images...")
    labeled_preds = trainer.predict(model=model, dataloaders=data_module.full_labeled_dataloader(), ckpt_path=best_ckpt, return_predictions=True)
    
    pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=None)

    # call this instance on a single vid's preds
    labeled_preds_df = pred_handler(preds=labeled_preds)

    labeled_preds_df.to_csv(os.path.join(hydra_output_directory, "predictions.csv"))
    
    # ----------------------------------------------------------------------------------
    # predict folder of videos
    # ----------------------------------------------------------------------------------
    # get dali loader for video, eval network on it, save preds.
    # cfg.eval.test_videos_directory holds videos to predict.
    if cfg.eval.test_videos_directory is None:
        filenames = []
    else:
        filenames = check_video_paths(return_absolute_path(
            cfg.eval.test_videos_directory))
        pretty_print_str(
            "Found {} videos to predict on (in cfg.eval.test_videos_directory)".format(
                len(filenames)))

    for video_file in filenames:
        assert os.path.isfile(video_file)
        pretty_print_str("Predicting video: {}...".format(video_file))
        # base model: check we can build and run pipe and get a decent looking batch
        model_type = "context" if cfg.model.do_context else "base"
        # initialize
        vid_pred_class = PrepareDALI(train_stage="predict", model_type=model_type, dali_config=cfg.dali, filenames=[video_file], resize_dims=[dataset.height, dataset.width])
        # get loader
        predict_loader = vid_pred_class()
        # predict 
        preds = trainer.predict(model=model, ckpt_path=best_ckpt, dataloaders=predict_loader, return_predictions=True)
        # initialize prediction handler class, can process multiple vids with a shared cfg and data_module
        pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=video_file)
        # call this instance on a single vid's preds
        preds_df = pred_handler(preds=preds)
        # save the predictions to a csv
        # e.g.,: '/home/jovyan/dali-seq-testing/test_vid_with_fr.mp4' -> 'test_vid_with_fr.csv'
        base_vid_name_for_save = os.path.basename(video_file).split('.')[0]
        video_pred_dir = os.path.join(hydra_output_directory, 'video_preds')
        # create directory if it doesn't exist
        os.makedirs(video_pred_dir, exist_ok=True)
        preds_df.to_csv(os.path.join(video_pred_dir, "{}.csv".format(base_vid_name_for_save)))
        
        # TODO: generate a video if cfg.eval.save_video is True
        # use create_labeled_videos() func.
        if cfg.eval.save_vids_after_training:
            pretty_print_str("Generating video...")
            # TODO: wrap inside a func
            labeled_vid_dir = os.path.join(video_pred_dir, 'labeled_videos')
            os.makedirs(labeled_vid_dir, exist_ok=True)
            video_file_labeled = os.path.join(labeled_vid_dir, base_vid_name_for_save + '_labeled.mp4')
            video_clip = VideoFileClip(video_file)
            
            # transform df to numpy array
            keypoints_arr = np.reshape(
                    preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
            xs_arr = keypoints_arr[:, :, 0]
            ys_arr = keypoints_arr[:, :, 1]
            mask_array = keypoints_arr[:, :, 2] > cfg.eval.confidence_thresh_for_vid

            # do here the video generation
            create_labeled_video(
                    clip=video_clip, xs_arr=xs_arr, ys_arr=ys_arr,
                    mask_array=mask_array, filename=video_file_labeled)
    
    # ----------------------------------------------------------------------------------


def pretty_print(cfg):

    for key, val in cfg.items():
        if key == "eval":
            continue
        print("--------------------")
        print("%s parameters" % key)
        print("--------------------")
        for k, v in val.items():
            print("{}: {}".format(k, v))
        print()
    print("\n\n")


if __name__ == "__main__":
    train()
