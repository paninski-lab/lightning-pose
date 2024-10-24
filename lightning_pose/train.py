"""Example model training function."""

import os
import random
import sys
import warnings

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from typeguard import typechecked

from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import predict_dataset
from lightning_pose.utils.scripts import (
    calculate_train_batches,
    compute_metrics,
    export_predictions_and_labeled_video,
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)

# to ignore imports for sphix-autoapidoc
__all__ = ["train"]


@typechecked
def train(cfg: DictConfig) -> None:

    # reset all seeds
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # record lightning-pose version
    from lightning_pose import __version__ as lightning_pose_version
    with open_dict(cfg):
        cfg.model.lightning_pose_version = lightning_pose_version

    print("Our Hydra config file:")
    pretty_print_cfg(cfg)

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

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)
    # Log hydra config to tensorboard as helpful metadata.
    for key, value in cfg.items():
        logger.experiment.add_text(
            "hydra_config_%s" % key, "```\n%s```" % OmegaConf.to_yaml(value)
        )

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    callbacks = get_callbacks(
        cfg,
        early_stopping=cfg.training.get("early_stopping", False),
        lr_monitor=True,
        ckpt_every_n_epochs=cfg.training.get("ckpt_every_n_epochs", None)
    )

    # calculate number of batches for both labeled and unlabeled data per epoch
    limit_train_batches = calculate_train_batches(cfg, dataset)

    # set up trainer

    # Old configs may have num_gpus: 0. We will remove support in a future release.
    if cfg.training.num_gpus == 0:
        warnings.warn(
            "Config contains unsupported value num_gpus: 0. "
            "Update num_gpus to 1 in your config."
        )
    cfg.training.num_gpus = max(cfg.training.num_gpus, 1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.training.num_gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=min(
            cfg.training.check_val_every_n_epoch,
            cfg.training.max_epochs,  # for debugging or otherwise training for a short time
        ),
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        profiler=cfg.training.get("profiler", None),
        sync_batchnorm=True,
    )

    # train model!
    trainer.fit(model=model, datamodule=data_module)

    # When devices > 0, lightning creates a process per device.
    # Kill processes other than the main process, otherwise they all go forward.
    if not trainer.is_global_zero:
        sys.exit(0)

    # ----------------------------------------------------------------------------------
    # Post-training analysis
    # ----------------------------------------------------------------------------------
    hydra_output_directory = os.getcwd()
    print(f"Hydra output directory: {hydra_output_directory}")
    # get best ckpt
    best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)
    print(f"Best checkpoint: {os.path.basename(best_ckpt)}")
    # check if best_ckpt is a file
    if not os.path.isfile(best_ckpt):
        raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")
    # save config file
    cfg_file_local = os.path.join(hydra_output_directory, "config.yaml")
    with open(cfg_file_local, "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    # make unaugmented data_loader if necessary
    if cfg.training.imgaug != "default":
        cfg_pred = cfg.copy()
        cfg_pred.training.imgaug = "default"
        imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
        dataset_pred = get_dataset(
            cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred
        )
        data_module_pred = get_data_module(cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
        data_module_pred.setup()
    else:
        data_module_pred = data_module

    # ----------------------------------------------------------------------------------
    # predict on all labeled frames (train/val/test)
    # ----------------------------------------------------------------------------------
    # Rebuild trainer with devices=1 for prediction. Training flags not needed.
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    pretty_print_str("Predicting train/val/test images...")
    # compute and save frame-wise predictions
    preds_file = os.path.join(hydra_output_directory, "predictions.csv")
    predict_dataset(
        cfg=cfg,
        trainer=trainer,
        model=model,
        data_module=data_module_pred,
        ckpt_file=best_ckpt,
        preds_file=preds_file,
    )
    # compute and save various metrics
    try:
        # take care of multiview case, where multiple csv files have been saved
        preds_files = [
            os.path.join(hydra_output_directory, path) for path in
            os.listdir(hydra_output_directory) if path.endswith(".csv")
        ]
        if len(preds_files) > 1:
            preds_file = preds_files
        compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
    except Exception as e:
        print(f"Error computing metrics\n{e}")

    # ----------------------------------------------------------------------------------
    # predict folder of videos
    # ----------------------------------------------------------------------------------
    if cfg.eval.predict_vids_after_training:
        pretty_print_str("Predicting videos...")
        if cfg.eval.test_videos_directory is None:
            filenames = []
        else:
            filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
            vidstr = "video" if (len(filenames) == 1) else "videos"
            pretty_print_str(
                f"Found {len(filenames)} {vidstr} to predict (in cfg.eval.test_videos_directory)"
            )

        for video_file in filenames:
            assert os.path.isfile(video_file)
            pretty_print_str(f"Predicting video: {video_file}...")
            # get save name for prediction csv file
            video_pred_dir = os.path.join(hydra_output_directory, "video_preds")
            video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
            prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
            # get save name labeled video csv
            if cfg.eval.save_vids_after_training:
                labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
                labeled_mp4_file = os.path.join(labeled_vid_dir, video_pred_name + "_labeled.mp4")
            else:
                labeled_mp4_file = None
            # predict on video
            export_predictions_and_labeled_video(
                video_file=video_file,
                cfg=cfg,
                ckpt_file=best_ckpt,
                prediction_csv_file=prediction_csv_file,
                labeled_mp4_file=labeled_mp4_file,
                trainer=trainer,
                model=model,
                data_module=data_module_pred,
                save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
            )
            # compute and save various metrics
            # try:
            compute_metrics(
                cfg=cfg,
                preds_file=prediction_csv_file,
                data_module=data_module_pred,
            )
            # except Exception as e:
            #     print(f"Error predicting on video {video_file}:\n{e}")
            #     continue

    # ----------------------------------------------------------------------------------
    # predict on OOD frames
    # ----------------------------------------------------------------------------------
    # update config file to point to OOD data
    if isinstance(cfg.data.csv_file, list) or isinstance(cfg.data.csv_file, ListConfig):
        csv_file_ood = []
        for csv_file in cfg.data.csv_file:
            csv_file_ood.append(
                os.path.join(cfg.data.data_dir, csv_file).replace(".csv", "_new.csv"))
    else:
        csv_file_ood = os.path.join(
            cfg.data.data_dir, cfg.data.csv_file).replace(".csv", "_new.csv")

    if (isinstance(csv_file_ood, str) and os.path.exists(csv_file_ood)) \
            or (isinstance(csv_file_ood, list) and os.path.exists(csv_file_ood[0])):
        cfg_ood = cfg.copy()
        cfg_ood.data.csv_file = csv_file_ood
        cfg_ood.training.imgaug = "default"
        cfg_ood.training.train_prob = 1
        cfg_ood.training.val_prob = 0
        cfg_ood.training.train_frames = 1
        # build dataset/datamodule
        imgaug_transform_ood = get_imgaug_transform(cfg=cfg_ood)
        dataset_ood = get_dataset(
            cfg=cfg_ood, data_dir=data_dir, imgaug_transform=imgaug_transform_ood
        )
        data_module_ood = get_data_module(cfg=cfg_ood, dataset=dataset_ood, video_dir=video_dir)
        data_module_ood.setup()
        pretty_print_str("Predicting OOD images...")
        # compute and save frame-wise predictions
        preds_file_ood = os.path.join(hydra_output_directory, "predictions_new.csv")
        predict_dataset(
            cfg=cfg_ood,
            trainer=trainer,
            model=model,
            data_module=data_module_ood,
            ckpt_file=best_ckpt,
            preds_file=preds_file_ood,
        )
        # compute and save various metrics
        try:
            # take care of multiview case, where multiple csv files have been saved
            preds_files = [
                os.path.join(hydra_output_directory, path) for path in
                os.listdir(hydra_output_directory) if path.startswith("predictions_new")
            ]
            if len(preds_files) > 1:
                preds_file_ood = preds_files
            compute_metrics(cfg=cfg_ood, preds_file=preds_file_ood, data_module=data_module_ood)
        except Exception as e:
            print(f"Error computing metrics\n{e}")
