"""Example model training script."""

import hydra
from omegaconf import DictConfig
import os
import lightning.pytorch as pl
import torch
import numpy as np

from lightning_pose.callbacks.callbacks import AnnealWeight
from lightning_pose.data.utils import (
    compute_num_train_frames,
    split_sizes_from_probabilities,
)
from lightning_pose.utils import get_gpu_list_from_cfg, pretty_print_str
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import predict_dataset
from lightning_pose.utils.scripts import (
    export_predictions_and_labeled_video,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
    compute_metrics,
)


@hydra.main(config_path="configs", config_name="config_toy-dataset")
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

    if (
        ("temporal" in cfg.model.losses_to_use)
        and model.do_context
        and not data_module.unlabeled_dataloader.context_sequences_successive
    ):
        raise ValueError(
            f"Temporal loss is not compatible with non-successive context sequences. "
            f"Please change cfg.dali.context.train.consecutive_sequences=True."
        )

    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_supervised_loss",
        patience=cfg.training.early_stop_patience,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor="val_supervised_loss")
    transfer_unfreeze_callback = pl.callbacks.BackboneFinetuning(
        unfreeze_backbone_at_epoch=cfg.training.unfreezing_epoch,
        lambda_func=lambda epoch: 1.5,
        backbone_initial_ratio_lr=0.1,
        should_align=True,
        train_bn=True,
    )
    anneal_weight_callback = AnnealWeight(**cfg.callbacks.anneal_weight)
    callbacks = [
        early_stopping,
        lr_monitor,
        ckpt_callback,
        transfer_unfreeze_callback
    ]
    # we just need this callback for unsupervised models
    if (cfg.model.losses_to_use != []) and (cfg.model.losses_to_use is not None):
        callbacks.append(anneal_weight_callback)

    # determine gpu setup
    gpus = get_gpu_list_from_cfg(cfg)

    # calculate limit_train_batches; for semi-supervised models, this tells us how many
    # batches to take from each dataloader (labeled and unlabeled) during a given epoch.
    # The default set here is to exhaust all batches from the labeled data loader, often
    # leaving many video frames untouched. But the unlabeled data loader will be
    # randomly reset for the next epoch. We also enforce a minimum value of 10 so that
    # models with a small number of labeled frames will cycle through the dataset
    # multiple times per epoch, which we have found to be useful empirically.
    if cfg.training.limit_train_batches is None:
        # TODO: small bit of redundant code from datamodule
        datalen = dataset.__len__()
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
        )
        num_train_frames = compute_num_train_frames(
            data_splits_list[0], cfg.training.get("train_frames", None)
        )
        num_labeled_batches = int(
            np.ceil(num_train_frames / cfg.training.train_batch_size)
        )
        limit_train_batches = np.max([num_labeled_batches, 10])  # 10 is minimum
    else:
        limit_train_batches = cfg.training.limit_train_batches

    # set up trainer
    trainer = pl.Trainer(  # TODO: be careful with devices when scaling to multiple gpus
        gpus=gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        multiple_trainloader_mode=cfg.training.multiple_trainloader_mode,
        profiler=cfg.training.profiler,
    )

    # train model!
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

    # make unaugmented data_loader if necessary
    if cfg.training.imgaug != "default":
        cfg_pred = cfg.copy()
        cfg_pred.training.imgaug = "default"
        imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
        dataset_pred = get_dataset(
            cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred
        )
        data_module_pred = get_data_module(
            cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir
        )
        data_module_pred.setup()
    else:
        data_module_pred = data_module

    # ----------------------------------------------------------------------------------
    # predict on all labeled frames (train/val/test)
    # ----------------------------------------------------------------------------------
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
        compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
    except:
        pass

    # ----------------------------------------------------------------------------------
    # predict folder of videos
    # ----------------------------------------------------------------------------------
    if cfg.eval.predict_vids_after_training:
        pretty_print_str("Predicting videos...")
        if cfg.eval.test_videos_directory is None:
            filenames = []
        else:
            filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
            pretty_print_str(
                "Found {} videos to predict on (in cfg.eval.test_videos_directory)".format(
                    len(filenames)))
        for video_file in filenames:
            assert os.path.isfile(video_file)
            pretty_print_str("Predicting video: {}...".format(video_file))
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
                gpu_id=cfg.training.gpu_id,
                data_module=data_module_pred,
                save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
            )
            # compute and save various metrics
            try:
                compute_metrics(
                    cfg=cfg, preds_file=prediction_csv_file, data_module=data_module_pred
                )
            except:
                continue

    # ----------------------------------------------------------------------------------
    # predict on OOD frames
    # ----------------------------------------------------------------------------------
    # update config file to point to OOD data
    csv_file_ood = os.path.join(cfg.data.data_dir, cfg.data.csv_file).replace(".csv", "_new.csv")
    if os.path.exists(csv_file_ood):
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
        data_module_ood = get_data_module(
            cfg=cfg_ood, dataset=dataset_ood, video_dir=video_dir
        )
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
            compute_metrics(cfg=cfg_ood, preds_file=preds_file_ood, data_module=data_module_ood)
        except:
            pass


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
