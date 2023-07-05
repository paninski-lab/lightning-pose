"""Example model training script."""

import os

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

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


@hydra.main(config_path="configs", config_name="config_mirror-mouse-example")
def train(cfg: DictConfig):
    """Main fitting function, accessed from command line."""

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

    # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
    callbacks = get_callbacks(cfg)

    # calculate number of batches for both labeled and unlabeled data per epoch
    limit_train_batches = calculate_train_batches(cfg, dataset)

    # set up trainer
    trainer = pl.Trainer(  # TODO: be careful with devices when scaling to multiple gpus
        accelerator="gpu",  # TODO: control from outside
        devices=1,  # TODO: control from outside
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        profiler=cfg.training.get("profiler", None),
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
        raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

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
            filenames = check_video_paths(
                return_absolute_path(cfg.eval.test_videos_directory)
            )
            vidstr = "video" if (len(filenames) == 1) else "videos"
            pretty_print_str(
                f"Found {len(filenames)} {vidstr} to predict on (in cfg.eval.test_videos_directory)"
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
                labeled_mp4_file = os.path.join(
                    labeled_vid_dir, video_pred_name + "_labeled.mp4"
                )
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
                save_heatmaps=cfg.eval.get(
                    "predict_vids_after_training_save_heatmaps", False
                ),
            )
            # compute and save various metrics
            try:
                compute_metrics(
                    cfg=cfg,
                    preds_file=prediction_csv_file,
                    data_module=data_module_pred,
                )
            except Exception as e:
                print(f"Error predicting on video {video_file}:\n{e}")
                continue

    # ----------------------------------------------------------------------------------
    # predict on OOD frames
    # ----------------------------------------------------------------------------------
    # update config file to point to OOD data
    csv_file_ood = os.path.join(cfg.data.data_dir, cfg.data.csv_file).replace(
        ".csv", "_new.csv"
    )
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
            compute_metrics(
                cfg=cfg_ood, preds_file=preds_file_ood, data_module=data_module_ood
            )
        except Exception as e:
            print(f"Error computing metrics\n{e}")


if __name__ == "__main__":
    train()
