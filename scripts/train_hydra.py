"""Example model training script."""

import hydra
from omegaconf import DictConfig, ListConfig
import os
import pytorch_lightning as pl
import torch

from pose_est_nets.callbacks.callbacks import AnnealWeight
from pose_est_nets.utils.io import return_absolute_data_paths
from pose_est_nets.utils.plotting_utils import predict_dataset
from pose_est_nets.utils.scripts import (
    get_data_module,
    get_dataset,
    get_imgaug_tranform,
    get_loss_factories,
    get_model,
)

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """Main fitting function, accessed from command line."""

    print("Our Hydra config file:")
    pretty_print(cfg)

    # path handling for toy datasets
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

    # ----------------------------------------------------------------------------------
    # Set up data/model objects
    # ----------------------------------------------------------------------------------

    # imgaug transform
    imgaug_transform = get_imgaug_tranform(cfg=cfg)

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

    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_supervised_loss",
        patience=cfg.training.early_stop_patience,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval="epoch"
    )
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
        limit_train_batches=cfg.training.limit_train_batches,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        multiple_trainloader_mode=cfg.training.multiple_trainloader_mode,
        profiler=cfg.training.profiler,
    )
    trainer.fit(model=model, datamodule=data_module)

    # ----------------------------------------------------------------------------------
    # Post-training cleanup
    # ----------------------------------------------------------------------------------

    hydra_output_directory = os.getcwd()
    print("Hydra output directory: {}".format(hydra_output_directory))
    model_ckpt = trainer.checkpoint_callback.best_model_path
    model_ckpt_abs = os.path.abspath(model_ckpt)
    print("Best model path: {}".format(model_ckpt_abs))
    if not os.path.isfile(model_ckpt_abs):
        raise FileNotFoundError(
            "Cannot find model checkpoint. Have you trained for too few epochs?"
        )
    # export predictions on train/val/test data to a csv saved in model directory
    predict_dataset(
        cfg=cfg,
        data_module=data_module,
        hydra_output_directory=hydra_output_directory,
        ckpt_file=model_ckpt,
    )

    # generate a video
    # evaluate the network on everything in the video_dir, and make videos.


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
    print("------------------------------------------------------------------\n")


if __name__ == "__main__":
    train()
