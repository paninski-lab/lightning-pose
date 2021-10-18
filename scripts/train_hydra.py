import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import imgaug.augmenters as iaa
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
from pose_est_nets.models.heatmap_tracker import (
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
)
from pose_est_nets.callbacks.freeze_unfreeze_callback import (
    FeatureExtractorFreezeUnfreeze,
)
from pytorch_lightning.loggers import TensorBoardLogger

import os

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: move the datapaths from cfg.training
@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    print(cfg)
    data_transform = []
    data_transform.append(
        iaa.Resize(
            {
                "height": cfg.data.image_resize_dims.height,
                "width": cfg.data.image_resize_dims.width,
            }
        )
    )
    imgaug_transform = iaa.Sequential(data_transform)
    if cfg.model.model_type == "regression":
        dataset = BaseTrackingDataset(
            root_directory=cfg.data.data_dir,
            csv_path=cfg.data.csv_path,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
        )
    elif cfg.model.model_type == "heatmap":
        dataset = HeatmapDataset(
            root_directory=cfg.data.data_dir,
            csv_path=cfg.data.csv_path,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
            downsample_factor=cfg.data.downsample_factor,
        )
    else:
        print("INVALID DATASET SPECIFIED")
        exit()

    if not (cfg.model["semi_supervised"]):
        datamod = BaseDataModule(
            dataset=dataset,
            train_batch_size=cfg.training.train_batch_size,
            validation_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
        )
        if cfg.model.model_type == "regression":
            model = RegressionTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
            )

        elif cfg.model.model_type == "heatmap":
            model = HeatmapTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=dataset.output_shape,
            )
        else:
            print("INVALID DATASET SPECIFIED")
            exit()

    else:
        loss_param_dict = OmegaConf.to_object(cfg.losses)
        losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)
        datamod = UnlabeledDataModule(
            dataset=dataset,
            video_paths_list=cfg.data.video_dir,  # just a single path for now
            specialized_dataprep=losses_to_use,
            loss_param_dict=loss_param_dict,
            train_batch_size=cfg.training.train_batch_size,
            validation_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
        )
        if cfg.model.model_type == "regression":
            model = SemiSupervisedRegressionTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                loss_params=datamod.loss_param_dict,
                semi_super_losses_to_use=losses_to_use,
            )

        elif cfg.model.model_type == "heatmap":
            model = SemiSupervisedHeatmapTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=dataset.output_shape,
                loss_params=datamod.loss_param_dict,
                semi_super_losses_to_use=losses_to_use,
            )
    logger = TensorBoardLogger("tb_logs", name= cfg.model.model_name)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, mode="min"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor="val_loss")
    transfer_unfreeze_callback = FeatureExtractorFreezeUnfreeze(
        cfg.training.unfreezing_epoch
    )  # Not used for now
    # TODO: add backbone refinement, add wandb?
    trainer = pl.Trainer(  # TODO: be careful with the devices here if you want to scale to multiple gpus
        gpus=1 if _TORCH_DEVICE == "cuda" else 0,
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=[
            early_stopping,
            lr_monitor,
            ckpt_callback,
            transfer_unfreeze_callback,
        ],
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamod)


if __name__ == "__main__":
    train()  # I think you get issues when you try to get return values from a hydra function
