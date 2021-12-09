import pytorch_lightning as pl
from pytorch_lightning import profiler
import torch
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
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
from pose_est_nets.utils.io import (
    verify_real_data_paths,
    check_if_semi_supervised,
    ckpt_path_from_base_path,
    format_and_update_loss_info,
)
from pose_est_nets.utils.plotting_utils import predict_dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import BackboneFinetuning

import os

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    print("Our Hydra config file:")
    print(cfg)
    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)

    data_dir, video_dir = verify_real_data_paths(cfg.data)

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
            root_directory=data_dir,
            csv_path=cfg.data.csv_file,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
        )
    elif cfg.model.model_type == "heatmap":
        dataset = HeatmapDataset(
            root_directory=data_dir,
            csv_path=cfg.data.csv_file,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
            downsample_factor=cfg.data.downsample_factor,
        )
    else:
        raise NotImplementedError(
            "%s is an invalid cfg.model.model_type" % cfg.model.model_type
        )

    if not semi_supervised:
        if not (cfg.training.gpu_id, int):
            raise NotImplementedError(
                "Cannot currently fit fully supervised model on multiple gpus"
            )
        datamod = BaseDataModule(
            dataset=dataset,
            train_batch_size=cfg.training.train_batch_size,
            val_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            torch_seed=cfg.training.rng_seed_data_pt,
        )
        if cfg.model.model_type == "regression":
            model = RegressionTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                torch_seed=cfg.training.rng_seed_model_pt,
            )

        elif cfg.model.model_type == "heatmap":
            model = HeatmapTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                downsample_factor=cfg.data.downsample_factor,
                supervised_loss=cfg.model.heatmap_loss_type,
                output_shape=dataset.output_shape,
                torch_seed=cfg.training.rng_seed_model_pt,
            )
        else:
            raise NotImplementedError(
                "%s is an invalid cfg.model.model_type for a fully supervised model"
                % cfg.model.model_type
            )

    else:  # semi_supervised == True
        if not (cfg.training.gpu_id, int):
            raise NotImplementedError(
                "Cannot currently fit semi-supervised model on multiple gpus"
            )
        # copy data-specific details into loss dict
        loss_param_dict, losses_to_use = format_and_update_loss_info(cfg)
        
        datamod = UnlabeledDataModule(
            dataset=dataset,
            video_paths_list=video_dir,
            losses_to_use=losses_to_use,
            loss_param_dict=loss_param_dict,
            train_batch_size=cfg.training.train_batch_size,
            val_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers,
            train_probability=cfg.training.train_prob,
            val_probability=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            unlabeled_batch_size=1,
            unlabeled_sequence_length=cfg.training.unlabeled_sequence_length,
            torch_seed=cfg.training.rng_seed_data_pt,
            dali_seed=cfg.training.rng_seed_data_dali,
            device_id=cfg.training.gpu_id,
        )
        if cfg.model.model_type == "regression":
            model = SemiSupervisedRegressionTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                loss_params=datamod.loss_param_dict,
                semi_super_losses_to_use=losses_to_use,
                torch_seed=cfg.training.rng_seed_model_pt,
            )

        elif cfg.model.model_type == "heatmap":
            print(datamod.loss_param_dict)
            model = SemiSupervisedHeatmapTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                downsample_factor=cfg.data.downsample_factor,
                supervised_loss=cfg.model.heatmap_loss_type,
                output_shape=dataset.output_shape,
                loss_params=datamod.loss_param_dict,
                semi_super_losses_to_use=losses_to_use,
                torch_seed=cfg.training.rng_seed_model_pt,
            )
        else:
            raise NotImplementedError(
                "%s is an invalid cfg.model.model_type for a semi-supervised model"
                % cfg.model.model_type
            )

    logger = TensorBoardLogger("tb_logs", name=cfg.model.model_name)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.training.early_stop_patience, mode="min"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor="val_loss")
    transfer_unfreeze_callback = BackboneFinetuning(
        unfreeze_backbone_at_epoch=cfg.training.unfreezing_epoch,
        lambda_func=lambda epoch: 1.5,
        backbone_initial_ratio_lr=0.1,
        should_align=True,
        train_bn=True,
    )
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
    trainer = pl.Trainer(  # TODO: be careful with devices if you want to scale to multiple gpus
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
        ],
        logger=logger,
        limit_train_batches=cfg.training.limit_train_batches,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        multiple_trainloader_mode=cfg.training.multiple_trainloader_mode,
        profiler=cfg.training.profiler,
    )
    trainer.fit(model=model, datamodule=datamod)

    hydra_output_directory = os.getcwd()
    print("Hydra output directory: {}".format(hydra_output_directory))
    model_ckpt = trainer.checkpoint_callback.best_model_path
    print("Best model path: {}".format(model_ckpt))
    if not os.path.isfile(os.path.abspath(model_ckpt)):
        raise FileNotFoundError(
            "Cannot find a checkpoint to the model. Check if you have trained for too few epochs."
        )
    predict_dataset(
        cfg=cfg,
        datamod=datamod,
        hydra_output_directory=hydra_output_directory,
        ckpt_file=model_ckpt,
    )


if __name__ == "__main__":
    train()
