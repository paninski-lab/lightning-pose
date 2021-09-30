"""want:
1. give paths to checkpoints (with models)
2. initialize those models
3. setup a datamodule with (train/test/val images; one or more datamodules needed?)
4. get predictions and log them to fiftyone
5. launch fiftyone
6. separately connect with ssh tunnel and inspect"""
import os
import pytorch_lightning as pl
from pose_est_nets.models.new_heatmap_tracker import SemiSupervisedHeatmapTracker
import hydra
from omegaconf import DictConfig, OmegaConf
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
import imgaug.augmenters as iaa
from pose_est_nets.utils.fiftyone_plotting_utils import evaluate
import time

path = "/home/jovyan/pose-estimation-nets/outputs/2021-09-30/00-08-34/tb_logs/my_test_model/version_0/checkpoints/epoch=1-step=105.ckpt"
assert os.path.isfile(path)


@hydra.main(config_path="configs", config_name="config")
def predict(cfg: DictConfig):
    # How to transform the images
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
    # Losses used for models
    loss_param_dict = OmegaConf.to_object(cfg.losses)
    losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)
    # Init dataset
    dataset = HeatmapDataset(
        root_directory=cfg.data.data_dir,
        csv_path=cfg.data.csv_path,
        header_rows=OmegaConf.to_object(cfg.data.header_rows),
        imgaug_transform=imgaug_transform,
        downsample_factor=cfg.data.downsample_factor,
    )
    # Init datamodule
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
    # Init model
    # for now this works without saving the pca params to dict
    model = SemiSupervisedHeatmapTracker.load_from_checkpoint(
        path, semi_super_losses_to_use=losses_to_use
    )

    # model = model.load_from_checkpoint(path, strict=False)
    print("loaded weights")

    evaluate(cfg, datamod, model)
    # time.sleep(5 * 60)  # wait for five minutes


if __name__ == "__main__":
    predict()
