from pose_est_nets.utils.plotting_utils import predict_dataset, predict_videos
from pose_est_nets.models.heatmap_tracker import HeatmapTracker
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.datasets.datamodules import BaseDataModule
from omegaconf import OmegaConf
import yaml
import imgaug.augmenters as iaa
import os

ckpt = "toy_datasets/toy_model/epoch=9-step=49.ckpt"
hydra_output_directory = "toy_datasets/toy_model"
cfg_path = "toy_datasets/toy_model/config.yaml"
cfg = OmegaConf.load(cfg_path)

def test_predict_dataset():
    data_transform = []
    data_transform.append(
        iaa.Resize({
            "height": cfg.data.image_resize_dims.height,
            "width": cfg.data.image_resize_dims.width,
        })
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    heatmap_data = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    heatmap_module = BaseDataModule(heatmap_data)  # and default args
    heatmap_module.setup()
    predict_dataset(
        cfg=cfg, 
        datamod=heatmap_module, 
        hydra_output_directory=hydra_output_directory, 
        ckpt_file=ckpt
    )
    assert(os.path.exists("toy_datasets/toy_model/predictions.csv"))
    os.remove("toy_datasets/toy_model/predictions.csv")
    assert(os.path.exists("toy_datasets/toy_model/dataset_split_indices.json"))
    os.remove("toy_datasets/toy_model/dataset_split_indices.json")

def test_predict_videos():
    predict_videos(
        video_path="toy_datasets/toymouseRunningData/unlabeled_videos/",
        ckpt_file=ckpt,
        cfg_file=cfg,
        save_file="toy_datasets/toy_model/video_predictions.csv"
    )
    assert(os.path.exists("toy_datasets/toy_model/video_predictions.csv"))
    os.remove("toy_datasets/toy_model/video_predictions.csv")
    


