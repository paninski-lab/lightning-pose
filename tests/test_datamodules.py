# TODO: Initialize a dataset, then a data module, for each type

import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
import imgaug.augmenters as iaa
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
import numpy as np
import yaml

data_transform = []
data_transform.append(
    iaa.Resize({"height": 384, "width": 384})
)  # dlc dimensions need to be repeatably divisable by 2
imgaug_transform = iaa.Sequential(data_transform)

# video_directory = os.path.join(
#     "/home/jovyan/mouseRunningData/unlabeled_videos"
# )  # DAN's
video_directory = os.path.join("unlabeled_videos")  # NICK's
video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]
assert os.path.exists(video_files[0])

regData = BaseTrackingDataset(
    root_directory="toy_datasets/toymouseRunningData",
    csv_path="CollectedData_.csv",
    header_rows=[1, 2],
    imgaug_transform=imgaug_transform,
)

heatmapData = HeatmapDataset(
    root_directory="toy_datasets/toymouseRunningData",
    csv_path="CollectedData_.csv",
    header_rows=[1, 2],
    imgaug_transform=imgaug_transform,
)

with open("pose_est_nets/losses/default_hypers.yaml") as f:
        loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)

def test_base_datamodule():

    regModule = BaseDataModule(regData)  # and default args
    regModule.setup()
    batch = next(iter(regModule.train_dataloader()))
    assert (
        torch.tensor(batch[0].shape)
        == torch.tensor([regModule.train_batch_size, 3, 384, 384])
    ).all()
    assert (
        torch.tensor(batch[1].shape) == torch.tensor([regModule.train_batch_size, 34])
    ).all()

    heatmapModule = BaseDataModule(heatmapData)  # and default args
    heatmapModule.setup()
    batch = next(iter(heatmapModule.train_dataloader()))
    assert (
        torch.tensor(batch[0].shape)
        == torch.tensor([heatmapModule.train_batch_size, 3, 384, 384])
    ).all()
    assert (
        torch.tensor(batch[1].shape)
        == torch.tensor(
            [
                heatmapModule.train_batch_size,
                17,
                384 / (2 ** heatmapData.downsample_factor),
                384 / (2 ** heatmapData.downsample_factor),
            ]
        )
    ).all()


def test_UnlabeledDataModule():
    # TODO: make a short video in toydatasets
    # TODO: seperate into a heatmap test + regression test
    unlabeled_module_regression = UnlabeledDataModule(
        regData, video_paths_list=video_files[0]
    )  # and default args
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmapData, video_paths_list=video_files[0]
    )  # and default args
    unlabeled_module_regression.setup()
    unlabeled_module_heatmap.setup()

    loader = CombinedLoader(unlabeled_module_regression.train_dataloader())
    out = next(iter(loader))
    assert list(out.keys())[0] == "labeled"
    assert list(out.keys())[1] == "unlabeled"
    assert out["unlabeled"].shape == (
        unlabeled_module_regression.train_batch_size,
        3,
        384,
        384,
    )

    loader = CombinedLoader(unlabeled_module_heatmap.train_dataloader())
    out = next(iter(loader))
    assert list(out.keys())[0] == "labeled"
    assert list(out.keys())[1] == "unlabeled"
    assert out["unlabeled"].shape == (
        unlabeled_module_heatmap.train_batch_size,
        3,
        384,
        384,
    )


def test_PCA(): #TODO FINISH WRITING TEST
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmapData, video_paths_list=video_files[0], loss_param_dict = loss_param_dict, specialized_dataprep = 'pca'
    )
    #unlabeled_module_heatmap.setup()
    #unlabeled_module_heatmap.computePCA_params() #These get automatically run now


def test_reshape():
    ints = np.arange(34)
    ints_reshaped = ints.reshape(-1, 2)
    ints_reverted = ints_reshaped.reshape(34)
    assert (ints_reshaped[:, 0] == np.arange(0, 34, 2)).all()
    assert (ints_reverted == ints).all()
