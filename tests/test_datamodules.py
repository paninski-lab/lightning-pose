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


data_transform = []
data_transform.append(
    iaa.Resize({"height": 384, "width": 384})
)  # dlc dimensions need to be repeatably divisable by 2
imgaug_transform = iaa.Sequential(data_transform)

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


def test_base_datamodule():

    regModule = BaseDataModule(regData)  # and default args
    regModule.setup()
    batch = next(iter(regModule.train_dataloader()))
    assert (
        torch.tensor(batch[0].shape)
        == torch.tensor([regModule.train_batch_size, 3, 384, 384])
    ).all()
    assert (
        torch.tensor(batch[1].shape)
        == torch.tensor([regModule.train_batch_size, 17, 2])
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
        regData, unlabeled_video_path=video_files[0]
    )  # and default args
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmapData, unlabeled_video_path=video_files[0]
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

def test_PCA():
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmapData, unlabeled_video_path=video_files[0]
    )
    unlabeled_module_heatmap.setup()
    unlabeled_module_heatmap.computePCA_params()
    print(unlabeled_module_heatmap.pca_param_dict)

    


