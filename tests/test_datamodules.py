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

video_directory = "toy_datasets/toymouseRunningData/unlabeled_videos"
assert os.path.exists(video_directory)

video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]
vids = []
for (
    f
) in (
    video_files
):  # video_directory may contain other random files that are not vids, DALI will try to read them
    if f.endswith(".mp4"):  # hardcoded for the toydataset folder
        vids.append(f)

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

# grab example loss config file from repo
base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
loss_cfg = os.path.join(base_dir, "scripts", "configs", "losses", "loss_params.yaml")
with open(loss_cfg) as f:
    loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)
# hard code multivew pca info for now
loss_param_dict["pca_multiview"]["mirrored_column_matches"] = [
    [0, 1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13, 14]
]


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

    # test subsampling of training frames
    train_frames = 10  # integer
    heatmap_module = BaseDataModule(heatmapData, train_frames=train_frames)
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == train_frames

    train_frames = 1  # integer
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        heatmapData,
        train_frames=train_frames,
        train_probability=train_probability)
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert (
        len(train_dataloader.dataset)
        == int(train_probability * len(heatmap_module.dataset))
    )

    train_frames = 0.1  # fraction < 1
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        heatmapData,
        train_frames=train_frames,
        train_probability=train_probability)
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert (
        len(train_dataloader.dataset)
        == int(train_frames * train_probability * len(heatmap_module.dataset))
    )

    train_frames = 1000000  # integer larger than number of labeled frames
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        heatmapData,
        train_frames=train_frames,
        train_probability=train_probability)
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert (
        len(train_dataloader.dataset)
        == int(train_probability * len(heatmap_module.dataset))
    )

    # raise exception when not a path
    with pytest.raises(ValueError):
        train_frames = -1
        heatmap_module = BaseDataModule(
            heatmapData,
            train_frames=train_frames)
        heatmap_module.setup()


def test_UnlabeledDataModule():
    # TODO: make a short video in toydatasets
    # TODO: seperate into a heatmap test + regression test
    unlabeled_module_regression = UnlabeledDataModule(
        regData, video_paths_list=vids  # video_files[0]
    )  # and default args
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmapData, video_paths_list=vids  # video_files[0]
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


def test_PCA():  # TODO FINISH WRITING TEST
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmapData,
        video_paths_list=vids,
        loss_param_dict=loss_param_dict,
        losses_to_use="pca_multiview",
    )


def test_reshape():
    ints = np.arange(34)
    ints_reshaped = ints.reshape(-1, 2)
    ints_reverted = ints_reshaped.reshape(34)
    assert (ints_reshaped[:, 0] == np.arange(0, 34, 2)).all()
    assert (ints_reverted == ints).all()
