import os
import torch
import numpy as np
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
from pose_est_nets.utils.IO import set_or_open_folder, load_object
from pose_est_nets.datasets.datamodules import UnlabeledDataModule
from pose_est_nets.datasets.datasets import BaseTrackingDataset
from typing import Optional
from pose_est_nets.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
import yaml
import imgaug.augmenters as iaa

# TODO: add more tests as we consolidate datasets
_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BATCH_SIZE = 12
_HEIGHT = 256  # TODO: should be different numbers?
_WIDTH = 256

resnet_versions = [18, 34, 50, 101, 152]

repres_shape_list = [
    torch.Size([_BATCH_SIZE, 512, 1, 1]),
    torch.Size([_BATCH_SIZE, 512, 1, 1]),
    torch.Size([_BATCH_SIZE, 2048, 1, 1]),
    torch.Size([_BATCH_SIZE, 2048, 1, 1]),
    torch.Size([_BATCH_SIZE, 2048, 1, 1]),
]

num_keypoints = 34

fake_image_batch = torch.rand(
    size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
)
fake_keypoints = torch.rand(_BATCH_SIZE, num_keypoints, device=_TORCH_DEVICE) * _HEIGHT


def test_forward():
    """loop over different resnet versions and make sure that the
    resulting representation shapes make sense."""

    model = RegressionTracker(resnet_version=50, num_targets=34).to(_TORCH_DEVICE)
    representations = model.get_representations(fake_image_batch)
    assert representations.shape == repres_shape_list[2]
    preds = model(fake_image_batch)
    assert preds.shape == fake_keypoints.shape


def test_semisupervised():
    # define unsupervised datamodule
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)

    dataset = BaseTrackingDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    if os.path.exists("/home/jovyan"):
        video_directory = os.path.join(
            "/home/jovyan/mouseRunningData/unlabeled_videos"
        )  # DAN's
    else:
        video_directory = os.path.join("unlabeled_videos")  # NICK's
    video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]
    assert os.path.exists(video_files[0])
    with open("pose_est_nets/losses/default_hypers.yaml") as f:
        loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)
    print(loss_param_dict)
    semi_super_losses_to_use = ["pca"]
    datamod = UnlabeledDataModule(
        dataset=dataset,
        video_paths_list=video_files[0],
        specialized_dataprep="pca",
        loss_param_dict=loss_param_dict,
    )
    # for param_name, param_value in datamod.pca_param_dict[
    #     semi_super_losses_to_use[0]
    # ].items():
    #     loss_param_dict[semi_super_losses_to_use[0]][param_value] = param_value
    print(loss_param_dict)

    model = SemiSupervisedRegressionTracker(
        resnet_version=50,
        num_targets=34,
        loss_params=loss_param_dict,
        semi_super_losses_to_use=semi_super_losses_to_use,
    ).to(_TORCH_DEVICE)
    trainer = pl.Trainer(
        gpus=1 if _TORCH_DEVICE == "cuda" else 0,
        max_epochs=1,
        log_every_n_steps=1,
        auto_scale_batch_size=False,
    )  # auto_scale_batch_size not working
    trainer.fit(model=model, datamodule=datamod)


def test_nan_cleanup():
    # TODO: move to datamodules tests? used in pca for reshaped arr
    data = torch.rand(size=(4, 7))
    # in two different columns (i.e., body parts) make one view invisble
    data[[0, 1, 2, 3], [2, 2, 6, 6]] = torch.tensor(np.nan)
    nan_bool = (
        torch.sum(torch.isnan(data), dim=0) > 0
    )  # those columns (keypoints) that have more than zero nans
    assert nan_bool[2] == True
    assert nan_bool[6] == True
    clean_data = data[:, ~nan_bool]
    assert clean_data.shape == (4, 5)

    def clean_any_nans(data: torch.tensor, dim: int) -> torch.tensor:
        nan_bool = (
            torch.sum(torch.isnan(data), dim=dim) > 0
        )  # e.g., when dim == 0, those columns (keypoints) that have more than zero nans
        return data[:, ~nan_bool]

    out = clean_any_nans(data, 0)
    assert (out == clean_data).all()
