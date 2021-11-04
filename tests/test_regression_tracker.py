import os
import torch
import numpy as np
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
from pose_est_nets.utils.io import set_or_open_folder, load_object
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

# @pytest.fixture
# def create_dataset():
#     from pose_est_nets.datasets.datasets import BaseTrackingDataset
#
#     dataset = BaseTrackingDataset(
#         root_directory="toy_datasets/toymouseRunningData",
#         csv_path="CollectedData_.csv",
#         header_rows=[1, 2],
#     )
#     return dataset


# @pytest.fixture
# def initialize_model():
#     from pose_est_nets.models.regression_tracker import RegressionTracker
#
#     model = RegressionTracker(num_targets=34, resnet_version=18)
#     return model


# @pytest.fixture
# def initialize_data_module(create_dataset):
#     from pose_est_nets.datasets.datasets import TrackingDataModule
#
#     data_module = TrackingDataModule(
#         create_dataset,
#         train_batch_size=4,
#         validation_batch_size=2,
#         test_batch_size=2,
#         num_workers=8,
#     )
#     return data_module


def test_forward():
    """loop over different resnet versions and make sure that the
    resulting representation shapes make sense."""

    fake_image_batch = torch.rand(
        size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
    )
    model = RegressionTracker(resnet_version=50, num_targets=34).to(_TORCH_DEVICE)
    representations = model.get_representations(fake_image_batch)
    assert representations.shape == repres_shape_list[2]
    preds = model(fake_image_batch)
    # assert preds.shape == fake_keypoints.shape
    assert preds.shape == torch.Size([_BATCH_SIZE, num_keypoints])
    # remove model/data from gpu; then cache can be cleared
    del fake_image_batch
    del model
    del representations
    del preds
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_semisupervised():
    # define unsupervised datamodule
    from pose_est_nets.utils.plotting_utils import get_videos_in_dir

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
    video_directory = "toy_datasets/toymouseRunningData/unlabeled_videos"
    video_files = get_videos_in_dir(video_directory)

    # grab example loss config file from repo
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    loss_cfg = os.path.join(
        base_dir, "scripts", "configs", "losses", "loss_params.yaml"
    )
    with open(loss_cfg) as f:
        loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)
    # hard code multivew pca info for now
    loss_param_dict["pca_multiview"]["mirrored_column_matches"] = [
        [0, 1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13, 14]
    ]

    semi_super_losses_to_use = ["pca_multiview"]
    datamod = UnlabeledDataModule(
        dataset=dataset,
        video_paths_list=video_files[0],
        losses_to_use="pca_multiview",
        loss_param_dict=loss_param_dict,
        train_batch_size=4,
    )
    model = SemiSupervisedRegressionTracker(
        resnet_version=18,
        num_targets=34,
        loss_params=datamod.loss_param_dict,
        semi_super_losses_to_use=semi_super_losses_to_use,
    ).to(_TORCH_DEVICE)
    trainer = pl.Trainer(
        gpus=1 if _TORCH_DEVICE == "cuda" else 0,
        max_epochs=1,
        log_every_n_steps=1,
        auto_scale_batch_size=False,
    )  # auto_scale_batch_size not working
    trainer.fit(model=model, datamodule=datamod)

    # remove model/data from gpu; then cache can be cleared
    del dataset
    del datamod
    del model
    del trainer
    torch.cuda.empty_cache()  # remove tensors from gpu


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


torch.cuda.empty_cache()
