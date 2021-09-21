import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
from pose_est_nets.utils.IO import set_or_open_folder, load_object
from pose_est_nets.data.datamodules import UnlabeledDataModule
from pose_est_nets.data.datasets import BaseTrackingDataset
from typing import Optional
from pose_est_nets.models.regression_tracker import RegressionTracker, SemiSupervisedRegressionTracker

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
    #define unsupervised datamodule
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
    #video_directory = os.path.join("/home/jovyan/mouseRunningData/unlabeled_videos") #DAN's
    video_directory = os.path.join("unlabeled_videos") #NICK's
    video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]
    datamod = UnlabeledDataModule(dataset, video_files[0])
    train_loader = datamod.train_dataloader()
    pca_param_dict =
    semi_super_losses_to_use = 
    model = SemiSupervisedRegressionTracker(resnet_version = 50, num_targets=34, ).to(_TORCH_DEVICE)
    
