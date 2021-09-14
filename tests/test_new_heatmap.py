import os
from pose_est_nets.models.new_heatmap_tracker import HeatmapTracker
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
from pose_est_nets.utils.IO import set_or_open_folder, load_object
from typing import Optional
import torchvision

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BATCH_SIZE = 12
_HEIGHT = 256  # TODO: should be different numbers?
_WIDTH = 256


def test_create_double_upsampling_layer():
    fake_image_batch = torch.rand(
        size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
    )
    heatmap_model = HeatmapTracker(num_targets=34)
    upsampling_layer = HeatmapTracker.create_double_upsampling_layer(
        in_channels=512, out_channels=HeatmapTracker.num_targets
    )
    representation = HeatmapTracker.get_representations(fake_image_batch)
    upsampled = upsampling_layer(representation)
    assert upsampled.shape[-2:] == 2 * representation.shape[-2:]
