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
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.datasets.datamodules import UnlabeledDataModule
from pose_est_nets.models.new_heatmap_tracker import HeatmapTracker, SemiSupervisedHeatmapTracker
import imgaug.augmenters as iaa
import yaml
from pytorch_lightning.trainer.supporters import CombinedLoader

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_BATCH_SIZE = 12
_HEIGHT = 256  # TODO: should be different numbers?
_WIDTH = 256


def test_init():
    model = HeatmapTracker(num_targets=34)
    assert model.num_keypoints == 17
    assert model.num_filters_for_upsampling == 512
    assert model.coordinate_scale == 4


def test_create_double_upsampling_layer():
    fake_image_batch = torch.rand(
        size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
    )
    heatmap_model = HeatmapTracker(num_targets=34).to(_TORCH_DEVICE)
    upsampling_layer = heatmap_model.create_double_upsampling_layer(
        in_channels=512, out_channels=heatmap_model.num_keypoints
    ).to(_TORCH_DEVICE)
    representations = heatmap_model.get_representations(fake_image_batch)
    upsampled = upsampling_layer(representations)
    assert (
        torch.tensor(upsampled.shape[-2:])
        == torch.tensor(representations.shape[-2:]) * 2
    ).all()
    # now another upsampling layer with a different number of in channels
    upsampling_layer_two = heatmap_model.create_double_upsampling_layer(
        in_channels=heatmap_model.num_keypoints,
        out_channels=heatmap_model.num_keypoints,
    ).to(_TORCH_DEVICE)
    twice_upsampled = upsampling_layer_two(upsampled)
    assert (
        torch.tensor(twice_upsampled.shape[-2:])
        == torch.tensor(upsampled.shape[-2:]) * 2
    ).all()

    # TODO: revisit this test
    # # test the output
    # pix_shuff = torch.nn.PixelShuffle(2)
    # pix_shuffled = pix_shuff(representations)
    # print(pix_shuffled.shape)
    # print(representations.shape)
    # assert (
    #     torch.tensor(pix_shuffled.shape[-2:])
    #     == torch.tensor(representations.shape[-2:] * 2)
    # ).all()


def test_heatmaps_from_representations():
    fake_image_batch = torch.rand(
        size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
    )
    heatmap_model = HeatmapTracker(num_targets=34).to(_TORCH_DEVICE)
    representations = heatmap_model.get_representations(fake_image_batch)
    heatmaps = heatmap_model.heatmaps_from_representations(representations)
    assert (
        torch.tensor(heatmaps.shape[-2:])
        == torch.tensor(fake_image_batch.shape[-2:])
        // (2 ** heatmap_model.downsample_factor)
    ).all()


def test_unsupervised():  # TODO Finish writing test
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform
    )
    # video_directory = os.path.join(
    #     "/home/jovyan/mouseRunningData/unlabeled_videos"
    # )  # DAN's
    video_directory = os.path.join("unlabeled_videos")  # NICK's
    video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]
    assert os.path.exists(video_files[0])
    with open("pose_est_nets/losses/default_hypers.yaml") as f:
        loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)

    datamod = UnlabeledDataModule(
        dataset = dataset,
        video_paths_list=video_files[0], 
        specialized_dataprep="pca", 
        loss_param_dict = loss_param_dict
    )
    datamod.setup()
    
    semi_super_losses_to_use = ["pca"]
    model = SemiSupervisedHeatmapTracker(
        resnet_version = 18,
        num_targets=34,
        loss_params = loss_param_dict,
        semi_super_losses_to_use = semi_super_losses_to_use,
        output_shape = dataset.output_shape
    ).to(_TORCH_DEVICE)
    loader = CombinedLoader(datamod.train_dataloader())
    out = next(iter(loader))
    assert list(out.keys())[0] == "labeled"
    assert list(out.keys())[1] == "unlabeled"
    assert out["unlabeled"].shape == (
        datamod.train_batch_size,
        3,
        384,
        384,
    )
    print(out["labeled"][0].device)
    print(out["unlabeled"].device)
    print(model.device)
    out_heatmaps_labeled = model.forward(out["labeled"][0].to(_TORCH_DEVICE))
    out_heatmaps_unlabeled = model.forward(out["unlabeled"])

    assert out_heatmaps_labeled.shape == (
        datamod.train_batch_size,
        model.num_keypoints,
        384 // (2 ** model.downsample_factor),
        384 // (2 ** model.downsample_factor),
    )

    assert out_heatmaps_unlabeled.shape == (
        datamod.train_batch_size,
        model.num_keypoints,
        384 // (2 ** model.downsample_factor),
        384 // (2 ** model.downsample_factor),
    )

    spm_l, spm_u = model.run_subpixelmaxima(out_heatmaps_labeled, out_heatmaps_unlabeled)

    print(spm_l.shape, spm_u.shape)
    assert(spm_l.shape == (datamod.train_batch_size, model.num_targets))
    
    trainer = pl.Trainer(
        gpus=1 if _TORCH_DEVICE == "cuda" else 0,
        max_epochs=1,
        log_every_n_steps=1,
        auto_scale_batch_size=False,
    )  # auto_scale_batch_size not working
    trainer.fit(model=model, datamodule=datamod)
