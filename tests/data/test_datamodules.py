"""Test datamodule functionality."""

import pytest
import torch

# from pytorch_lightning.trainer.supporters import CombinedLoader
from lightning.pytorch.utilities import CombinedLoader


def test_base_datamodule(cfg, base_data_module):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    train_size = base_data_module.train_batch_size
    val_size = base_data_module.val_batch_size
    test_size = base_data_module.test_batch_size
    num_targets = base_data_module.dataset.num_targets

    # check batch properties
    batch = next(iter(base_data_module.train_dataloader()))
    assert batch["images"].shape == (train_size, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)

    batch = next(iter(base_data_module.val_dataloader()))
    assert batch["images"].shape[1:] == (3, im_height, im_width)
    assert batch["keypoints"].shape[1:] == (num_targets,)
    assert batch["images"].shape[0] == batch["keypoints"].shape[0]
    assert batch["images"].shape[0] <= val_size

    batch = next(iter(base_data_module.test_dataloader()))
    assert batch["images"].shape[1:] == (3, im_height, im_width)
    assert batch["keypoints"].shape[1:] == (num_targets,)
    assert batch["images"].shape[0] == batch["keypoints"].shape[0]
    assert batch["images"].shape[0] <= test_size

    # cleanup
    del batch


def test_heatmap_datamodule(cfg, heatmap_data_module):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg.data.downsample_factor)
    im_width_ds = im_width / (2 ** cfg.data.downsample_factor)
    train_size = heatmap_data_module.train_batch_size
    num_targets = heatmap_data_module.dataset.num_targets

    # check batch properties
    batch = next(iter(heatmap_data_module.train_dataloader()))
    assert batch["images"].shape == (train_size, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)
    assert batch["heatmaps"].shape == (
        train_size,
        num_targets // 2,
        im_height_ds,
        im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == heatmap_data_module.dataset.output_shape

    # cleanup
    del batch


def test_subsampling_of_training_frames(base_dataset):

    from lightning_pose.data.datamodules import BaseDataModule

    len_dataset = len(base_dataset)

    # test subsampling of training frames
    train_frames = 10  # integer
    heatmap_module = BaseDataModule(base_dataset, train_frames=train_frames)
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == train_frames

    train_frames = 1  # integer
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == int(train_probability * len_dataset)

    train_frames = 0.1  # fraction < 1
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == int(
        train_frames * train_probability * len_dataset
    )

    train_frames = 1000000  # integer larger than number of labeled frames
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    heatmap_module.setup()
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == int(train_probability * len_dataset)

    # raise exception when not a path
    with pytest.raises(ValueError):
        train_frames = -1
        heatmap_module = BaseDataModule(base_dataset, train_frames=train_frames)
        heatmap_module.setup()

    # cleanup
    del heatmap_module
    del train_dataloader
    torch.cuda.empty_cache()


def test_base_data_module_combined(cfg, base_data_module_combined):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    train_size = base_data_module_combined.train_batch_size
    num_targets = base_data_module_combined.dataset.num_targets

    loader = CombinedLoader(base_data_module_combined.train_dataloader())
    batch = next(iter(loader))
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == ["images", "keypoints", "idxs"]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms"]
    assert batch["labeled"]["images"].shape == (train_size, 3, im_height, im_width)
    assert batch["labeled"]["keypoints"].shape == (train_size, num_targets)
    assert batch["unlabeled"]["frames"].shape == (train_size, 3, im_height, im_width)

    # cleanup
    del loader
    del batch
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_heatmap_data_module_combined(cfg, heatmap_data_module_combined):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg.data.downsample_factor)
    im_width_ds = im_width / (2 ** cfg.data.downsample_factor)
    train_size = heatmap_data_module_combined.train_batch_size
    num_targets = heatmap_data_module_combined.dataset.num_targets

    loader = CombinedLoader(heatmap_data_module_combined.train_dataloader())
    batch = next(iter(loader))
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == ["images", "keypoints", "idxs", "heatmaps"]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms"]
    assert batch["labeled"]["images"].shape == (train_size, 3, im_height, im_width)
    assert batch["labeled"]["keypoints"].shape == (train_size, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size,
        num_targets // 2,
        im_height_ds,
        im_width_ds,
    )
    assert batch["unlabeled"]["frames"].shape == (train_size, 3, im_height, im_width)

    # cleanup
    del loader
    del batch
    torch.cuda.empty_cache()  # remove tensors from gpu
