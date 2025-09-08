"""Test datamodule functionality."""

import numpy as np
import pytest
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import WeightedRandomSampler, RandomSampler, DataLoader, BatchSampler

from lightning_pose.data.datamodules import BaseDataModule

def test_base_datamodule(cfg, base_data_module):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    train_size = base_data_module.train_batch_size
    val_size = base_data_module.val_batch_size
    test_size = base_data_module.test_batch_size
    num_targets = base_data_module.dataset.num_targets

    # check train batch properties
    train_dataloader = base_data_module.train_dataloader()
    assert isinstance(train_dataloader.sampler, RandomSampler)
    batch = next(iter(train_dataloader))
    assert batch["images"].shape == (train_size, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)
    # check imgaug pipeline makes no-repeatable data
    b1 = base_data_module.train_dataset[0]
    b2 = base_data_module.train_dataset[0]
    assert not np.allclose(b1["images"], b2["images"])
    assert not np.allclose(b1["keypoints"], b2["keypoints"], equal_nan=True)

    # check val batch properties
    val_dataloader = base_data_module.val_dataloader()
    batch = next(iter(val_dataloader))
    assert not isinstance(val_dataloader.sampler, RandomSampler)
    assert batch["images"].shape[1:] == (3, im_height, im_width)
    assert batch["keypoints"].shape[1:] == (num_targets,)
    assert batch["images"].shape[0] == batch["keypoints"].shape[0]
    assert batch["images"].shape[0] <= val_size
    # check imgaug pipeline makes repeatable data
    b1 = base_data_module.val_dataset[0]
    b2 = base_data_module.val_dataset[0]
    assert np.allclose(b1["images"], b2["images"])
    assert np.allclose(b1["keypoints"], b2["keypoints"], equal_nan=True)

    test_dataloader = base_data_module.test_dataloader()
    batch = next(iter(test_dataloader))
    assert not isinstance(test_dataloader.sampler, RandomSampler)
    assert batch["images"].shape[1:] == (3, im_height, im_width)
    assert batch["keypoints"].shape[1:] == (num_targets,)
    assert batch["images"].shape[0] == batch["keypoints"].shape[0]
    assert batch["images"].shape[0] <= test_size
    # check imgaug pipeline makes repeatable data
    b1 = base_data_module.test_dataset[0]
    b2 = base_data_module.test_dataset[0]
    assert np.allclose(b1["images"], b2["images"])
    assert np.allclose(b1["keypoints"], b2["keypoints"], equal_nan=True)


def test_heatmap_datamodule(cfg, heatmap_data_module):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg.data.get("downsample_factor", 2))
    train_size = heatmap_data_module.train_batch_size
    num_targets = heatmap_data_module.dataset.num_targets

    # check batch properties
    batch = next(iter(heatmap_data_module.train_dataloader()))
    assert batch["images"].shape == (train_size, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)
    assert batch["heatmaps"].shape == (
        train_size, num_targets // 2, im_height_ds, im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == heatmap_data_module.dataset.output_shape


def test_multiview_heatmap_datamodule(cfg_multiview, multiview_heatmap_data_module):

    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    train_size = multiview_heatmap_data_module.train_batch_size
    val_size = multiview_heatmap_data_module.val_batch_size
    test_size = multiview_heatmap_data_module.test_batch_size
    num_targets = multiview_heatmap_data_module.dataset.num_targets
    num_view = multiview_heatmap_data_module.dataset.num_views

    # check train batch properties
    batch = next(iter(multiview_heatmap_data_module.train_dataloader()))
    assert batch["images"].shape == (train_size, num_view, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)
    assert batch["heatmaps"].shape == (
        train_size, int(num_targets / 2), im_height_ds, im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == multiview_heatmap_data_module.dataset.output_shape
    # check imgaug pipeline makes no-repeatable data
    b1 = multiview_heatmap_data_module.train_dataset[0]
    b2 = multiview_heatmap_data_module.train_dataset[0]
    assert not np.allclose(b1["images"], b2["images"])
    assert not np.allclose(b1["keypoints"], b2["keypoints"], equal_nan=True)

    # check val batch properties
    val_dataloader = multiview_heatmap_data_module.val_dataloader()
    batch = next(iter(val_dataloader))
    assert batch["images"].shape == (val_size, num_view, 3, im_height, im_width)
    assert batch["keypoints"].shape == (val_size, num_targets)
    assert batch["heatmaps"].shape == (
        val_size, int(num_targets / 2), im_height_ds, im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == multiview_heatmap_data_module.dataset.output_shape
    # check imgaug pipeline makes repeatable data
    b1 = multiview_heatmap_data_module.val_dataset[0]
    b2 = multiview_heatmap_data_module.val_dataset[0]
    assert np.allclose(b1["images"], b2["images"])
    assert np.allclose(b1["keypoints"], b2["keypoints"], equal_nan=True)

    test_dataloader = multiview_heatmap_data_module.test_dataloader()
    batch = next(iter(test_dataloader))
    assert batch["images"].shape == (test_size, num_view, 3, im_height, im_width)
    assert batch["keypoints"].shape == (test_size, num_targets)
    assert batch["heatmaps"].shape == (
        test_size, int(num_targets / 2), im_height_ds, im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == multiview_heatmap_data_module.dataset.output_shape
    # check imgaug pipeline makes repeatable data
    b1 = multiview_heatmap_data_module.test_dataset[0]
    b2 = multiview_heatmap_data_module.test_dataset[0]
    assert np.allclose(b1["images"], b2["images"])
    assert np.allclose(b1["keypoints"], b2["keypoints"], equal_nan=True)


def test_heatmap_datamodule_context(cfg, heatmap_data_module_context):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg.data.get("downsample_factor", 2))
    train_size = heatmap_data_module_context.train_batch_size
    num_targets = heatmap_data_module_context.dataset.num_targets
    num_context = 5

    # check batch properties
    batch = next(iter(heatmap_data_module_context.train_dataloader()))
    assert batch["images"].shape == (train_size, num_context, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)
    assert batch["heatmaps"].shape == (
        train_size, num_targets // 2, im_height_ds, im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == heatmap_data_module_context.dataset.output_shape


def test_multiview_heatmap_datamodule_context(
    cfg_multiview,
    multiview_heatmap_data_module_context,
):
    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    train_size = multiview_heatmap_data_module_context.train_batch_size
    num_targets = multiview_heatmap_data_module_context.dataset.num_targets
    num_view = multiview_heatmap_data_module_context.dataset.num_views

    # check batch properties
    batch = next(iter(multiview_heatmap_data_module_context.train_dataloader()))
    assert batch["images"].shape == (train_size, num_view, 5, 3, im_height, im_width)
    assert batch["keypoints"].shape == (train_size, num_targets)
    assert batch["heatmaps"].shape == (
        train_size, int(num_targets / 2), im_height_ds, im_width_ds,
    )
    assert batch["heatmaps"].shape[2:] == \
        multiview_heatmap_data_module_context.dataset.output_shape


def test_subsampling_of_training_frames(base_dataset):

    from lightning_pose.data.datamodules import BaseDataModule

    len_dataset = len(base_dataset)

    # test subsampling of training frames
    train_frames = 10  # integer
    heatmap_module = BaseDataModule(base_dataset, train_frames=train_frames)
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == train_frames

    train_frames = 1  # integer
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == int(train_probability * len_dataset)

    train_frames = 0.1  # fraction < 1
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == int(
        train_frames * train_probability * len_dataset
    )

    train_frames = 1000000  # integer larger than number of labeled frames
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    train_dataloader = heatmap_module.train_dataloader()
    assert len(train_dataloader.dataset) == int(train_probability * len_dataset)

    # raise exception when not a path
    with pytest.raises(ValueError):
        train_frames = -1
        BaseDataModule(base_dataset, train_frames=train_frames)


def test_base_data_module_combined(cfg, base_data_module_combined):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    train_size_labeled = base_data_module_combined.train_batch_size
    train_size_unlabeled = base_data_module_combined.dali_config["context"]["train"]["batch_size"]
    num_targets = base_data_module_combined.dataset.num_targets

    # test outputs for single batch
    loader = base_data_module_combined.train_dataloader()
    assert isinstance(loader.sampler["labeled"], RandomSampler) # shuffle=True

    batch = next(iter(loader))
    # batch tuple in lightning >=2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == ["images", "keypoints", "idxs", "bbox"]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]
    assert batch["labeled"]["images"].shape == (train_size_labeled, 3, im_height, im_width)
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["unlabeled"]["frames"].shape == (train_size_unlabeled, 3, im_height, im_width)

    # cleanup
    del loader
    del batch

    # test iterating through combined data loader
    dataset_length = len(base_data_module_combined.train_dataset)
    loader = CombinedLoader(
        base_data_module_combined.train_dataloader().iterables, mode="min_size",
    )
    image_counter = 0
    for batch in loader:
        batch = batch[0] if isinstance(batch, tuple) else batch
        image_counter += len(batch["labeled"]["keypoints"])
    assert image_counter == dataset_length


def test_heatmap_data_module_combined(cfg, heatmap_data_module_combined):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg.data.get("downsample_factor", 2))
    train_size_labeled = heatmap_data_module_combined.train_batch_size
    train_size_unlabel = heatmap_data_module_combined.dali_config["context"]["train"]["batch_size"]
    num_targets = heatmap_data_module_combined.dataset.num_targets

    loader = heatmap_data_module_combined.train_dataloader()
    batch = next(iter(loader))
    # batch is tuple as of lightning 2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == ["images", "keypoints", "idxs", "bbox", "heatmaps"]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]
    assert batch["labeled"]["images"].shape == (train_size_labeled, 3, im_height, im_width)
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size_labeled, num_targets // 2, im_height_ds, im_width_ds,
    )
    assert batch["unlabeled"]["frames"].shape == (train_size_unlabel, 3, im_height, im_width)


def test_multiview_heatmap_data_module_combined(
    cfg_multiview,
    multiview_heatmap_data_module_combined,
):

    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    train_size_labeled = multiview_heatmap_data_module_combined.train_batch_size
    train_size_unlabeled = \
        multiview_heatmap_data_module_combined.dali_config["base"]["train"]["sequence_length"]
    num_targets = multiview_heatmap_data_module_combined.dataset.num_targets
    num_views = len(cfg_multiview.data.view_names)

    loader = multiview_heatmap_data_module_combined.train_dataloader()
    batch = next(iter(loader))
    # batch is tuple as of lightning 2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch

    # check batch contains the expected components
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == [
        "images", "keypoints", "heatmaps", "bbox", "idxs", "num_views", "concat_order",
        "view_names",
    ]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]

    # check labeled batch shapes
    assert batch["labeled"]["images"].shape == (
        train_size_labeled, num_views, 3, im_height, im_width,
    )
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size_labeled, num_targets // 2, im_height_ds, im_width_ds,
    )

    # check unlabled batch shapes
    assert batch["unlabeled"]["frames"].shape == (
        train_size_unlabeled, num_views, 3, im_height, im_width,
    )
    assert batch["unlabeled"]["transforms"].shape == (num_views, 1, 2, 3)
    assert batch["unlabeled"]["bbox"].shape == (train_size_unlabeled, num_views * 4)


def test_heatmap_data_module_combined_context(cfg, heatmap_data_module_combined_context):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg.data.get("downsample_factor", 2))
    train_size_labeled = heatmap_data_module_combined_context.train_batch_size
    train_size_unlabel = \
        heatmap_data_module_combined_context.dali_config["context"]["train"]["batch_size"]
    num_targets = heatmap_data_module_combined_context.dataset.num_targets
    num_context = 5

    loader = heatmap_data_module_combined_context.train_dataloader()
    batch = next(iter(loader))
    # batch is tuple as of lightning 2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == ["images", "keypoints", "idxs", "bbox", "heatmaps"]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]
    assert batch["labeled"]["images"].shape == (
        train_size_labeled, num_context, 3, im_height, im_width,
    )
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size_labeled, num_targets // 2, im_height_ds, im_width_ds,
    )
    assert batch["unlabeled"]["frames"].shape == (train_size_unlabel, 3, im_height, im_width)


def test_multiview_heatmap_data_module_combined_context(
    cfg_multiview,
    multiview_heatmap_data_module_combined_context,
):

    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    im_height_ds = im_height / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    im_width_ds = im_width / (2 ** cfg_multiview.data.get("downsample_factor", 2))
    train_size_labeled = multiview_heatmap_data_module_combined_context.train_batch_size
    train_size_unlabeled = multiview_heatmap_data_module_combined_context.dali_config[
        "base"
    ]["train"]["sequence_length"]
    num_targets = multiview_heatmap_data_module_combined_context.dataset.num_targets
    num_views = len(cfg_multiview.data.view_names)
    num_context = 5

    loader = multiview_heatmap_data_module_combined_context.train_dataloader()
    batch = next(iter(loader))
    # batch is tuple as of lightning 2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch

    # check batch contains the expected components
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == [
        "images", "keypoints", "heatmaps", "bbox", "idxs", "num_views", "concat_order",
        "view_names",
    ]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]

    # check labeled batch shapes
    assert batch["labeled"]["images"].shape == (
        train_size_labeled, num_views, num_context, 3, im_height, im_width,
    )
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size_labeled, num_targets // 2, im_height_ds, im_width_ds,
    )

    # check unlabled batch shapes
    assert batch["unlabeled"]["frames"].shape == (
        train_size_unlabeled, num_views, 3, im_height, im_width,
    )
    assert batch["unlabeled"]["transforms"].shape == (num_views, 1, 2, 3)
    assert batch["unlabeled"]["bbox"].shape == (train_size_unlabeled, num_views * 4)

@pytest.mark.parametrize("enable_weighted_sampler", [True, False])
def test_weighted_sampler_optionality(base_dataset, enable_weighted_sampler, capsys): # Add capsys to capture prints
    """Verify that WeightedRandomSampler is used only when configured."""

    print(f"\nInstantiating BaseDataModule with enable_weighted_sampler={enable_weighted_sampler}")
    data_module = BaseDataModule(
        dataset=base_dataset,
        train_batch_size=4,
        val_batch_size=4,
        test_batch_size=4,
        num_workers=0,
        train_probability=0.8,
        enable_weighted_sampler=enable_weighted_sampler,
        torch_seed=123,
    )
    # Capture print output during setup
    captured_setup = capsys.readouterr()
    print("--- Output during BaseDataModule setup ---")
    print(captured_setup.out)
    print(captured_setup.err)
    print("-----------------------------------------")

    train_loader = data_module.train_dataloader()

    # Debugging prints
    print(f"\n--- Testing with enable_weighted_sampler={enable_weighted_sampler} ---")
    print(f"data_module.train_sampler type: {type(data_module.train_sampler)}")
    print(f"train_loader type: {type(train_loader)}")
    # DataLoader usually wraps the sampler in a BatchSampler. Accessing .sampler might be tricky.
    # Let's inspect the batch_sampler attribute
    print(f"train_loader.batch_sampler type: {type(train_loader.batch_sampler)}")
    if hasattr(train_loader.batch_sampler, 'sampler'):
        print(f"train_loader.batch_sampler.sampler type: {type(train_loader.batch_sampler.sampler)}")
    else:
         print("train_loader.batch_sampler has no 'sampler' attribute")
    print(f"train_loader.shuffle: {getattr(train_loader, 'shuffle', 'N/A')}") # shuffle attr might not exist depending on init
    print("-----------------------------------------")


    if enable_weighted_sampler:
        # Case 1: Sampler was enabled AND successfully created
        if data_module.train_sampler is not None:
            print("Checking case: Sampler enabled and created.")
            assert isinstance(data_module.train_sampler, WeightedRandomSampler), \
                "data_module.train_sampler should be WeightedRandomSampler instance"
            # When a sampler is provided, shuffle=False is passed to DataLoader init.
            # The DataLoader instance itself might not have a .shuffle attribute after init.
            # The key is that the batch_sampler's underlying sampler is ours.
            assert isinstance(train_loader.batch_sampler, BatchSampler), \
                "DataLoader should use a BatchSampler"
            assert isinstance(train_loader.batch_sampler.sampler, WeightedRandomSampler), \
                "BatchSampler's underlying sampler should be WeightedRandomSampler"

        # Case 2: Sampler was enabled BUT creation failed (fallback)
        else:
            print("Checking case: Sampler enabled but creation failed (fallback).")
            # Check our internal state
            assert data_module.train_sampler is None, \
                "data_module.train_sampler should be None if creation failed"
            # Check the DataLoader's resulting state (should be like sampler disabled)
            assert isinstance(train_loader.batch_sampler, BatchSampler), \
                "DataLoader should use a BatchSampler"
            # When shuffle=True, DataLoader uses RandomSampler internally
            assert isinstance(train_loader.batch_sampler.sampler, RandomSampler), \
                "BatchSampler's underlying sampler should be RandomSampler in fallback"

    else: # enable_weighted_sampler is False
        print("Checking case: Sampler disabled.")
        # Check our internal state
        assert data_module.train_sampler is None, \
            "data_module.train_sampler should be None when disabled"
        # Check the DataLoader's resulting state
        assert isinstance(train_loader.batch_sampler, BatchSampler), \
            "DataLoader should use a BatchSampler"
        # When shuffle=True, DataLoader uses RandomSampler internally
        assert isinstance(train_loader.batch_sampler.sampler, RandomSampler), \
            "BatchSampler's underlying sampler should be RandomSampler when disabled"
