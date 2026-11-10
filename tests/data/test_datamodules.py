"""Test datamodule functionality."""

from collections.abc import Sized

import numpy as np
import pytest
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import RandomSampler, TensorDataset

from lightning_pose.data.datamodules import BaseDataModule
from lightning_pose.data.samplers import TemperatureSampler
from lightning_pose.data.utils import split_sizes_from_probabilities


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
    np.random.seed(0)
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

    # ----------------------------
    # check train batch properties
    # ----------------------------
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

    # ----------------------------
    # check val batch properties
    # ----------------------------
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

    # ----------------------------
    # check test batch properties
    # ----------------------------
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
    dataset = train_dataloader.dataset
    assert isinstance(dataset, Sized)
    assert len(dataset) == train_frames

    train_frames = 1  # integer
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    train_dataloader = heatmap_module.train_dataloader()
    dataset = train_dataloader.dataset
    assert isinstance(dataset, Sized)
    assert len(dataset) == int(train_probability * len_dataset)

    train_frames = 0.1  # fraction < 1
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    train_dataloader = heatmap_module.train_dataloader()
    dataset = train_dataloader.dataset
    assert isinstance(dataset, Sized)
    assert len(dataset) == int(train_frames * train_probability * len_dataset)

    train_frames = 1000000  # integer larger than number of labeled frames
    train_probability = 0.8
    heatmap_module = BaseDataModule(
        base_dataset, train_frames=train_frames, train_probability=train_probability
    )
    train_dataloader = heatmap_module.train_dataloader()
    dataset = train_dataloader.dataset
    assert isinstance(dataset, Sized)
    assert len(dataset) == int(train_probability * len_dataset)

    # raise exception when not a path
    with pytest.raises(ValueError):
        train_frames = -1
        BaseDataModule(base_dataset, train_frames=train_frames)


@pytest.mark.gpu
def test_base_data_module_combined(cfg, base_data_module_combined):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    train_size_labeled = base_data_module_combined.train_batch_size
    train_size_unlabeled = base_data_module_combined.dali_config["context"]["train"]["batch_size"]
    num_targets = base_data_module_combined.dataset.num_targets

    # test outputs for single batch
    loader = base_data_module_combined.train_dataloader()
    assert isinstance(loader.sampler["labeled"], RandomSampler)  # shuffle=True

    batch = next(iter(loader))
    # batch tuple in lightning >=2.0.9
    batch = batch[0] if isinstance(batch, tuple) else batch
    assert list(batch.keys())[0] == "labeled"
    assert list(batch.keys())[1] == "unlabeled"
    assert list(batch["labeled"].keys()) == ["images", "keypoints", "idxs", "bbox", "visibility"]
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


@pytest.mark.gpu
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
    assert list(batch["labeled"].keys()) == [
        "images", "keypoints", "idxs", "bbox", "visibility", "heatmaps",
    ]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]
    assert batch["labeled"]["images"].shape == (train_size_labeled, 3, im_height, im_width)
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size_labeled, num_targets // 2, im_height_ds, im_width_ds,
    )
    assert batch["unlabeled"]["frames"].shape == (train_size_unlabel, 3, im_height, im_width)


@pytest.mark.gpu
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
        "view_names", "keypoints_3d", "intrinsic_matrix", "extrinsic_matrix", "distortions",
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


@pytest.mark.gpu
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
    assert list(batch["labeled"].keys()) == [
        "images", "keypoints", "idxs", "bbox", "visibility", "heatmaps",
    ]
    assert list(batch["unlabeled"].keys()) == ["frames", "transforms", "bbox", "is_multiview"]
    assert batch["labeled"]["images"].shape == (
        train_size_labeled, num_context, 3, im_height, im_width,
    )
    assert batch["labeled"]["keypoints"].shape == (train_size_labeled, num_targets)
    assert batch["labeled"]["heatmaps"].shape == (
        train_size_labeled, num_targets // 2, im_height_ds, im_width_ds,
    )
    assert batch["unlabeled"]["frames"].shape == (train_size_unlabel, 3, im_height, im_width)


@pytest.mark.gpu
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
        "view_names", "keypoints_3d", "intrinsic_matrix", "extrinsic_matrix", "distortions",
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


class _MultiDatasetStub(torch.utils.data.Dataset):
    """Minimal labeled dataset with per-example dataset ids (no images, no augmentation)."""

    def __init__(
        self,
        frames_per_dataset: list[int],
        with_ids: bool = True,
        with_visibility: bool = True,
    ) -> None:
        n = sum(frames_per_dataset)
        self.dataset_names = [f'ds{i}' for i in range(len(frames_per_dataset))]
        self.dataset_ids = (
            torch.cat([torch.full((c,), i) for i, c in enumerate(frames_per_dataset)])
            if with_ids else None
        )
        self.image_names = [f'labeled-data/img{i:03d}.png' for i in range(n)]
        # dataset i labels i + 1 of its 3 keypoints in every frame
        visibility = torch.zeros(n, 3, dtype=torch.long)
        keypoints = torch.full((n, 3, 2), float('nan'))
        for i in range(len(frames_per_dataset)):
            rows = self.dataset_ids == i if with_ids else torch.ones(n, dtype=torch.bool)
            visibility[rows, : i + 1] = 2
            keypoints[rows, : i + 1] = 1.0
        self.visibility = visibility if with_visibility else None
        self.keypoints = keypoints
        self.imgaug_transform = [object()]   # resize only: the shared-subset path
        self.imgaug_hflip = False

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx)


def _indices(subset: torch.utils.data.Subset | None) -> list[int]:
    assert subset is not None
    return list(subset.indices)


def _data_module(dataset: _MultiDatasetStub, **kwargs) -> BaseDataModule:
    return BaseDataModule(
        dataset=dataset,  # type: ignore[arg-type]
        num_workers=0,
        train_probability=0.8,
        val_probability=0.1,
        test_probability=0.1,
        **kwargs,
    )


class TestSplitIndices:
    """Test the method BaseDataModule._split_indices."""

    def test_split_indices_stratified_by_dataset(self):
        dataset = _MultiDatasetStub([100, 20])
        dm = _data_module(dataset)
        ids = dataset.dataset_ids
        assert ids is not None
        # the split proportions are applied within each source dataset
        sizes = [split_sizes_from_probabilities(n, 0.8, 0.1, 0.1) for n in (100, 20)]
        for split, subset in enumerate([dm.train_dataset, dm.val_dataset, dm.test_dataset]):
            counts = torch.bincount(ids[_indices(subset)], minlength=2).tolist()
            assert counts == [sizes[0][split], sizes[1][split]]
        all_idxs = _indices(dm.train_dataset) + _indices(dm.val_dataset)
        all_idxs += _indices(dm.test_dataset)
        assert sorted(all_idxs) == list(range(120))

    def test_split_indices_without_ids_matches_random_split(self):
        dm = _data_module(_MultiDatasetStub([50], with_ids=False))
        expected = torch.utils.data.random_split(
            TensorDataset(torch.arange(50)), [40, 5, 5],
            generator=torch.Generator().manual_seed(42),
        )
        assert _indices(dm.train_dataset) == list(expected[0].indices)
        assert dm.split_manifest() is None


class TestSetupTrainSampler:
    """Test the method BaseDataModule._setup_train_sampler."""

    @pytest.mark.parametrize('temperature', [None, 1])
    def test_setup_train_sampler_off(self, temperature):
        dm = _data_module(_MultiDatasetStub([30, 10]), sampling_temperature=temperature)
        assert dm.train_sampler is None
        assert isinstance(dm.train_dataloader().sampler, RandomSampler)

    @pytest.mark.parametrize('with_visibility', [True, False])
    def test_setup_train_sampler_temperature(self, with_visibility):
        dataset = _MultiDatasetStub([30, 10], with_visibility=with_visibility)
        dm = _data_module(dataset, sampling_temperature=2)
        assert isinstance(dm.train_sampler, TemperatureSampler)
        n_train = [split_sizes_from_probabilities(n, 0.8, 0.1, 0.1)[0] for n in (30, 10)]
        assert dm.train_sampler.n_d.tolist() == n_train
        # supervision mass n_d * kbar_d (dataset 0 labels 1 keypoint per frame, dataset 1 two),
        # shares proportional to its square root at T = 2
        mass = np.array([n_train[0] * 1.0, n_train[1] * 2.0])
        expected = np.sqrt(mass) / np.sqrt(mass).sum()
        assert dm.train_sampler.p_d.tolist() == pytest.approx(expected.tolist())
        loader = dm.train_dataloader()
        assert loader.sampler is dm.train_sampler

    def test_setup_train_sampler_errors(self):
        with pytest.raises(ValueError, match='requires a multi-dataset corpus'):
            _data_module(_MultiDatasetStub([30], with_ids=False), sampling_temperature=2)


class TestSplitManifest:
    """Test the method BaseDataModule.split_manifest."""

    def test_split_manifest_counts_and_names(self):
        dm = _data_module(_MultiDatasetStub([100, 20]))
        manifest = dm.split_manifest()
        assert manifest is not None
        assert manifest['dataset_names'] == ['ds0', 'ds1']
        sizes = [split_sizes_from_probabilities(n, 0.8, 0.1, 0.1) for n in (100, 20)]
        assert manifest['train']['per_dataset'] == {'ds0': sizes[0][0], 'ds1': sizes[1][0]}
        assert manifest['val']['n'] == sizes[0][1] + sizes[1][1]
        assert len(manifest['test']['image_names']) == manifest['test']['n']
