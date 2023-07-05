"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests. Their
construction relies heavily on the utility functions provided in `utils/scripts.py`.

"""

import copy
import os
import shutil
from typing import Callable, List

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from lightning_pose.data.dali import LitDaliWrapper, PrepareDALI
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import BaseTrackingDataset, HeatmapDataset
from lightning_pose.utils.io import get_videos_in_dir
from lightning_pose.utils.scripts import (
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)

TOY_DATA_ROOT_DIR = "data/mirror-mouse-example"


@pytest.fixture
def cfg() -> dict:
    """Load all toy data config file without hydra."""
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    config_file = os.path.join(base_dir, "scripts", "configs", "config_mirror-mouse-example.yaml")
    cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)
    cfg["model"]["do_context"] = False
    cfg["training"]["imgaug"] = "default"  # so pca tests don't break
    return OmegaConf.create(cfg)


@pytest.fixture
def cfg_context() -> dict:
    """Load all toy data config file without hydra."""
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    config_file = os.path.join(base_dir, "scripts", "configs", "config_mirror-mouse-example.yaml")
    cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)
    cfg["model"]["do_context"] = True
    # make small batches so that we can run on a gpu with limited memory
    cfg["training"]["train_batch_size"] = 4
    cfg["training"]["val_batch_size"] = 4
    cfg["training"]["test_batch_size"] = 4
    cfg["training"]["imgaug"] = "default"  # so pca tests don't break
    cfg["dali"]["context"]["train"]["batch_size"] = 8
    return OmegaConf.create(cfg)


@pytest.fixture
def imgaug_transform(cfg) -> iaa.Sequential:
    """Create basic resizing transform."""
    return get_imgaug_transform(cfg)


@pytest.fixture
def base_dataset(cfg, imgaug_transform) -> BaseTrackingDataset:
    """Create a dataset for regression models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    base_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield base_dataset

    # cleanup after all tests have run (no more calls to yield)
    del base_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def base_dataset_context(cfg_context, imgaug_transform) -> BaseTrackingDataset:
    """Create a dataset for regression models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.model_type = "regression"
    base_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield base_dataset

    # cleanup after all tests have run (no more calls to yield)
    del base_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_dataset(cfg, imgaug_transform) -> HeatmapDataset:
    """Create a dataset for heatmap models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    heatmap_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield heatmap_dataset

    # cleanup after all tests have run (no more calls to yield)
    del heatmap_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_dataset_context(cfg_context, imgaug_transform) -> HeatmapDataset:
    """Create a dataset for heatmap models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.model_type = "heatmap"
    heatmap_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield heatmap_dataset

    # cleanup after all tests have run (no more calls to yield)
    del heatmap_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def base_data_module(cfg, base_dataset) -> BaseDataModule:
    """Create a labeled data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = []
    # bump up training data so we can test pca_singleview loss
    cfg_tmp.training.train_prob = 0.95
    cfg_tmp.training.val_prob = 0.025
    data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def base_data_module_context(cfg_context, base_dataset_context) -> BaseDataModule:
    """Create a labeled data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.losses_to_use = []
    # bump up training data so we can test pca_singleview loss
    cfg_tmp.training.train_prob = 0.95
    cfg_tmp.training.val_prob = 0.025
    data_module = get_data_module(cfg_tmp, dataset=base_dataset_context, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module(cfg, heatmap_dataset) -> BaseDataModule:
    """Create a labeled data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = []
    # bump up training data so we can test pca_singleview loss
    cfg_tmp.training.train_prob = 0.95
    cfg_tmp.training.val_prob = 0.025
    data_module = get_data_module(cfg_tmp, dataset=heatmap_dataset, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module_context(cfg_context, heatmap_dataset_context) -> BaseDataModule:
    """Create a labeled data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.losses_to_use = []
    # bump up training data so we can test pca_singleview loss
    cfg_tmp.training.train_prob = 0.95
    cfg_tmp.training.val_prob = 0.025
    data_module = get_data_module(cfg_tmp, dataset=heatmap_dataset_context, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def base_data_module_combined(cfg, base_dataset) -> UnlabeledDataModule:
    """Create a combined data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = ["temporal"]
    data_module = get_data_module(
        cfg_tmp,
        dataset=base_dataset,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def base_data_module_combined_context(
    cfg_context, base_dataset_context
) -> UnlabeledDataModule:
    """Create a combined data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.losses_to_use = ["temporal"]
    data_module = get_data_module(
        cfg_tmp,
        dataset=base_dataset_context,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module_combined(cfg, heatmap_dataset) -> UnlabeledDataModule:
    """Create a combined data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = ["temporal"]  # trigger semi-supervised data module
    data_module = get_data_module(
        cfg_tmp,
        dataset=heatmap_dataset,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module_combined_context(
    cfg_context, heatmap_dataset_context
) -> UnlabeledDataModule:
    """Create a combined data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.losses_to_use = ["temporal"]  # trigger semi-supervised data module
    data_module = get_data_module(
        cfg_tmp,
        dataset=heatmap_dataset_context,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def video_dataloader(cfg, base_dataset, video_list) -> LitDaliWrapper:
    """Create a prediction dataloader for a new video."""

    # setup
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="base",
        dali_config=cfg.dali,
        filenames=video_list,
        resize_dims=[base_dataset.height, base_dataset.width],
    )
    video_dataloader = vid_pred_class()

    # return to tests
    yield video_dataloader

    # cleanup after all tests have run (no more calls to yield)
    del video_dataloader
    torch.cuda.empty_cache()


@pytest.fixture
def trainer(cfg) -> pl.Trainer:
    """Create a basic pytorch lightning trainer for testing models."""

    cfg.training.unfreezing_epoch = 10
    callbacks = get_callbacks(
        cfg, early_stopping=False, lr_monitor=False, ckpt_model=True, backbone_unfreeze=True)

    trainer = pl.Trainer(
        accelerator="gpu",  # TODO: control from outside
        devices=1,  # TODO: control from outside
        max_epochs=2,
        min_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        limit_train_batches=2,
    )

    return trainer


@pytest.fixture
def remove_logs() -> Callable:
    def _remove_logs():
        base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
        logging_dir = os.path.join(base_dir, "lightning_logs")
        shutil.rmtree(logging_dir)

    return _remove_logs


@pytest.fixture
def video_list() -> List[str]:
    return get_videos_in_dir(os.path.join(TOY_DATA_ROOT_DIR, "videos"))


@pytest.fixture
def toy_data_dir() -> str:
    return TOY_DATA_ROOT_DIR
