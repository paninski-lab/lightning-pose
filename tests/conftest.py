"""Provide pytest fixtures for the entire test suite.

These fixtures create datasets and data modules that can be reused by other tests. Their
construction relies heavily on the utility functions provided in `utils/scripts.py`.

"""

import copy
import imgaug.augmenters as iaa
from omegaconf import ListConfig, OmegaConf
import os
import pytest
import pytorch_lightning as pl
import torch
from typing import List, Optional
import yaml

from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.utils.scripts import (
    get_data_module,
    get_dataset,
    get_imgaug_tranform,
)
from pose_est_nets.utils.plotting_utils import get_videos_in_dir

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOY_DATA_ROOT_DIR = "toy_datasets/toymouseRunningData"


@pytest.fixture
def cfg() -> dict:
    """Load all toy data config files without hydra."""

    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    config_dir = os.path.join(base_dir, "scripts", "configs")

    keys = ["data", "losses", "model", "training"]
    cfg = {}

    for key in keys:
        cfg_tmp = os.path.join(config_dir, key, "%s_params.yaml" % key)
        with open(cfg_tmp) as f:
            dict_tmp = yaml.load(f, Loader=yaml.FullLoader)
        cfg[key] = dict_tmp

    return OmegaConf.create(cfg)


@pytest.fixture
def imgaug_transform(cfg) -> iaa.Sequential:
    """Create basic resizing transform."""
    return get_imgaug_tranform(cfg)


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
def base_data_module(cfg, base_dataset) -> BaseDataModule:
    """Create a labeled data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = [""]
    data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)

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
    data_module = get_data_module(cfg_tmp, dataset=heatmap_dataset, video_dir=None)

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
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "unlabeled_videos")
    )

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
    cfg_tmp.model.losses_to_use = ["temporal"]
    data_module = get_data_module(
        cfg_tmp,
        dataset=heatmap_dataset,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "unlabeled_videos")
    )

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def trainer(cfg) -> pl.Trainer:
    """Create a basic pytorch lightning trainer for testing models."""

    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_supervised_loss"
    )
    transfer_unfreeze_callback = pl.callbacks.BackboneFinetuning(
        unfreeze_backbone_at_epoch=10,
        lambda_func=lambda epoch: 1.5,
        backbone_initial_ratio_lr=0.1,
        should_align=True,
        train_bn=True,
    )
    # determine gpu setup
    if _TORCH_DEVICE == "cpu":
        gpus = 0
    elif isinstance(cfg.training.gpu_id, list):
        gpus = cfg.training.gpu_id
    elif isinstance(cfg.training.gpu_id, ListConfig):
        gpus = list(cfg.training.gpu_id)
    elif isinstance(cfg.training.gpu_id, int):
        gpus = [cfg.training.gpu_id]
    else:
        raise NotImplementedError(
            "training.gpu_id must be list or int, not {}".format(
                type(cfg.training.gpu_id)
            )
        )
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=2,
        min_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[ckpt_callback, transfer_unfreeze_callback],
        limit_train_batches=2,
    )

    return trainer


# @pytest.fixture
# def unlabeled_data_module(

# @pytest.fixture
# def video_list() -> List[str]:
#     return get_videos_in_dir("toy_datasets/toymouseRunningData/unlabeled_videos")
