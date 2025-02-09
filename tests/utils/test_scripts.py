"""Test the scripts module.

Note that many of the functions in the scripts module are explicitly used (and therefore implicitly
tested) in conftest.py

"""

import copy
import os
from unittest.mock import Mock

import lightning.pytorch as pl
import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from lightning_pose.data.datasets import BaseTrackingDataset
from lightning_pose.utils.scripts import (
    calculate_train_batches,
    get_data_module,
)


def test_calculate_train_batches(cfg, base_dataset):
    """Test the computation of train batches, which is a function of labeled and unlabeled info."""
    cfg_tmp = copy.deepcopy(cfg)

    # return value if set in config
    for n in [2, 12, 22]:
        cfg_tmp.training.limit_train_batches = n
        n_batches = calculate_train_batches(cfg_tmp, base_dataset)
        assert n_batches == n

    # None with small number of train frames - return minimum of 10
    cfg_tmp.training.limit_train_batches = None
    cfg_tmp.training.train_frames = 2
    n_batches = calculate_train_batches(cfg_tmp, base_dataset)
    # TODO: fixme
    # assert n_batches == 10

    # None with large number of frames
    n = 50
    cfg_tmp.training.limit_train_batches = None
    cfg_tmp.training.train_frames = n
    cfg_tmp.training.train_batch_size = 1
    n_batches = calculate_train_batches(cfg_tmp, base_dataset)
    assert n_batches == n


def _supervised_multi_gpu_cfg(cfg):
    return OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "model": {
                    "losses_to_use": [],
                },
                "training": {
                    "num_gpus": 2,
                    "train_batch_size": 4,
                    "val_batch_size": 16,
                    "test_batch_size": 16,
                },
            }
        ),
    )


def _unsupervised_multi_gpu_cfg(cfg):
    cfg = _supervised_multi_gpu_cfg(cfg)
    cfg.model.losses_to_use = ["temporal"]  # trigger unsupervised datamodule
    return cfg


def test_get_data_module_num_gpus_0(cfg):
    cfg = _supervised_multi_gpu_cfg(cfg)
    # when num_gpus is set to 0, i.e. from a deprecated config
    cfg.training.num_gpus = 0
    data_module = get_data_module(cfg, Mock(spec=BaseTrackingDataset))

    # assert num_gpus gets modified to 1
    assert cfg.training.num_gpus == 1
    # the rest of the behavior follows correctly
    assert data_module.train_batch_size == cfg.training.train_batch_size
    assert data_module.val_batch_size == cfg.training.val_batch_size
    assert data_module.test_batch_size == cfg.training.test_batch_size


def test_get_data_module_multi_gpu_batch_size_adjustment_supervised(cfg):
    cfg = _supervised_multi_gpu_cfg(cfg)
    data_module = get_data_module(cfg, Mock(spec=BaseTrackingDataset))
    # train, val batch sizes should be divided by num_gpus
    assert (
        data_module.train_batch_size
        == cfg.training.train_batch_size / cfg.training.num_gpus
    )
    assert (
        data_module.val_batch_size
        == cfg.training.val_batch_size / cfg.training.num_gpus
    )
    assert data_module.test_batch_size == cfg.training.test_batch_size


def test_get_data_module_multi_gpu_batch_size_adjustment_unsupervised(
    cfg, heatmap_dataset, toy_data_dir
):
    cfg = _unsupervised_multi_gpu_cfg(cfg)
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )
    # train, val batch sizes should be divided by num_gpus
    assert (
        data_module.train_batch_size
        == cfg.training.train_batch_size / cfg.training.num_gpus
    )
    assert (
        data_module.val_batch_size
        == cfg.training.val_batch_size / cfg.training.num_gpus
    )
    assert data_module.test_batch_size == cfg.training.test_batch_size

    # sequence length should be divided by num_gups
    assert (
        data_module.dali_config.base.train.sequence_length
        == cfg.dali.base.train.sequence_length / cfg.training.num_gpus
    )

    # context batch size is more nuance, tested separately


def test_get_data_module_multi_gpu_batch_size_adjustment_ceiling(
    cfg, heatmap_dataset, toy_data_dir
):
    cfg = _unsupervised_multi_gpu_cfg(cfg)
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )

    # When batch_size is indivisible by 2
    cfg.training.train_batch_size += 1
    cfg.training.val_batch_size += 1
    cfg.dali.base.train.sequence_length += 1
    cfg.dali.context.train.batch_size += 1

    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )

    # batch size should be the ceiling of batch_size divided by num_gups
    assert data_module.train_batch_size == int(
        np.ceil(cfg.training.train_batch_size / cfg.training.num_gpus)
    )
    assert data_module.val_batch_size == int(
        np.ceil(cfg.training.val_batch_size / cfg.training.num_gpus)
    )
    assert data_module.test_batch_size == cfg.training.test_batch_size

    # sequence length should be divided by num_gups
    assert data_module.dali_config.base.train.sequence_length == int(
        np.ceil(cfg.dali.base.train.sequence_length / cfg.training.num_gpus)
    )

    # context batch size is more nuance, tested separately


def test_get_data_module_multi_gpu_context_batch_size_adjustment(
    cfg, heatmap_dataset, toy_data_dir
):
    cfg = _unsupervised_multi_gpu_cfg(cfg)
    cfg.model.model_type = "heatmap_mhcrnn"
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )

    # batch size of 6 -> effective 2 -> per-gpu effective 1 -> per-gpu 5
    cfg.dali.context.train.batch_size = 6
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )
    assert data_module.dali_config.context.train.batch_size == 5

    # batch size of 5 -> effective 1 -> per-gpu effective 1 -> per-gpu 5
    cfg.dali.context.train.batch_size = 5
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )
    assert data_module.dali_config.context.train.batch_size == 5

    # batch size of 28 -> effective 24 -> per-gpu effective 12 -> per-gpu 16
    cfg.dali.context.train.batch_size = 28
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )
    assert data_module.dali_config.context.train.batch_size == 16

    # batch size of 27 -> effective 23 -> per-gpu effective 12 -> per-gpu 16
    cfg.dali.context.train.batch_size = 27
    data_module = get_data_module(
        cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
    )
    assert data_module.dali_config.context.train.batch_size == 16

    # batch size of 4 -> effective 0 -> should throw an error.
    cfg.dali.context.train.batch_size = 4
    with pytest.raises(ValidationError):
        data_module = get_data_module(
            cfg, heatmap_dataset, os.path.join(toy_data_dir, "videos")
        )
