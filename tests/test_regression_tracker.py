"""Test the initialization and training of regression models."""

import copy
import torch
import pytest
import pytorch_lightning as pl

from pose_est_nets.utils.scripts import get_loss_factories, get_model


def test_supervised_regression(cfg, base_data_module, trainer):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = []

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=base_data_module)

    # build model
    model = get_model(
        cfg=cfg_tmp, data_module=base_data_module, loss_factories=loss_factories
    )

    trainer.fit(model=model, datamodule=base_data_module)


def test_unsupervised_regression_temporal(cfg, base_data_module_combined, trainer):
    """Test the initialization and training of an unsupervised regression model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = ["temporal"]

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(
        cfg=cfg_tmp, data_module=base_data_module_combined
    )

    # model
    model = get_model(
        cfg=cfg_tmp,
        data_module=base_data_module_combined,
        loss_factories=loss_factories,
    )

    trainer.fit(model=model, datamodule=base_data_module_combined)
