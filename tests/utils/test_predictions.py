"""Test the predictions module."""

import copy
from unittest.mock import MagicMock

import pandas as pd
import pytest
from omegaconf import OmegaConf

from lightning_pose.losses import get_loss_factories
from lightning_pose.utils.predictions import make_dlc_pandas_index, predict_dataset
from lightning_pose.utils.scripts import get_model


class TestMakeDlcPandasIndex:
    """Test the make_dlc_pandas_index function."""

    def test_make_dlc_pandas_index_structure(self):
        """Returned MultiIndex has the correct names, levels, and length."""
        cfg = OmegaConf.create({'model': {'model_type': 'heatmap'}})
        keypoint_names = ['nose', 'left_ear', 'right_ear']

        idx = make_dlc_pandas_index(cfg=cfg, keypoint_names=keypoint_names)

        assert isinstance(idx, pd.MultiIndex)
        assert list(idx.names) == ['scorer', 'bodyparts', 'coords']
        assert len(idx) == len(keypoint_names) * 3  # x, y, likelihood per keypoint

    def test_make_dlc_pandas_index_scorer_uses_model_type(self):
        """Scorer level is '<model_type>_tracker'."""
        cfg = OmegaConf.create({'model': {'model_type': 'regression'}})

        idx = make_dlc_pandas_index(cfg=cfg, keypoint_names=['nose'])

        scorers = idx.get_level_values('scorer').unique().tolist()
        assert scorers == ['regression_tracker']

    def test_make_dlc_pandas_index_coords(self):
        """Coords level always contains exactly x, y, likelihood."""
        cfg = OmegaConf.create({'model': {'model_type': 'heatmap'}})

        idx = make_dlc_pandas_index(cfg=cfg, keypoint_names=['nose', 'tail'])

        coords = idx.get_level_values('coords').unique().tolist()
        assert coords == ['x', 'y', 'likelihood']


class TestPredictDataset:
    """Test the predict_dataset function."""

    @pytest.fixture()
    def mock_model(self, cfg, heatmap_data_module):
        """Untrained heatmap model wrapped in a minimal Model-like mock."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.model_type = 'heatmap'
        cfg_tmp.model.losses_to_use = []

        loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=heatmap_data_module)
        lightning_model = get_model(
            cfg=cfg_tmp, data_module=heatmap_data_module, loss_factories=loss_factories,
        )

        mock = MagicMock()
        mock.model = lightning_model
        mock.config.cfg = cfg_tmp
        return mock

    def test_predict_dataset_explicit_cfg(self, mock_model, cfg, heatmap_data_module, tmpdir):
        """Predictions are written when cfg is passed explicitly."""
        predict_dataset(
            model=mock_model,
            data_module=heatmap_data_module,
            preds_file=str(tmpdir.join('preds.csv')),
            cfg=cfg,
        )

    def test_predict_dataset_cfg_fallback(self, mock_model, heatmap_data_module, tmpdir):
        """Predictions are written when cfg falls back to model.config.cfg."""
        predict_dataset(
            model=mock_model,
            data_module=heatmap_data_module,
            preds_file=str(tmpdir.join('preds.csv')),
        )
