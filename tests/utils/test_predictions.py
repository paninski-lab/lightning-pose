"""Test the predictions module."""

import copy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from lightning_pose.losses import get_loss_factories
from lightning_pose.models import get_model
from lightning_pose.utils.predictions import make_dlc_pandas_index, predict_dataset, predict_video


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


class TestPredictVideoBboxFile:
    """Test bbox_file support in predict_video."""

    @pytest.fixture
    def mock_model(self):
        """Minimal Model mock compatible with predict_video."""
        model = MagicMock()
        model.config.cfg.model.model_type = 'heatmap'
        model.config.cfg.data.image_resize_dims.height = 256
        model.config.cfg.data.image_resize_dims.width = 256
        return model

    @pytest.fixture
    def bbox_csv(self, tmp_path):
        """3-row bbox CSV; returns (path, dataframe)."""
        df = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'h': [10, 10, 10],
            'w': [10, 10, 10],
        })
        path = tmp_path / 'bbox.csv'
        df.to_csv(path)
        return path, df

    def test_bbox_df_forwarded_to_prepare_dali(self, tmp_path, mock_model, bbox_csv):
        """PrepareDALI receives the loaded bbox_df when bbox_file is provided."""
        bbox_file, _ = bbox_csv
        mock_dali = MagicMock()

        with (
            patch('lightning_pose.utils.predictions.count_frames', return_value=3),
            patch('lightning_pose.utils.predictions.PrepareDALI', mock_dali),
            patch('lightning_pose.utils.predictions.pl.Trainer'),
            patch('lightning_pose.utils.predictions.PredictionHandler'),
        ):
            predict_video(
                video_file=str(tmp_path / 'vid.mp4'),
                model=mock_model,
                bbox_file=bbox_file,
            )

        call_kwargs = mock_dali.call_args.kwargs
        assert call_kwargs['bbox_df'] is not None
        assert list(call_kwargs['bbox_df'].columns) == ['x', 'y', 'h', 'w']
        assert len(call_kwargs['bbox_df']) == 3

    def test_none_bbox_df_when_bbox_file_not_provided(self, tmp_path, mock_model):
        """PrepareDALI receives bbox_df=None when bbox_file is not provided."""
        mock_dali = MagicMock()

        with (
            patch('lightning_pose.utils.predictions.PrepareDALI', mock_dali),
            patch('lightning_pose.utils.predictions.pl.Trainer'),
            patch('lightning_pose.utils.predictions.PredictionHandler'),
        ):
            predict_video(
                video_file=str(tmp_path / 'vid.mp4'),
                model=mock_model,
            )

        call_kwargs = mock_dali.call_args.kwargs
        assert call_kwargs['bbox_df'] is None

    def test_raises_on_frame_count_mismatch(self, tmp_path, mock_model, bbox_csv):
        """ValueError is raised when bbox_file row count doesn't match video frame count."""
        bbox_file, _ = bbox_csv  # 3 rows

        with patch('lightning_pose.utils.predictions.count_frames', return_value=10):
            with pytest.raises(
                ValueError, match='bbox_file has 3 rows but video has 10 frames',
            ):
                predict_video(
                    video_file=str(tmp_path / 'vid.mp4'),
                    model=mock_model,
                    bbox_file=bbox_file,
                )
