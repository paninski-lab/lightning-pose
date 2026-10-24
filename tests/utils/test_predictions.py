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
        mock.pl_precision = '32-true'
        return mock

    @pytest.mark.gpu
    def test_predict_dataset_explicit_cfg(self, mock_model, cfg, heatmap_data_module, tmpdir):
        """Predictions are written when cfg is passed explicitly."""
        predict_dataset(
            model=mock_model,
            data_module=heatmap_data_module,
            preds_file=str(tmpdir.join('preds.csv')),
            cfg=cfg,
        )

    @pytest.mark.gpu
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

    @pytest.mark.gpu
    def test_bbox_df_forwarded_to_prepare_dali(self, tmp_path, mock_model, bbox_csv):
        """PrepareDALI receives the loaded bbox_df when bbox_file is provided."""
        bbox_file, _ = bbox_csv
        mock_dali = MagicMock()

        with (
            patch('lightning_pose.utils.predictions.count_frames', return_value=3),
            patch('lightning_pose.data.video.dali.PrepareDALI', mock_dali),
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

    @pytest.mark.gpu
    def test_none_bbox_df_when_bbox_file_not_provided(self, tmp_path, mock_model):
        """PrepareDALI receives bbox_df=None when bbox_file is not provided."""
        mock_dali = MagicMock()

        with (
            patch('lightning_pose.data.video.dali.PrepareDALI', mock_dali),
            patch('lightning_pose.utils.predictions.pl.Trainer'),
            patch('lightning_pose.utils.predictions.PredictionHandler'),
        ):
            predict_video(
                video_file=str(tmp_path / 'vid.mp4'),
                model=mock_model,
            )

        call_kwargs = mock_dali.call_args.kwargs
        assert call_kwargs['bbox_df'] is None

    def test_multiview_bbox_files_loaded_per_view(self, tmp_path, mock_model, bbox_csv):
        """Multiview bbox_file list is loaded into a list of DataFrames, one per view."""
        bbox_file, _ = bbox_csv
        mock_model.config.cfg.data.view_names = ['view0', 'view1']
        mock_reader = MagicMock()

        with (
            patch('lightning_pose.utils.predictions.count_frames', return_value=3),
            patch('lightning_pose.utils.predictions.build_video_reader', mock_reader),
            patch('lightning_pose.utils.predictions.pl.Trainer'),
            patch('lightning_pose.utils.predictions.PredictionHandler'),
        ):
            predict_video(
                video_file=[str(tmp_path / 'view0.mp4'), str(tmp_path / 'view1.mp4')],
                model=mock_model,
                bbox_file=[bbox_file, bbox_file],
            )

        bbox_df = mock_reader.call_args.kwargs['bbox_df']
        assert isinstance(bbox_df, list)
        assert len(bbox_df) == 2
        assert all(len(df) == 3 for df in bbox_df)

    def test_raises_when_multiview_bbox_file_not_a_list(self, tmp_path, mock_model, bbox_csv):
        """ValueError is raised when a single bbox path is given for multiview videos."""
        bbox_file, _ = bbox_csv
        mock_model.config.cfg.data.view_names = ['view0', 'view1']

        video_files = [str(tmp_path / 'view0.mp4'), str(tmp_path / 'view1.mp4')]
        with pytest.raises(ValueError, match='list of 2 bbox files'):
            predict_video(
                video_file=video_files,  # type: ignore[arg-type]
                model=mock_model,
                bbox_file=bbox_file,
            )

    def test_raises_when_multiview_bbox_file_wrong_length(self, tmp_path, mock_model, bbox_csv):
        """ValueError is raised when the number of bbox files differs from the view count."""
        bbox_file, _ = bbox_csv
        mock_model.config.cfg.data.view_names = ['view0', 'view1']

        video_files = [str(tmp_path / 'view0.mp4'), str(tmp_path / 'view1.mp4')]
        with pytest.raises(ValueError, match='list of 2 bbox files'):
            predict_video(
                video_file=video_files,  # type: ignore[arg-type]
                model=mock_model,
                bbox_file=[bbox_file],  # type: ignore[arg-type]
            )

    def test_raises_when_singleview_bbox_file_is_a_list(self, tmp_path, mock_model, bbox_csv):
        """ValueError is raised when a list of bbox paths is given for a single video."""
        bbox_file, _ = bbox_csv
        with pytest.raises(ValueError, match='single path'):
            predict_video(
                video_file=str(tmp_path / 'vid.mp4'),
                model=mock_model,
                bbox_file=[bbox_file],  # type: ignore[arg-type]
            )

    def test_raises_on_multiview_frame_count_mismatch(self, tmp_path, mock_model, bbox_csv):
        """ValueError names the view whose bbox row count differs from its frame count."""
        bbox_file, _ = bbox_csv  # 3 rows
        mock_model.config.cfg.data.view_names = ['view0', 'view1']

        with patch('lightning_pose.utils.predictions.count_frames', return_value=10):
            with pytest.raises(ValueError, match='has 3 rows but view0.mp4 has 10 frames'):
                predict_video(
                    video_file=[str(tmp_path / 'view0.mp4'), str(tmp_path / 'view1.mp4')],
                    model=mock_model,
                    bbox_file=[bbox_file, bbox_file],
                )

    def test_raises_on_frame_count_mismatch(self, tmp_path, mock_model, bbox_csv):
        """ValueError is raised when bbox_file row count doesn't match video frame count."""
        bbox_file, _ = bbox_csv  # 3 rows

        with patch('lightning_pose.utils.predictions.count_frames', return_value=10):
            with pytest.raises(
                ValueError, match='bbox.csv has 3 rows but vid.mp4 has 10 frames',
            ):
                predict_video(
                    video_file=str(tmp_path / 'vid.mp4'),
                    model=mock_model,
                    bbox_file=bbox_file,
                )
