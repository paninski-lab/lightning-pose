"""Test the predict CLI command argument parsing."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lightning_pose.cli.commands.predict import _predict_multi_type, get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_predict(self):
        """Returned parser has prog set to 'litpose predict'."""
        assert get_parser().prog == 'litpose predict'


class TestPredictParser:
    """Test the predict subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video = tmp_path / 'video.mp4'
        args = parser.parse_args(['predict', str(model_dir), str(video)])
        assert args.model_dir == model_dir
        assert args.input_path == [video]
        assert not args.skip_viz
        assert not args.overwrite

    def test_multiple_input_paths(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video1 = tmp_path / 'video1.mp4'
        video2 = tmp_path / 'video2.mp4'
        args = parser.parse_args(['predict', str(model_dir), str(video1), str(video2)])
        assert args.input_path == [video1, video2]

    def test_missing_model_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['predict', str(tmp_path / 'missing'), str(tmp_path / 'video.mp4')]
            )

    def test_skip_viz_flag(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--skip_viz'])
        assert args.skip_viz

    def test_overwrite_flag(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--overwrite'])
        assert args.overwrite

    def test_overrides(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        override = 'dali.base.predict.batch_size=4'
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--overrides', override])
        assert args.overrides == [override]

    def test_bbox_dir_default_is_none(self, parser, tmp_path):
        """bbox_dir defaults to None when not provided."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4'])
        assert args.bbox_dir is None

    def test_bbox_dir_arg(self, parser, tmp_path):
        """--bbox_dir is parsed as a Path."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        bbox_dir = tmp_path / 'bboxes'
        args = parser.parse_args([
            'predict', str(model_dir), 'labels.csv', '--bbox_dir', str(bbox_dir),
        ])
        assert args.bbox_dir == bbox_dir


class TestPredictMultiType:
    """Test _predict_multi_type bbox threading."""

    @pytest.fixture
    def mock_model(self, tmp_path):
        """Mock Model with path methods returning real paths."""
        model = MagicMock()
        model.image_preds_dir.return_value = tmp_path / 'image_preds'
        model.video_preds_dir.return_value = tmp_path / 'video_preds'
        return model

    def test_csv_passes_bbox_file_when_bbox_dir_given(self, tmp_path, mock_model):
        """CSV input with bbox_dir passes <bbox_dir>/bbox.csv to predict_on_label_csv."""
        bbox_dir = tmp_path / 'bboxes'
        csv_path = tmp_path / 'labels.csv'
        _predict_multi_type(
            mock_model, csv_path, skip_viz=True, skip_existing=False, bbox_dir=bbox_dir,
        )
        mock_model.predict_on_label_csv.assert_called_once()
        call_kwargs = mock_model.predict_on_label_csv.call_args.kwargs
        assert call_kwargs['bbox_file'] == bbox_dir / 'bbox.csv'

    def test_csv_passes_none_bbox_file_when_no_bbox_dir(self, tmp_path, mock_model):
        """CSV input without bbox_dir passes bbox_file=None to predict_on_label_csv."""
        csv_path = tmp_path / 'labels.csv'
        _predict_multi_type(
            mock_model, csv_path, skip_viz=True, skip_existing=False, bbox_dir=None,
        )
        call_kwargs = mock_model.predict_on_label_csv.call_args.kwargs
        assert call_kwargs['bbox_file'] is None

    def test_mp4_does_not_use_bbox_dir(self, tmp_path, mock_model):
        """MP4 input ignores bbox_dir (video sub-issue not yet implemented)."""
        bbox_dir = tmp_path / 'bboxes'
        video_path = tmp_path / 'vid.mp4'
        _predict_multi_type(
            mock_model, video_path, skip_viz=True, skip_existing=False, bbox_dir=bbox_dir,
        )
        mock_model.predict_on_video_file.assert_called_once()
        call_kwargs = mock_model.predict_on_video_file.call_args.kwargs
        assert 'bbox_file' not in call_kwargs

    def test_directory_input_with_bbox_dir_recurses_into_mp4s(self, tmp_path, mock_model):
        """A directory input with bbox_dir passes through to each recursive mp4 call."""
        video_dir = tmp_path / 'videos'
        video_dir.mkdir()
        (video_dir / 'a.mp4').touch()
        bbox_dir = tmp_path / 'bboxes'
        _predict_multi_type(
            mock_model, video_dir, skip_viz=True, skip_existing=False, bbox_dir=bbox_dir,
        )
        mock_model.predict_on_video_file.assert_called_once()


class TestHandle:
    """Test the handle function."""

    @pytest.fixture
    def mock_model(self):
        """Mock Model returned by Model.from_dir2."""
        model = MagicMock()
        model.config.is_multi_view.return_value = False
        return model

    def _make_args(self, tmp_path, video, bbox_dir=None):
        return argparse.Namespace(
            model_dir=tmp_path / 'model',
            input_path=[video],
            skip_viz=False,
            overwrite=False,
            overrides=None,
            progress_file=None,
            bbox_dir=bbox_dir,
        )

    def test_handle_threads_bbox_dir_to_predict_multi_type(self, tmp_path, mock_model):
        """handle() forwards args.bbox_dir to _predict_multi_type."""
        bbox_dir = tmp_path / 'bboxes'
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', bbox_dir=bbox_dir)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch(
                'lightning_pose.cli.commands.predict._predict_multi_type',
            ) as mock_predict,
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        assert mock_predict.call_args.kwargs['bbox_dir'] == bbox_dir

    def test_handle_threads_none_bbox_dir_when_not_provided(self, tmp_path, mock_model):
        """handle() passes bbox_dir=None to _predict_multi_type when not provided."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', bbox_dir=None)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch(
                'lightning_pose.cli.commands.predict._predict_multi_type',
            ) as mock_predict,
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        assert mock_predict.call_args.kwargs['bbox_dir'] is None
