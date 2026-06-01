"""Test the crop CLI command argument parsing."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lightning_pose.cli.commands.crop import get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_crop(self):
        """Returned parser has prog set to 'litpose crop'."""
        assert get_parser().prog == 'litpose crop'


class TestCropParser:
    """Test the crop subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video = tmp_path / 'video.mp4'
        args = parser.parse_args(['crop', str(model_dir), str(video)])
        assert args.model_dir == model_dir
        assert args.input_path == [video]
        assert args.bbox_dir is None

    def test_missing_model_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['crop', str(tmp_path / 'missing'), str(tmp_path / 'video.mp4')]
            )

    def test_bbox_dir(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        bbox_dir = tmp_path / 'bboxes'
        args = parser.parse_args(
            ['crop', str(model_dir), 'video.mp4', '--bbox_dir', str(bbox_dir)]
        )
        assert args.bbox_dir == bbox_dir

    def test_multiple_input_paths(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video1 = tmp_path / 'video1.mp4'
        video2 = tmp_path / 'video2.mp4'
        args = parser.parse_args(['crop', str(model_dir), str(video1), str(video2)])
        assert args.input_path == [video1, video2]


class TestHandle:
    """Test the handle function."""

    @pytest.fixture
    def mock_model(self, tmp_path):
        """Mock Model instance returned by Model.from_dir."""
        model = MagicMock()
        model.config.cfg.data.data_dir = str(tmp_path / 'data')
        return model

    def _make_args(self, tmp_path, input_path, bbox_dir=None):
        return argparse.Namespace(
            model_dir=tmp_path / 'model',
            input_path=[input_path],
            bbox_dir=bbox_dir,
        )

    def test_mp4_calls_crop_video(self, tmp_path, mock_model):
        """MP4 input triggers crop_video with default bbox path."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.crop_video') as mock_crop,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_crop.assert_called_once()
        call_kwargs = mock_crop.call_args.kwargs
        assert call_kwargs['input_video_file'] == tmp_path / 'vid.mp4'

    def test_mp4_with_bbox_dir(self, tmp_path, mock_model):
        """MP4 input with --bbox_dir uses the provided directory for bbox lookup."""
        bbox_dir = tmp_path / 'bboxes'
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', bbox_dir=bbox_dir)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.crop_video') as mock_crop,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        call_kwargs = mock_crop.call_args.kwargs
        assert call_kwargs['input_bbox_file'] == bbox_dir / 'vid_bbox.csv'

    def test_csv_calls_crop_labeled_frames(self, tmp_path, mock_model):
        """CSV input triggers crop_labeled_frames."""
        args = self._make_args(tmp_path, tmp_path / 'labels.csv')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.crop_labeled_frames') as mock_crop,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_crop.assert_called_once()

    def test_csv_with_bbox_dir(self, tmp_path, mock_model):
        """CSV input with --bbox_dir looks for bbox.csv inside that directory."""
        bbox_dir = tmp_path / 'bboxes'
        args = self._make_args(tmp_path, tmp_path / 'labels.csv', bbox_dir=bbox_dir)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.crop_labeled_frames') as mock_crop,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        call_kwargs = mock_crop.call_args.kwargs
        assert call_kwargs['input_bbox_file'] == bbox_dir / 'bbox.csv'

    def test_unsupported_extension_raises(self, tmp_path, mock_model):
        """Unsupported file extension raises NotImplementedError."""
        args = self._make_args(tmp_path, tmp_path / 'file.avi')
        with (
            patch('lightning_pose.api.Model') as MockModel,
        ):
            MockModel.from_dir.return_value = mock_model
            with pytest.raises(NotImplementedError):
                handle(args)

    def test_directory_input_expands_to_mp4s(self, tmp_path, mock_model):
        """A directory input is expanded to the *.mp4 files it contains."""
        video_dir = tmp_path / 'videos'
        video_dir.mkdir()
        (video_dir / 'a.mp4').touch()
        (video_dir / 'b.mp4').touch()
        (video_dir / 'notes.txt').touch()
        args = argparse.Namespace(
            model_dir=tmp_path / 'model',
            input_path=[video_dir],
            bbox_dir=None,
        )
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.crop_video') as mock_crop,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        assert mock_crop.call_count == 2

    def test_empty_directory_calls_no_crop_video(self, tmp_path, mock_model):
        """An empty directory (no mp4s) results in zero crop_video calls."""
        video_dir = tmp_path / 'empty'
        video_dir.mkdir()
        args = argparse.Namespace(
            model_dir=tmp_path / 'model',
            input_path=[video_dir],
            bbox_dir=None,
        )
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.crop_video') as mock_crop,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_crop.assert_not_called()
