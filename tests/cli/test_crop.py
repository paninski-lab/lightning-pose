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
        assert args.crop_ratio is None
        assert args.crop_size is None
        assert args.anchor_keypoints == ''

    def test_missing_model_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['crop', str(tmp_path / 'missing'), str(tmp_path / 'video.mp4')]
            )

    def test_crop_ratio(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['crop', str(model_dir), 'video.mp4', '--crop_ratio', '3.5'])
        assert args.crop_ratio == 3.5

    def test_anchor_keypoints(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['crop', str(model_dir), 'video.mp4', '--anchor_keypoints', 'nose,tail']
        )
        assert args.anchor_keypoints == 'nose,tail'

    def test_crop_size(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['crop', str(model_dir), 'video.mp4', '--crop_size', '100'])
        assert args.crop_size == 100
        assert args.crop_ratio is None

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

    def _make_args(
        self,
        tmp_path,
        input_path,
        crop_ratio=None,
        crop_size=None,
        anchor_keypoints='',
    ):
        return argparse.Namespace(
            model_dir=tmp_path / 'model',
            input_path=[input_path],
            crop_ratio=crop_ratio,
            crop_size=crop_size,
            anchor_keypoints=anchor_keypoints,
        )

    def test_raises_when_both_crop_ratio_and_crop_size(self, tmp_path, mock_model):
        """Raises ValueError when --crop_ratio and --crop_size are both provided."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', crop_ratio=2.0, crop_size=100)
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            with pytest.raises(ValueError, match='mutually exclusive'):
                handle(args)

    def test_raises_when_crop_size_not_positive(self, tmp_path, mock_model):
        """Raises ValueError when --crop_size is not a positive integer."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', crop_size=0)
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            with pytest.raises(ValueError, match='positive integer'):
                handle(args)

    def test_raises_when_crop_ratio_not_greater_than_one(self, tmp_path, mock_model):
        """Raises ValueError when --crop_ratio is <= 1."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', crop_ratio=1.0)
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            with pytest.raises(ValueError, match='greater than 1'):
                handle(args)

    def test_default_crop_ratio_when_neither_given(self, tmp_path, mock_model):
        """Defaults to crop_ratio=2.0 when neither flag is supplied."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_cropped_video') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        cfg = mock_gen.call_args.kwargs['detector_cfg']
        assert cfg['crop_ratio'] == 2.0
        assert 'crop_height' not in cfg
        assert 'crop_width' not in cfg

    def test_crop_size_sets_detector_cfg(self, tmp_path, mock_model):
        """crop_height and crop_width are set from --crop_size; crop_ratio is absent."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', crop_size=100)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_cropped_video') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        cfg = mock_gen.call_args.kwargs['detector_cfg']
        assert cfg['crop_height'] == 100
        assert cfg['crop_width'] == 100
        assert 'crop_ratio' not in cfg

    def test_crop_ratio_sets_detector_cfg(self, tmp_path, mock_model):
        """crop_ratio is passed through to detector_cfg."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', crop_ratio=3.5)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_cropped_video') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        cfg = mock_gen.call_args.kwargs['detector_cfg']
        assert cfg['crop_ratio'] == 3.5
        assert 'crop_height' not in cfg
        assert 'crop_width' not in cfg

    def test_csv_input_calls_generate_cropped_labeled_frames(self, tmp_path, mock_model):
        """CSV input triggers generate_cropped_labeled_frames with correct detector_cfg."""
        args = self._make_args(tmp_path, tmp_path / 'labels.csv', crop_size=200)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_cropped_labeled_frames') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        cfg = mock_gen.call_args.kwargs['detector_cfg']
        assert cfg['crop_height'] == 200
        assert cfg['crop_width'] == 200
