"""Test the create_bbox CLI command argument parsing."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lightning_pose.cli.commands.create_bbox import get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_create_bbox(self):
        """Returned parser has prog set to 'litpose create_bbox'."""
        assert get_parser().prog == 'litpose create_bbox'


class TestCreateBboxParser:
    """Test the create_bbox subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video = tmp_path / 'video.mp4'
        args = parser.parse_args(['create_bbox', str(model_dir), str(video)])
        assert args.model_dir == model_dir
        assert args.input_path == [video]
        assert args.crop_ratio is None
        assert args.crop_size is None
        assert args.anchor_keypoints == ''

    def test_missing_model_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['create_bbox', str(tmp_path / 'missing'), str(tmp_path / 'video.mp4')]
            )

    def test_crop_ratio(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['create_bbox', str(model_dir), 'video.mp4', '--crop_ratio', '3.5']
        )
        assert args.crop_ratio == 3.5

    def test_crop_size(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['create_bbox', str(model_dir), 'video.mp4', '--crop_size', '100']
        )
        assert args.crop_size == 100
        assert args.crop_ratio is None

    def test_anchor_keypoints(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['create_bbox', str(model_dir), 'video.mp4', '--anchor_keypoints', 'nose,tail']
        )
        assert args.anchor_keypoints == 'nose,tail'

    def test_multiple_input_paths(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video1 = tmp_path / 'video1.mp4'
        video2 = tmp_path / 'video2.mp4'
        args = parser.parse_args(
            ['create_bbox', str(model_dir), str(video1), str(video2)]
        )
        assert args.input_path == [video1, video2]


class TestHandle:
    """Test the handle function."""

    @pytest.fixture
    def mock_model(self, tmp_path):
        """Mock Model instance returned by Model.from_dir."""
        model = MagicMock()
        model.config.cfg.data.data_dir = str(tmp_path / 'data')
        model.video_preds_dir.return_value = tmp_path / 'video_preds'
        model.image_preds_dir.return_value = tmp_path / 'image_preds'
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
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
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
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
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
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        cfg = mock_gen.call_args.kwargs['detector_cfg']
        assert cfg['crop_ratio'] == 3.5

    def test_csv_input_calls_generate_bbox(self, tmp_path, mock_model):
        """CSV input triggers generate_bbox with the correct detector_cfg."""
        args = self._make_args(tmp_path, tmp_path / 'labels.csv', crop_size=200)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        cfg = mock_gen.call_args.kwargs['detector_cfg']
        assert cfg['crop_height'] == 200
        assert cfg['crop_width'] == 200

    def test_mp4_bbox_output_path(self, tmp_path, mock_model):
        """MP4 input writes bbox to <video_preds_dir>/<stem>_bbox.csv."""
        args = self._make_args(tmp_path, tmp_path / 'myvid.mp4')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        output_path = mock_gen.call_args.kwargs['output_bbox_file']
        assert output_path.name == 'myvid_bbox.csv'

    def test_csv_bbox_output_path(self, tmp_path, mock_model):
        """CSV input writes bbox to <image_preds_dir>/<csv_name>/bbox.csv."""
        args = self._make_args(tmp_path, tmp_path / 'labels.csv')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        output_path = mock_gen.call_args.kwargs['output_bbox_file']
        assert output_path.name == 'bbox.csv'

    def test_directory_input_expands_to_mp4s(self, tmp_path, mock_model):
        """A directory input is expanded to the *.mp4 files it contains."""
        video_dir = tmp_path / 'videos'
        video_dir.mkdir()
        (video_dir / 'a.mp4').touch()
        (video_dir / 'b.mp4').touch()
        (video_dir / 'notes.txt').touch()
        args = self._make_args(tmp_path, video_dir)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        assert mock_gen.call_count == 2
        called_stems = {c.kwargs['output_bbox_file'].stem for c in mock_gen.call_args_list}
        assert called_stems == {'a_bbox', 'b_bbox'}

    def test_empty_directory_calls_no_generate_bbox(self, tmp_path, mock_model):
        """An empty directory (no mp4s) results in zero generate_bbox calls."""
        video_dir = tmp_path / 'empty'
        video_dir.mkdir()
        args = self._make_args(tmp_path, video_dir)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.utils.cropzoom.generate_bbox') as mock_gen,
        ):
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_gen.assert_not_called()
