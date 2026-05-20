"""Test the crop CLI command argument parsing."""

import argparse

import pytest

from lightning_pose.cli.commands.crop import get_parser


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
