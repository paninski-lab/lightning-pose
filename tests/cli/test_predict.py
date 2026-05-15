"""Test the predict CLI command argument parsing."""

import argparse

import pytest

from lightning_pose.cli.commands.predict import get_parser


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
