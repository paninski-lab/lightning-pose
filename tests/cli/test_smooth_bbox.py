"""Test the smooth_bbox CLI command argument parsing."""

import argparse
from unittest.mock import patch

import pytest

from lightning_pose.cli.commands.smooth_bbox import get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_smooth_bbox(self):
        """Returned parser has prog set to 'litpose smooth_bbox'."""
        assert get_parser().prog == 'litpose smooth_bbox'


class TestSmoothBboxParser:
    """Test the smooth_bbox subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        bbox_dir = tmp_path / 'bboxes'
        output_dir = tmp_path / 'smooth'
        args = parser.parse_args([
            'smooth_bbox', str(bbox_dir), '--output_dir', str(output_dir),
        ])
        assert args.bbox_dir == bbox_dir
        assert args.output_dir == output_dir
        assert args.method == 'median'
        assert args.window == 5

    def test_output_dir_required(self, parser, tmp_path):
        """Exits when --output_dir is not provided."""
        with pytest.raises(SystemExit):
            parser.parse_args(['smooth_bbox', str(tmp_path / 'bboxes')])

    def test_method_arg(self, parser, tmp_path):
        args = parser.parse_args([
            'smooth_bbox', str(tmp_path / 'bboxes'),
            '--output_dir', str(tmp_path / 'out'),
            '--method', 'median',
        ])
        assert args.method == 'median'

    def test_window_arg(self, parser, tmp_path):
        args = parser.parse_args([
            'smooth_bbox', str(tmp_path / 'bboxes'),
            '--output_dir', str(tmp_path / 'out'),
            '--window', '11',
        ])
        assert args.window == 11


class TestHandle:
    """Test the handle function."""

    def test_delegates_to_smooth_bbox(self, tmp_path):
        """handle calls smooth_bbox with the parsed arguments."""
        args = argparse.Namespace(
            bbox_dir=tmp_path / 'raw',
            output_dir=tmp_path / 'smooth',
            method='median',
            window=7,
        )
        with patch('lightning_pose.utils.cropzoom.smooth_bbox') as mock_smooth:
            handle(args)
        mock_smooth.assert_called_once_with(
            input_bbox_dir=tmp_path / 'raw',
            output_dir=tmp_path / 'smooth',
            method='median',
            window=7,
        )
