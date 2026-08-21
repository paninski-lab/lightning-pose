"""Test the convert CLI command argument parsing and dispatch."""

import argparse
from unittest.mock import patch

import pytest

from lightning_pose.cli.commands.convert import get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_convert(self):
        """Returned parser has prog set to 'litpose convert'."""
        assert get_parser().prog == 'litpose convert'


class TestConvertParser:
    """Test the convert subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        lp_dir = tmp_path / 'lp_dir'
        args = parser.parse_args(['convert', str(tmp_path), '--lp_dir', str(lp_dir)])
        assert args.dataset_path == tmp_path
        assert args.lp_dir == lp_dir

    def test_missing_lp_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(['convert', str(tmp_path)])

    def test_missing_dataset_path_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(['convert', '--lp_dir', '/tmp/lp_dir'])


class TestHandle:
    """Test the handle function dispatches to the right converter."""

    def _make_args(self, dataset_path, lp_dir):
        return argparse.Namespace(dataset_path=dataset_path, lp_dir=lp_dir)

    def test_directory_dispatches_to_dlc(self, tmp_path):
        dataset_path = tmp_path / 'dlc_proj'
        dataset_path.mkdir()
        lp_dir = tmp_path / 'lp_dir'
        args = self._make_args(dataset_path, lp_dir)

        with patch('lightning_pose.converters.dlc.convert') as mock_convert:
            handle(args)

        mock_convert.assert_called_once_with(dataset_path, lp_dir)

    def test_slp_file_dispatches_to_sleap(self, tmp_path):
        dataset_path = tmp_path / 'project.pkg.slp'
        dataset_path.touch()
        lp_dir = tmp_path / 'lp_dir'
        args = self._make_args(dataset_path, lp_dir)

        with patch('lightning_pose.converters.sleap.convert') as mock_convert:
            handle(args)

        mock_convert.assert_called_once_with(dataset_path, lp_dir)

    def test_unrecognized_file_raises(self, tmp_path):
        dataset_path = tmp_path / 'project.txt'
        dataset_path.touch()
        lp_dir = tmp_path / 'lp_dir'
        args = self._make_args(dataset_path, lp_dir)

        with pytest.raises(ValueError, match='could not determine dataset type'):
            handle(args)
