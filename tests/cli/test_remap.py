"""Test the remap CLI command argument parsing."""

import argparse
from pathlib import Path

import pytest

from lightning_pose.cli.commands.remap import get_parser


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_remap(self):
        """Returned parser has prog set to 'litpose remap'."""
        assert get_parser().prog == 'litpose remap'


class TestRemapParser:
    """Test the remap subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        preds = tmp_path / 'predictions.csv'
        bbox = tmp_path / 'bbox.csv'
        args = parser.parse_args(['remap', str(preds), str(bbox)])
        assert args.preds_file == preds
        assert args.bbox_file == bbox

    def test_missing_preds_file_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(['remap'])

    def test_missing_bbox_file_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(['remap', str(tmp_path / 'predictions.csv')])

    def test_returns_path_objects(self, parser, tmp_path):
        preds = tmp_path / 'predictions.csv'
        bbox = tmp_path / 'bbox.csv'
        args = parser.parse_args(['remap', str(preds), str(bbox)])
        assert isinstance(args.preds_file, Path)
        assert isinstance(args.bbox_file, Path)
