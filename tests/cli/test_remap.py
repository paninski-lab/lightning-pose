"""Test the remap CLI command argument parsing."""

from pathlib import Path

import pytest


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
