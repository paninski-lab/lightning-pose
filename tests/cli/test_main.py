"""Test the main CLI entry point."""

import sys
from importlib.metadata import version

import pytest


class TestBuildParser:
    """Test the _build_parser function."""

    def test_build_parser_has_all_subcommands(self):
        from lightning_pose.cli.main import _build_parser

        parser = _build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, '_name_parser_map')
        )
        assert set(subparsers_action._name_parser_map.keys()) == {  # type: ignore[attr-defined]
            "train", "predict", "crop", "remap", "run_app",
        }

    def test_main_no_args_exits(self):
        from lightning_pose.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["litpose"]
            main()
        assert exc_info.value.code == 1


class TestVersion:
    """Test the --version flag."""

    def test_version_flag_exits_zero(self):
        from lightning_pose.cli.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--version'])
        assert exc_info.value.code == 0

    def test_version_flag_prints_version(self, capsys):
        from lightning_pose.cli.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--version'])
        assert version('lightning-pose') in capsys.readouterr().out
