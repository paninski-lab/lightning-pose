"""Test the main CLI entry point."""

import logging
import sys
from importlib.metadata import version

import pytest

from lightning_pose.cli.main import _build_parser, _setup_logging, main


class TestBuildParser:
    """Test the _build_parser function."""

    def test_build_parser_has_all_subcommands(self):
        parser = _build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, '_name_parser_map')
        )
        assert set(subparsers_action._name_parser_map.keys()) == {  # type: ignore[attr-defined]
            'train', 'predict', 'create_bbox', 'smooth_bbox', 'crop', 'remap', 'run_app',
        }

    def test_build_parser_has_debug_flag(self):
        parser = _build_parser()
        assert '--debug' in parser._option_string_actions

    def test_debug_flag_default_false(self):
        parser = _build_parser()
        assert parser.get_default('debug') is False

    def test_main_no_args_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["litpose"]
            main()
        assert exc_info.value.code == 1


class TestVersion:
    """Test the --version flag."""

    def test_version_flag_exits_zero(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--version'])
        assert exc_info.value.code == 0

    def test_version_flag_prints_version(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--version'])
        assert version('lightning-pose') in capsys.readouterr().out


class TestSetupLogging:
    """Test the _setup_logging function."""

    def setup_method(self):
        """Save lightning_pose logger state before each test."""
        pkg_logger = logging.getLogger('lightning_pose')
        self._saved_level = pkg_logger.level
        self._saved_handlers = list(pkg_logger.handlers)
        self._saved_propagate = pkg_logger.propagate

    def teardown_method(self):
        """Restore lightning_pose logger state after each test."""
        pkg_logger = logging.getLogger('lightning_pose')
        pkg_logger.level = self._saved_level
        pkg_logger.handlers = list(self._saved_handlers)
        pkg_logger.propagate = self._saved_propagate

    def test_setup_logging_info_level(self):
        _setup_logging(debug=False)
        assert logging.getLogger('lightning_pose').level == logging.INFO

    def test_setup_logging_debug_level(self):
        _setup_logging(debug=True)
        assert logging.getLogger('lightning_pose').level == logging.DEBUG

    def test_setup_logging_does_not_propagate(self):
        _setup_logging(debug=False)
        assert not logging.getLogger('lightning_pose').propagate

    def test_setup_logging_adds_stream_handler(self):
        pkg_logger = logging.getLogger('lightning_pose')
        n_before = len(pkg_logger.handlers)
        _setup_logging(debug=False)
        assert len(pkg_logger.handlers) == n_before + 1
        assert isinstance(pkg_logger.handlers[-1], logging.StreamHandler)

    def test_setup_logging_isolates_third_party(self):
        """Third-party loggers are unaffected."""
        torch_level_before = logging.getLogger('torch').level
        _setup_logging(debug=True)
        assert logging.getLogger('torch').level == torch_level_before
