"""Test the run_app CLI command argument parsing."""

import argparse

from lightning_pose.cli.commands.run_app import get_parser


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_run_app(self):
        """Returned parser has prog set to 'litpose run_app'."""
        assert get_parser().prog == 'litpose run_app'


class TestRunAppParser:
    """Test the run_app subcommand argument parsing."""

    def test_defaults(self, parser):
        args = parser.parse_args(['run_app'])
        assert args.port == 8080
        assert args.host == '127.0.0.1'

    def test_custom_port(self, parser):
        args = parser.parse_args(['run_app', '--port', '9090'])
        assert args.port == 9090

    def test_custom_host(self, parser):
        args = parser.parse_args(['run_app', '--host', '0.0.0.0'])
        assert args.host == '0.0.0.0'

    def test_custom_port_and_host(self, parser):
        args = parser.parse_args(['run_app', '--port', '5000', '--host', '0.0.0.0'])
        assert args.port == 5000
        assert args.host == '0.0.0.0'
