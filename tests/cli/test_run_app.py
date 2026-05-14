"""Test the run_app CLI command argument parsing."""


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
