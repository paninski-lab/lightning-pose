"""Fixtures for CLI tests."""

import pytest


@pytest.fixture
def parser():
    """Full CLI argument parser."""
    from lightning_pose.cli.main import _build_parser

    return _build_parser()
