"""Test CLI argument types."""

import argparse

import pytest


class TestConfigFile:
    """Test the config_file type validator."""

    def test_config_file_valid(self, tmp_path):
        from lightning_pose.cli.types import config_file

        f = tmp_path / "config.yaml"
        f.write_text("")
        result = config_file(str(f))
        assert result == f

    def test_config_file_not_found(self, tmp_path):
        from lightning_pose.cli.types import config_file

        with pytest.raises(argparse.ArgumentTypeError, match="File not found"):
            config_file(str(tmp_path / "missing.yaml"))

    def test_config_file_wrong_extension(self, tmp_path):
        from lightning_pose.cli.types import config_file

        f = tmp_path / "config.txt"
        f.write_text("")
        with pytest.raises(argparse.ArgumentTypeError, match="yaml"):
            config_file(str(f))


class TestModelDir:
    """Test the model_dir type validator."""

    def test_model_dir_returns_path(self, tmp_path):
        from pathlib import Path

        from lightning_pose.cli.types import model_dir

        result = model_dir(str(tmp_path))
        assert isinstance(result, Path)
        assert result == tmp_path

    def test_model_dir_nonexistent_allowed(self, tmp_path):
        from lightning_pose.cli.types import model_dir

        result = model_dir(str(tmp_path / "new_dir"))
        assert result == tmp_path / "new_dir"


class TestExistingModelDir:
    """Test the existing_model_dir type validator."""

    def test_existing_model_dir_valid(self, tmp_path):
        from lightning_pose.cli.types import existing_model_dir

        result = existing_model_dir(str(tmp_path))
        assert result == tmp_path

    def test_existing_model_dir_missing(self, tmp_path):
        from lightning_pose.cli.types import existing_model_dir

        with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
            existing_model_dir(str(tmp_path / "missing"))
