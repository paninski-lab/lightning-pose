"""Custom argparse type validators for CLI path arguments."""

import argparse
from pathlib import Path


def config_file(filepath: str) -> Path:
    """
    Custom argparse type for validating that a file exists and is a yaml file.

    Args:
    filepath: The file path string.

    Returns:
    A pathlib.Path object if the file is valid, otherwise raises an error.
    """
    path = Path(filepath)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    if not path.suffix == ".yaml":
        raise argparse.ArgumentTypeError(f"File must be a yaml file: {filepath}")
    return path


def model_dir(filepath: str | Path) -> Path:
    """Convert a filepath string or Path to a ``pathlib.Path`` for a model directory.

    Args:
        filepath: path string or Path object pointing to a model directory.

    Returns:
        ``pathlib.Path`` of the given filepath.
    """
    path = Path(filepath)
    return path


def existing_model_dir(filepath: str | Path) -> Path:
    """Validate and return the path to an existing model directory.

    Args:
        filepath: path string or Path object pointing to an existing model directory.

    Returns:
        ``pathlib.Path`` of the directory.

    Raises:
        argparse.ArgumentTypeError: if the path does not point to an existing directory.
    """
    path = model_dir(filepath)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Directory model_dir does not exist: {filepath}"
        )

    return path
