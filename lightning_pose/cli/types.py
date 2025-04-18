import argparse
from pathlib import Path


def config_file(filepath):
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


def model_dir(filepath):
    path = Path(filepath)
    return path


def existing_model_dir(filepath):
    path = model_dir(filepath)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Directory model_dir does not exist: {filepath}"
        )

    return path
