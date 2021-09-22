from typing import Optional, List
import numpy as np
from typeguard import typechecked
from pose_est_nets.datasets.datasets import HeatmapDataset
import torch


def split_data_deterministic(
    root_directory: str,
    csv_path: str,
    header_rows: list,
    imgaug_transform,
    downsample_factor=2,
    num_train_examples=183,
):

    train_data = HeatmapDataset(
        root_directory,
        csv_path,
        header_rows,
        imgaug_transform,
        downsample_factor=downsample_factor,
    )
    train_data.image_names = train_data.image_names[:num_train_examples]
    train_data.labels = train_data.labels[:num_train_examples]
    train_data.compute_heatmaps()
    val_data = HeatmapDataset(
        root_directory,
        csv_path,
        header_rows,
        imgaug_transform,
        downsample_factor=downsample_factor,
    )
    val_data.image_names = val_data.image_names[183 : 183 + 22]  # hardcoded for now
    val_data.labels = val_data.labels[183 : 183 + 22]  # hardcoded for now
    val_data.compute_heatmaps()
    test_data = HeatmapDataset(
        root_directory,
        csv_path,
        header_rows,
        imgaug_transform,
        downsample_factor=downsample_factor,
    )
    test_data.image_names = test_data.image_names[205:]
    test_data.labels = test_data.labels[205:]
    test_data.compute_heatmaps()

    return train_data, val_data, test_data


@typechecked
def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: float,
    test_probability: Optional[float] = None,
) -> List[int]:
    """a utility that takes in dataset length, and some probabilities for slicing, and spits out the number of examples for training, validation, testing.

    Args:
        total_number (int): total number of examples in dataset
        train_probability (float): fraction of examples used for training
        val_probability (float): fraction of examples used for validation
        test_probability (Optional[float], optional): fraction of examples used for test. Defaults to None. Can be computed as the remaining examples.

    Returns:
        List[int]: num training examples, num validation examples, num test examples
    """

    if test_probability is None:
        test_probability = 1.0 - train_probability - val_probability
    assert (
        test_probability + train_probability + val_probability == 1.0
    )  # probabilities should add to one
    train_number = int(np.floor(train_probability * total_number))
    val_number = int(np.floor(val_probability * total_number))
    test_number = (
        total_number - train_number - val_number
    )  # if we lose extra examples by flooring, send these to test_number
    assert (
        train_number + test_number + val_number == total_number
    )  # assert that we're using all datapoints
    return [train_number, val_number, test_number]


@typechecked
def clean_any_nans(data: torch.tensor, dim: int) -> torch.tensor:
    nan_bool = (
        torch.sum(torch.isnan(data), dim=dim) > 0
    )  # e.g., when dim == 0, those columns (keypoints) that have more than zero nans
    return data[:, ~nan_bool]
