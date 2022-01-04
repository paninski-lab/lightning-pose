"""Dataset/data module utilities."""

import numpy as np
import torch
from torch.utils import data
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Literal, Optional, Tuple, Union, Dict, Any
import pytorch_lightning as pl

# from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule


patch_typeguard()  # use before @typechecked


@typechecked
class DataExtractor:
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        cond: Literal["train", "test", "val"] = "train",
    ) -> None:
        self.data_module = data_module
        self.cond = cond

    @property
    def dataset_length(self) -> int:
        name = "%s_dataset" % self.cond
        return len(getattr(self.data_module, name))

    @typechecked
    def get_loader(self) -> Union[torch.utils.data.DataLoader, dict]:
        if self.cond == "train":
            return self.data_module.train_dataloader()
        if self.cond == "val":
            return self.data_module.val_dataloader()
        if self.cond == "test":
            return self.data_module.test_dataloader()

    @typechecked
    def verify_labeled_loader(
        self, loader: Union[torch.utils.data.DataLoader, dict]
    ) -> torch.utils.data.DataLoader:
        if type(loader) is dict and "labeled" in list(loader.keys()):
            # if we have a dictionary of dataloaders, we take the loader called "labeled" (the loader called "unlabeled" doesn't have keypoints)
            labeled_loader = loader["labeled"]
        else:
            labeled_loader = loader
        return labeled_loader

    @typechecked
    def iterate_over_dataloader(
        self, loader: torch.utils.data.DataLoader
    ) -> TensorType["num_training_examples", Any]:
        keypoints_list = []
        for ind, batch in enumerate(loader):
            keypoints_list.append(batch["keypoints"])
        concat_keypoints = torch.cat(keypoints_list, dim=0)
        # assert that indeed the number of columns does not change after concatenation, and that the number of rows is the dataset length.
        assert concat_keypoints.shape == (
            self.dataset_length,
            keypoints_list[0].shape[1],
        )

        return concat_keypoints

    @typechecked
    def __call__(self) -> TensorType["num_training_examples", Any]:
        loader = self.verify_labeled_loader(self.get_loader())
        data_tensor = self.iterate_over_dataloader(loader)
        return data_tensor


@typechecked
def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: Optional[float] = None,
    test_probability: Optional[float] = None,
) -> List[int]:
    """Returns the number of examples for train, val and test given split probs.

    Args:
        total_number: total number of examples in dataset
        train_probability: fraction of examples used for training
        val_probability: fraction of examples used for validation
        test_probability: fraction of examples used for test. Defaults to None.
            Can be computed as the remaining examples.

    Returns:
        [num training examples, num validation examples, num test examples]

    """

    if test_probability is None and val_probability is None:
        remaining_probability = 1.0 - train_probability
        # round each to 5 decimal places (issue with floating point precision)
        val_probability = round(remaining_probability / 2, 5)
        test_probability = round(remaining_probability / 2, 5)
    elif test_probability is None:
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
def clean_any_nans(data: torch.Tensor, dim: int) -> torch.Tensor:
    # currently supports only 2D arrays
    nan_bool = (
        torch.sum(torch.isnan(data), dim=dim) > 0
    )  # e.g., when dim == 0, those columns (keypoints) that have >0 nans
    if dim == 0:
        return data[:, ~nan_bool]
    elif dim == 1:
        return data[~nan_bool]


@typechecked
def count_frames(video_list: Union[List[str], str]) -> int:
    import cv2

    if isinstance(video_list, str):
        video_list = [video_list]

    num_frames = 0
    for video_file in video_list:
        cap = cv2.VideoCapture(video_file)
        num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return num_frames
