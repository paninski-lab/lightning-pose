"""Dataset/data module utilities."""

from kornia import image_to_tensor
import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Literal, Optional, Tuple, Union, Dict, Any
import pytorch_lightning as pl
import math
import os, glob

patch_typeguard()  # use before @typechecked


@typechecked
class DataExtractor(object):
    """Helper class to extract all data from a data module."""

    def __init__(
        self,
        data_module: pl.LightningDataModule,
        cond: Literal["train", "test", "val"] = "train",
        extract_images: bool = False,
    ) -> None:
        self.data_module = data_module
        self.cond = cond
        self.extract_images = extract_images

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
            # if we have a dictionary of dataloaders, we take the loader called
            # "labeled" (the loader called "unlabeled" doesn't have keypoints)
            labeled_loader = loader["labeled"]
        else:
            labeled_loader = loader
        return labeled_loader

    @typechecked
    def iterate_over_dataloader(
        self, loader: torch.utils.data.DataLoader
    ) -> Tuple[
        TensorType["num_examples", Any],
        Union[
            TensorType["num_examples", 3, "image_width", "image_height"],
            TensorType["num_examples", "frames", 3, "image_width", "image_height"],
            None,
        ],
    ]:
        keypoints_list = []
        images_list = []
        for ind, batch in enumerate(loader):
            keypoints_list.append(batch["keypoints"])
            if self.extract_images:
                images_list.append(batch["images"])
        concat_keypoints = torch.cat(keypoints_list, dim=0)
        if self.extract_images:
            concat_images = torch.cat(images_list, dim=0)
        else:
            concat_images = None
        # assert that indeed the number of columns does not change after concatenation,
        # and that the number of rows is the dataset length.
        assert concat_keypoints.shape == (
            self.dataset_length,
            keypoints_list[0].shape[1],
        )
        return concat_keypoints, concat_images

    @typechecked
    def __call__(
        self,
    ) -> Tuple[
        TensorType["num_examples", Any],
        Union[
            TensorType["num_examples", 3, "image_width", "image_height"],
            TensorType["num_examples", "frames", 3, "image_width", "image_height"],
            None,
        ],
    ]:
        loader = self.get_loader()
        loader = self.verify_labeled_loader(loader)
        return self.iterate_over_dataloader(loader)


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
    """Remove samples from a data array that contain nans."""
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
    """Simple function to count the number of frames in a video or a list of videos."""

    import cv2

    if isinstance(video_list, str):
        video_list = [video_list]
    num_frames = 0
    for video_file in video_list:
        cap = cv2.VideoCapture(video_file)
        num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return num_frames


@typechecked
def compute_num_train_frames(
    len_train_dataset: int,
    train_frames: Optional[Union[int, float]] = None,
) -> int:
    """Quickly compute number of training frames for a given dataset.

    Args:
        len_train_dataset: total number of frames in training dataset
        train_frames:
            <=1 - fraction of total train frames used for training
            >1 - number of total train frames used for training

    Returns:
    int
        total number of train frames

    """
    if train_frames is None:
        n_train_frames = len_train_dataset
    else:
        if train_frames >= len_train_dataset:
            # take max number of train frames
            print(
                f"Warning! Requested training frames exceeds training set size; "
                f"using all"
            )
            n_train_frames = len_train_dataset
        elif train_frames == 1:
            # assume this is a fraction; use full dataset
            n_train_frames = len_train_dataset
        elif train_frames > 1:
            # take this number of train frames
            n_train_frames = int(train_frames)
        elif train_frames > 0:
            # take this fraction of train frames
            n_train_frames = int(train_frames * len_train_dataset)
        else:
            raise ValueError("train_frames must be >0")

    return n_train_frames


@typechecked
def generate_heatmaps(
    keypoints: TensorType["batch", "num_keypoints", 2],
    height: int,  # height of full sized image
    width: int,  # width of full sized image
    output_shape: Tuple[int, int],  # dimensions of downsampled heatmap
    sigma: Union[float, int] = 1.25,  # sigma used for generating heatmaps
    normalize: bool = True,
    nan_heatmap_mode: str = "zero",
) -> TensorType["batch", "num_keypoints", "height", "width"]:
    """Generate 2D Gaussian heatmaps from mean and sigma.

    Args:
        keypoints: coordinates that serve as mean of gaussian bump
        height: height of original image (pixels)
        width: width of original image (pixels)
        output_shape: dimensions of downsampled heatmap, (height, width)
        sigma: control spread of gaussian
        normalize: normalize to a probability distribution (heatmap sums to one)
        nan_heatmap_mode: flag for how to treat nans: "uniform" | "zero"
            "uniform" returns a uniform probability distribution
            "zero" returns all zeros

    Returns:
        batch of 2D heatmaps

    """
    keypoints = keypoints.detach().clone()
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    nan_idxs = torch.isnan(keypoints)[:, :, 0]
    xv = torch.arange(out_width, device=keypoints.device)
    yv = torch.arange(out_height, device=keypoints.device)
    xx, yy = torch.meshgrid(
        yv, xv
    )  # note flipped order because of pytorch's ij and numpy's xy indexing for meshgrid
    # adds batch and num_keypoints dimensions to grids
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)
    # adds dimension corresponding to the first dimension of the 2d grid
    keypoints = keypoints.unsqueeze(2)
    # evaluates 2d gaussian with mean equal to the keypoint and var equal to sigma^2
    confidence = (yy - keypoints[:, :, :, :1]) ** 2  # also flipped order here
    confidence += (xx - keypoints[:, :, :, 1:]) ** 2  # also flipped order here
    confidence *= -1
    confidence /= 2 * sigma**2
    confidence = torch.exp(confidence)
    if not normalize:
        confidence /= sigma * torch.sqrt(2 * torch.tensor(np.pi))
    else:
        nan_heatmap_mode = "uniform"  # so normalization doesn't fail

    if nan_heatmap_mode == "uniform":
        uniform_heatmap = torch.ones(
            (out_height, out_width), device=keypoints.device
        ) / (out_height * out_width)
        confidence[nan_idxs] = uniform_heatmap
    else:  # nan_heatmap_mode == "zero"
        zero_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)
        confidence[nan_idxs] = zero_heatmap

    if normalize:
        # normalize all heatmaps to one
        confidence = confidence / torch.sum(confidence, dim=(2, 3), keepdim=True)

    return confidence


# @typechecked
# def evaluate_heatmaps_at_location(
#     heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
#     locs: TensorType["batch", "num_keypoints", 2],
# ) -> TensorType["batch", "num_keypoints"]:
#     """Evaluate 4D heatmaps using a 3D location tensor (last dim is x, y coords)."""
#     i = torch.arange(heatmaps.shape[0]).reshape(-1, 1, 1, 1)
#     j = torch.arange(heatmaps.shape[1]).reshape(1, -1, 1, 1)
#     k = locs[:, :, None, 1, None].type(torch.int64)  # y first
#     l = locs[:, :, 0, None, None].type(torch.int64)  # x second
#     vals = heatmaps[i, j, k, l].squeeze(-1).squeeze(-1)  # get rid of singleton dims
#     return vals


@typechecked
def evaluate_heatmaps_at_location(
    heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    locs: TensorType["batch", "num_keypoints", 2],
    sigma: Union[float, int] = 1.25,  # sigma used for generating heatmaps
) -> TensorType["batch", "num_keypoints"]:
    """Evaluate 4D heatmaps using a 3D location tensor (last dim is x, y coords)."""
    i = torch.arange(heatmaps.shape[0]).reshape(-1, 1, 1, 1)
    j = torch.arange(heatmaps.shape[1]).reshape(1, -1, 1, 1)
    k = locs[:, :, None, 1, None].type(torch.int64)
    l = locs[:, :, 0, None, None].type(torch.int64)
    pix_to_consider = int(np.ceil(sigma * 2.0))  # get all pixels within two stds.
    offsets = list(np.arange(-pix_to_consider, pix_to_consider + 1))
    vals_all = []
    for offset in offsets:
        k_offset = k + offset
        k_offset = torch.clamp(k_offset, min=0, max=heatmaps.shape[2] - 1)
        for offset_2 in offsets:
            l_offset = l + offset_2
            l_offset = torch.clamp(l_offset, min=0, max=heatmaps.shape[3] - 1)
            vals = (
                heatmaps[i, j, k_offset, l_offset].squeeze(-1).squeeze(-1)
            )  # get rid of singleton dims
            vals_all.append(vals)
    vals = torch.stack(vals_all, 0).sum(0)
    return vals
