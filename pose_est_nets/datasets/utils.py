"""Dataset/data module utilities."""

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Optional, Tuple, Union


patch_typeguard()  # use before @typechecked


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
def clean_any_nans(data: torch.tensor, dim: int) -> torch.tensor:
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


@typechecked
def generate_heatmaps(
    keypoints: TensorType["batch", "num_keypoints", 2],
    height: int,  # height of full sized image
    width: int,  # width of full sized image
    output_shape: Tuple[int, int],  # dimensions of downsampled heatmap
    sigma: Union[float, int] = 1.25,  # sigma used for generating heatmaps
    normalize: bool = True,
    nan_heatmap_mode: str = "zero"
) -> TensorType["batch", "num_keypoints", "height", "width"]:
    """Generate 2D Gaussian heatmaps from mean and sigma.

    Args:
        keypoints:
        height:
        width:
        output_shape:
        sigma:
        normalize:
        nan_heatmap_mode:

    Returns:
        batch of 2D heatmaps

    """
    keypoints = keypoints.detach().clone()
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    nan_idxes = torch.isnan(keypoints)[:, :, 0]
    # confidence = torch.zeros(
    #     (batch_dim, n_keypoints, out_height, out_width),
    #     device=keypoints.device
    # )
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
    confidence /= 2 * sigma ** 2
    confidence = torch.exp(confidence)
    if not normalize:
        confidence /= sigma * torch.sqrt(
            2 * torch.tensor(np.pi), device=keypoints.device
        )

    if nan_heatmap_mode == "uniform":
        uniform_heatmap = torch.ones(
            (out_height, out_width),
            device=keypoints.device
        ) / (out_height * out_width)
        confidence[nan_idxes] = uniform_heatmap
    else:  # nan_heatmap_mode == "zero"
        zero_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)
        confidence[nan_idxes] = zero_heatmap

    return confidence
