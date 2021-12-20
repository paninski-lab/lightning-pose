import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typing import Union, Tuple
from typeguard import typechecked
import math

# import inspect

patch_typeguard()  # use before @typechecked

# taken from https://github.com/jgraving/DeepPoseKit/blob/master/deepposekit/utils/keypoints.py
def draw_keypoints(keypoints, height, width, output_shape, sigma=1, normalize=True):
    keypoints = keypoints.copy()
    n_keypoints = keypoints.shape[0]
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, 1] *= out_height / height
    keypoints[:, 0] *= out_width / width
    confidence = np.zeros((out_height, out_width, n_keypoints))
    xv = np.arange(out_width)
    yv = np.arange(out_height)
    xx, yy = np.meshgrid(xv, yv)
    for idx in range(n_keypoints):
        keypoint = keypoints[idx]
        if np.any(keypoint != keypoint):  # keeps heatmaps with nans as all zeros
            continue
        gaussian = (yy - keypoint[1]) ** 2
        gaussian += (xx - keypoint[0]) ** 2
        gaussian *= -1
        gaussian /= 2 * sigma ** 2
        gaussian = np.exp(gaussian)
        confidence[..., idx] = gaussian
    if not normalize:
        confidence /= sigma * np.sqrt(2 * np.pi)
    return confidence


# batched implementation of above numpy function using torch
# assumes keypoints are detached, or doesn't really matter because we are doing it here
@typechecked
def generate_heatmaps(
    keypoints: TensorType["batch", "num_keypoints", 2],
    height: int,  # height of full sized image
    width: int,  # width of full sized image
    output_shape: Tuple[int, int],  # dimensions of downsampled heatmap
    sigma: Union[float, int] = 1.25,  # sigma used for generating heatmaps
    normalize: bool = True,
    nan_heatmap_mode: str = "zero"
):
    keypoints = keypoints.detach().clone()
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    nan_idxes = torch.isnan(keypoints)[:, :, 0]
    # confidence = torch.zeros((batch_dim, n_keypoints, out_height, out_width), device=keypoints.device)
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
    # evaluates 2d gaussian with mean equal to the keypoint and variance equal to sigma squared
    confidence = (yy - keypoints[:, :, :, :1]) ** 2  # also flipped order here
    confidence += (xx - keypoints[:, :, :, 1:]) ** 2  # also flipped order here
    confidence *= -1
    confidence /= 2 * sigma ** 2
    confidence = torch.exp(confidence)
    if not normalize:
        confidence /= sigma * torch.sqrt(
            2 * torch.tensor(math.pi), device=keypoints.device
        )
    
    if nan_heatmap_mode == "uniform":
        uniform_heatmap = torch.ones((out_height, out_width), device=keypoints.device) / (out_height * out_width)
        confidence[nan_idxes] = uniform_heatmap
    else: #nan_heatmap_mode == "zero"
        zero_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)
        confidence[nan_idxes] = zero_heatmap

    return confidence
