import torch
import math
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Optional, Tuple, Union


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
            2 * torch.tensor(math.pi), device=keypoints.device
        )

    if nan_heatmap_mode == "uniform":
        uniform_heatmap = torch.ones(
            (out_height, out_width), device=keypoints.device
        ) / (out_height * out_width)
        confidence[nan_idxes] = uniform_heatmap
    else:  # nan_heatmap_mode == "zero"
        zero_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)
        confidence[nan_idxes] = zero_heatmap

    return confidence
