import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
import math

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

#batched implementation of above numpy function using torch
#assumes keypoints are detached, or doesn't really matter because we are doing it here
@typechecked
def draw_keypoints_torch(
    keypoints: TensorType["batch", "num_keypoints", 2], 
    height: int, #height of full sized image
    width: int,  #width of full sized image
    output_shape: tuple, #dimensions of full sized image
    sigma: int = 1, #sigma used for generating heatmaps
    normalize: bool = True
):
    keypoints = keypoints.detach().clone()
    batch_dim = keypoints.shape[0]
    n_keypoints = keypoints.shape[1]
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    confidence = torch.zeros((batch_dim, n_keypoints, out_height, out_width))
    xv = torch.arange(out_width)
    yv = torch.arange(out_height)
    xx, yy = torch.meshgrid(xv, yv, indexing='xy')
    xx = xx.unsqueeze(0).unsqueeze(0)
    #shape is now (1, 1, out_width, out_height)
    yy = yy.unsqueeze(0).unsqueeze(0)
    #shape is now (1, 1, out_width, out_height)
    confidence = (yy - keypoints[:, :, 0])
    gaussian = yy keypoints[:, ]
    for idx in range(n_keypoints):
        keypoint = keypoints[idx]
        if torch.any(keypoint != keypoint):  # keeps heatmaps with nans as all zeros
            continue
        gaussian = (yy - keypoint[1]) ** 2
        gaussian += (xx - keypoint[0]) ** 2
        gaussian *= -1
        gaussian /= 2 * sigma ** 2
        gaussian = torch.exp(gaussian)
        confidence[..., idx] = gaussian
    if not normalize:
        confidence /= sigma * torch.sqrt(2 * torch.tensor(math.pi))
    return confidence
