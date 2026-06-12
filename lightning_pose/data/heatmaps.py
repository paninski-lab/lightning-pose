"""Heatmap generation and evaluation utilities."""

import numpy as np
import torch
from jaxtyping import Float, Int

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def generate_heatmaps(
    keypoints: Float[torch.Tensor, 'batch num_keypoints 2'],
    height: int,
    width: int,
    output_shape: tuple[int, int],
    sigma: float = 1.25,
    keep_gradients: bool = False,
    visibility: Int[torch.Tensor, 'batch num_keypoints'] | None = None,
) -> Float[torch.Tensor, 'batch num_keypoints height width']:
    """Generate 2D Gaussian heatmaps from mean and sigma.

    Args:
        keypoints: coordinates that serve as mean of gaussian bump
        height: height of reshaped image (pixels, e.g., 128, 256, 512...)
        width: width of reshaped image (pixels, e.g., 128, 256, 512...)
        output_shape: dimensions of downsampled heatmap, (height, width)
        sigma: control spread of gaussian
        keep_gradients: True to not detach gradients from keypoints before creating heatmaps
        visibility: per-keypoint visibility flags with values 0 (not labeled → zero heatmap),
            1 (occluded → uniform heatmap), or 2 (visible → Gaussian heatmap). When None,
            NaN/out-of-bounds keypoints produce zero heatmaps.

    Returns:
        batch of 2D heatmaps

    """
    if keep_gradients:
        keypoints = keypoints.clone()
    else:
        keypoints = keypoints.detach().clone()
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    nan_idxs = (
        torch.isnan(keypoints)[:, :, 0]
        | (keypoints[:, :, 0] < -1)
        | (keypoints[:, :, 0] > out_width + 1)
        | (keypoints[:, :, 1] < -1)
        | (keypoints[:, :, 1] > out_height + 1)
    )

    # clamp keypoints to prevent extreme Gaussian computations
    clamped_x = torch.clamp(keypoints[:, :, 0], -1, out_width + 1)
    clamped_y = torch.clamp(keypoints[:, :, 1], -1, out_height + 1)
    keypoints = torch.stack([clamped_x, clamped_y], dim=2)

    xv = torch.arange(out_width, device=keypoints.device)
    yv = torch.arange(out_height, device=keypoints.device)
    # note flipped order because of pytorch's ij and numpy's xy indexing for meshgrid
    xx, yy = torch.meshgrid(yv, xv, indexing='ij')
    # adds batch and num_keypoints dimensions to grids
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)
    # adds dimension corresponding to the first dimension of the 2d grid
    keypoints = keypoints.unsqueeze(2)
    # evaluates 2d gaussian with mean equal to the keypoint and var equal to sigma^2
    heatmaps = (yy - keypoints[:, :, :, :1]) ** 2  # also flipped order here
    heatmaps += (xx - keypoints[:, :, :, 1:]) ** 2  # also flipped order here
    heatmaps *= -1
    heatmaps /= 2 * sigma**2
    heatmaps = torch.exp(heatmaps)
    # normalize all heatmaps to one
    heatmaps = heatmaps / torch.sum(heatmaps, dim=(2, 3), keepdim=True)
    zero_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)
    uniform_heatmap = torch.ones(
        (out_height, out_width), device=keypoints.device
    ) / (out_height * out_width)

    if visibility is None:
        heatmaps[nan_idxs] = zero_heatmap
    else:
        heatmaps[visibility == 0] = zero_heatmap     # not labeled; ignore in loss
        heatmaps[visibility == 1] = uniform_heatmap  # occluded: encourage low confidence
        heatmaps[(visibility == 2) & nan_idxs] = zero_heatmap

    return heatmaps


def evaluate_heatmaps_at_location(
    heatmaps: Float[torch.Tensor, 'batch num_keypoints heatmap_height heatmap_width'],
    locs: Float[torch.Tensor, 'batch num_keypoints 2'],
    sigma: float = 1.25,
    num_stds: int = 2,
) -> Float[torch.Tensor, 'batch num_keypoints']:
    """Evaluate 4D heatmaps using a 3D location tensor (last dim is x, y coords).

    Since the model outputs heatmaps with a standard deviation of sigma, confidence
    will be spread across neighboring pixels.  To account for this, confidence is
    computed by taking all pixels within two standard deviations of the predicted pixel.

    Args:
        heatmaps: predicted heatmaps
        locs: predicted keypoint locations, last dim is (x, y)
        sigma: sigma used for generating heatmaps
        num_stds: num standard deviations of pixels to compute confidence

    Returns:
        per-keypoint confidence values

    """
    pix_to_consider = int(np.floor(sigma * num_stds))  # get all pixels within num_stds.
    num_pad = pix_to_consider
    heatmaps_padded = torch.zeros(
        (
            heatmaps.shape[0],
            heatmaps.shape[1],
            heatmaps.shape[2] + num_pad * 2,
            heatmaps.shape[3] + num_pad * 2,
        ),
        device=heatmaps.device,
    )
    heatmaps_padded[:, :, num_pad:-num_pad, num_pad:-num_pad] = heatmaps
    i = torch.arange(heatmaps_padded.shape[0], device=heatmaps_padded.device).reshape(
        -1, 1, 1, 1
    )
    j = torch.arange(heatmaps_padded.shape[1], device=heatmaps_padded.device).reshape(
        1, -1, 1, 1
    )
    k = locs[:, :, None, 1, None].type(torch.int64) + num_pad
    m = locs[:, :, 0, None, None].type(torch.int64) + num_pad
    offsets = list(np.arange(-pix_to_consider, pix_to_consider + 1))
    vals_all = []
    for offset in offsets:
        k_offset = k + int(offset)
        for offset_2 in offsets:
            m_offset = m + int(offset_2)
            # get rid of singleton dims
            vals = heatmaps_padded[i, j, k_offset, m_offset].squeeze(-1).squeeze(-1)
            vals_all.append(vals)
    vals = torch.stack(vals_all, 0).sum(0)
    return vals
