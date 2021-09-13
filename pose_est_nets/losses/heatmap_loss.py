import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch.nn import functional as F

patch_typeguard()  # use before @typechecked


@typechecked
def MaskedMSEHeatmapLoss(
    y: TensorType["Batch_Size", "Num_Keypoints", "Heatmap_Height", "Heatmap_Width"],
    y_hat: TensorType["Batch_Size", "Num_Keypoints", "Heatmap_Height", "Heatmap_Width"],
) -> TensorType[()]:
    """
    Computes mse loss between ground truth heatmap and predicted heatmap
    :return: mse loss
    """
    # apply mask, only computes loss on heatmaps where the ground truth heatmap is not all zeros (i.e., not an occluded keypoint)
    max_vals = torch.amax(y, dim=(2, 3))
    zeros = torch.zeros(size=(y.shape[0], y.shape[1]), device=y_hat.device)
    non_zeros = ~torch.eq(max_vals, zeros)
    mask = torch.reshape(non_zeros, [non_zeros.shape[0], non_zeros.shape[1], 1, 1])
    # compute loss
    loss = F.mse_loss(torch.masked_select(y_hat, mask), torch.masked_select(y, mask))
    return loss
