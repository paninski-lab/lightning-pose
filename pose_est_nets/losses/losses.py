import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch.nn import functional as F

patch_typeguard()  # use before @typechecked

@typechecked
def MaskedRegressionMSELoss(
    labels: TensorType["batch", "num_targets"],
    preds: TensorType["batch", "num_targets"],
) -> TensorType[(), float]:
    """
    Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
    :param y: ground truth. shape=(batch, num_targets)
    :param y_hat: prediction. shape=(batch, num_targets)
    :return: mse loss
    """
    mask = labels == labels  # labels is not none, bool.
    loss = F.mse_loss(
        torch.masked_select(labels, mask), torch.masked_select(preds, mask)
    )

    return loss

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

# TODO: this won't work unless the inputs are right, not implemented yet.
@typechecked
# what are we doing about NANS?
def MultiviewPCALoss(
    # TODO: y_hat should be already reshaped? if so, change below
    reshaped_maxima_preds: TensorType["Batch_Size", "Num_Keypoints", 2, float],
    discarded_evecs: TensorType["Views_Times_Two", "Num_Discarded_Evecs", float],
    epsilon: TensorType[float],
) -> TensorType[float]:
    """assume that we have keypoints after find_subpixel_maxima
    and that we have discarded confidence here, and that keypoints were reshaped"""
    # TODO: consider avoiding the transposes
    abs_proj_discarded = torch.abs(
        torch.matmul(reshaped_maxima_preds.T, discarded_evecs.T)
    )
    epsilon_masked_proj = abs_proj_discarded.masked_fill(
        mask=abs_proj_discarded > epsilon, value=0.0
    )
    assert (epsilon_masked_proj >= 0.0).all()  # every element should be positive
    assert torch.mean(epsilon_masked_proj) <= torch.mean(
        abs_proj_discarded
    )  # the scalar loss should be smaller after zeroing out elements.
    return torch.mean(epsilon_masked_proj)

