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
