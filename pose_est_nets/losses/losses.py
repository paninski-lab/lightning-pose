"""Supervised and unsupervised losses implemented in pytorch."""

from omegaconf import ListConfig
import torch
from torch.nn import functional as F
from torchtyping import TensorType, patch_typeguard
from geomloss import SamplesLoss
from typeguard import typechecked
from typing import Any, Callable, Dict, Tuple, List, Literal, Optional, Union

from pose_est_nets.datasets.preprocessing import (
    format_multiview_data_for_pca,
    compute_PCA_reprojection_error,
)
from pose_est_nets.utils.dataset_utils import generate_heatmaps

patch_typeguard()  # use before @typechecked


@typechecked
def MaskedRegressionMSELoss(
    keypoints: TensorType["batch", "two_x_num_keypoints"],
    preds: TensorType["batch", "two_x_num_keypoints"],
) -> TensorType[(), float]:
    """Compute MSE loss between ground truth and predicted coordinates.

    Args:
        keypoints: ground truth; shape=(batch, num_targets)
        preds: predictions; shape=(batch, num_targets)

    Returns:
        MSE loss averaged over both dimensions

    """
    mask = keypoints == keypoints  # keypoints is not none, bool.
    loss = F.mse_loss(
        torch.masked_select(keypoints, mask), torch.masked_select(preds, mask)
    )
    return loss


@typechecked
def MaskedRMSELoss(
    keypoints: TensorType["batch", "two_x_num_keypoints"],
    preds: TensorType["batch", "two_x_num_keypoints"],
) -> TensorType[(), float]:
    """Compute RMSE loss between ground truth and predicted coordinates.

    Args:
        keypoints: ground truth; shape=(batch, num_targets)
        preds: predictions; shape=(batch, num_targets)

    Returns:
        Root mean-square error per keypoint averaged

    """

    mask = keypoints == keypoints  # keypoints is not none, bool.
    loss = F.mse_loss(
        torch.masked_select(keypoints, mask),
        torch.masked_select(preds, mask),
        reduction="none",
    )

    return torch.mean(torch.sqrt(loss))


@typechecked
def MaskedHeatmapLoss(
    y: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    y_hat: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    loss_type: Literal["mse", "wasserstein"] = "mse",
    reach: Union[float, str] = None,
) -> TensorType[()]:
    """Computes heatmap (MSE or Wasserstein) loss between ground truth heatmap and predicted heatmap.

    Args:
        y: ground truth heatmaps
        y_hat: predicted heatmaps

    Returns:
        MSE or Wasserstein loss

    """
    # apply mask, only computes loss on heatmaps where the ground truth heatmap
    # is not all zeros (i.e., not an occluded keypoint)
    bs = y.shape[0]
    nk = y.shape[1]
    max_vals = torch.amax(y, dim=(2, 3))
    zeros = torch.zeros(size=(y.shape[0], y.shape[1]), device=y_hat.device)
    non_zeros = ~torch.eq(max_vals, zeros)
    mask = torch.reshape(non_zeros, [non_zeros.shape[0], non_zeros.shape[1], 1, 1])
    # compute loss
    if loss_type == "mse":
        loss = F.mse_loss(
            torch.masked_select(y_hat, mask), torch.masked_select(y, mask)
        )
    elif loss_type == "wasserstein":
        if reach == "None":
            reach = None
        wass_loss = SamplesLoss(
            loss="sinkhorn", reach=reach
        )  # maybe could speed up by creating this outside of the function
        loss = wass_loss(
            torch.masked_select(y_hat, mask).unsqueeze(0),
            torch.masked_select(y, mask).unsqueeze(0),
        )
        loss = loss / (
            bs * nk
        )  # we should divide this by batch size times number of keypoints so its on per heatmap scale

    return loss


# TODO: this won't work unless the inputs are right, not implemented yet.
# TODO: y_hat should be already reshaped? if so, change below
@typechecked
def MultiviewPCALoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints", float],
    kept_eigenvectors: TensorType["num_kept_evecs", "views_times_two", float],
    mean: TensorType[float],
    epsilon: TensorType[float],
    mirrored_column_matches: Union[ListConfig, List],
    **kwargs  # make loss robust to unneeded inputs
) -> TensorType[float]:
    """

    Assume that we have keypoints after find_subpixel_maxima and that we have
    discarded confidence here, and that keypoints were reshaped
    # TODO: check for this?

    Args:
        keypoint_preds:
        discarded_eigenvectors:
        epsilon:
        mirrored_column_matches:
        **kwargs:

    Returns:
        Projection of data onto discarded eigenvectors

    """
    # TODO: consider avoiding the transposes
    keypoint_preds = keypoint_preds.reshape(
        keypoint_preds.shape[0], -1, 2
    )  # shape = (batch_size, num_keypoints, 2)

    keypoint_preds = format_multiview_data_for_pca(
        data_arr=keypoint_preds, mirrored_column_matches=mirrored_column_matches
    )  # shape = (views * 2, num_batches * num_keypoints)

    reprojection_error = compute_PCA_reprojection_error(
        good_arr_for_pca=keypoint_preds, kept_eigenvectors=kept_eigenvectors, mean=mean
    )  # shape = (num_batches * num_keypoints, num_views)

    # loss values below epsilon as masked to zero
    reprojection_loss = reprojection_error.masked_fill(
        mask=reprojection_error < epsilon, value=0.0
    )
    # average across both (num_batches * num_keypoints) and num_views
    return torch.mean(reprojection_loss)


# TODO: write a unit-test for this without the toy_dataset
@typechecked
def SingleviewPCALoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints", float],
    kept_eigenvectors: TensorType["num_kept_evecs", "two_x_num_keypoints", float],
    mean: TensorType[float],
    epsilon: TensorType[float],
    **kwargs  # make loss robust to unneeded inputs
) -> TensorType[float]:
    """

    Assume that we have keypoints after find_subpixel_maxima and that we have
    discarded confidence here, and that keypoints were reshaped
    # TODO: check for this?

    Args:
        keypoint_preds:
        discarded_eigenvectors:
        epsilon:
        **kwargs:

    Returns:
        Average reprojection error across batch and num_keypoints

    """

    reprojection_error = compute_PCA_reprojection_error(
        good_arr_for_pca=keypoint_preds.T,
        kept_eigenvectors=kept_eigenvectors,
        mean=mean,
    )  # shape = (batch, num_keypoints)

    # loss values below epsilon as masked to zero
    reprojection_loss = reprojection_error.masked_fill(
        mask=reprojection_error < epsilon, value=0.0
    )
    # average across both batch and num_keypoints
    return torch.mean(reprojection_loss)


@typechecked
def TemporalLoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints"],
    epsilon: TensorType[float] = 5.0,
    **kwargs  # make loss robust to unneeded inputs
) -> TensorType[(), float]:
    """Penalize temporal differences for each target.

    Motion model: x_t = x_(t-1) + e_t, e_t ~ N(0, s)

    Args:
        preds: keypoint predictions; shape=(batch, num_targets)
        epsilon: loss values below this threshold are discarded (set to zero)

    Returns:
        Temporal loss averaged over batch

    """
    diffs = torch.diff(keypoint_preds, dim=0)  # shape (batch - 1, num_targets)
    reshape = torch.reshape(
        diffs, (diffs.shape[0], -1, 2)
    )  # shape (batch - 1, num_keypoints, 2)
    loss = torch.linalg.norm(reshape, ord=2, dim=2)  # shape (batch - 1, num_keypoints)
    # epsilon-insensitive loss
    loss = loss.masked_fill(mask=loss < epsilon, value=0.0)
    return torch.mean(loss)  # pixels


@typechecked
def UnimodalLoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints"],
    heatmap_preds: TensorType[
        "batch", "num_keypoints", "heatmap_height", "heatmap_width"
    ],
    original_image_height: int,
    original_image_width: int,
    output_shape: tuple,
    heatmap_loss_type: str,
    **kwargs  # make loss robust to unneeded inputs
) -> TensorType[(), float]:
    keypoint_preds = keypoint_preds.reshape(keypoint_preds.shape[0], -1, 2)
    ideal_heatmaps = generate_heatmaps(  # this process doesn't compute gradients
        keypoints=keypoint_preds,
        height=original_image_height,
        width=original_image_width,
        output_shape=output_shape,
    )
    return MaskedHeatmapLoss(ideal_heatmaps, heatmap_preds, heatmap_loss_type)


@typechecked
def filter_dict(mydict: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Filter dictionary by desired keys.

    Args:
        mydict: disctionary with strings as keys.
        keys: a list of key names to keep.

    Returns:
        the same dictionary only at the desired keys.

    """
    return {k: v for k, v in mydict.items() if k in keys}


@typechecked
def get_losses_dict(
    names_list: list = [],
) -> Dict[str, Callable]:
    """Get a dict with all the loss functions for semi supervised training.

    The training step of a given model will iterate over these, instead of
    manually computing each.

    Args:
        names_list: list of desired loss names; defaults to empty.

    Returns:
        Dict[str, Callable]: [description]

    """
    loss_dict = {
        "regression": MaskedRegressionMSELoss,
        "heatmap": MaskedHeatmapLoss,
        "pca_multiview": MultiviewPCALoss,
        "pca_singleview": SingleviewPCALoss,
        "temporal": TemporalLoss,
        "unimodal": UnimodalLoss,
    }
    return filter_dict(loss_dict, names_list)


@typechecked
def convert_dict_entries_to_tensors(
    loss_params: Dict[str, dict],
    device: Union[str, torch.device],
    losses_to_use: List[str],
    to_parameters: bool = False,
) -> Tuple:
    """Set scalars in loss to torch tensors for use with unsupervised losses.

    Args:
        loss_params: dictionary of loss dictionaries, each containing weight, and other args.
        losses_to_use: a list of string with names of losses we'll use for training. these names will match the keys of loss_params
        device: device to send floats and ints to
        to_parameters: boolean saying whether we make the values into torch.nn.Parameter, allowing them to be recognized as by torch.nn.module as module.parameters() (and potentially be trained). if False, keep them as tensors.

    Returns:
        dict with updated values

    """
    loss_weights_dict = {}  # for parameters that can be represented as a tensor
    loss_params_dict = {}  # for parameters like a tuple or a string which should not be
    for loss, params in loss_params.items():
        if loss in losses_to_use:
            loss_params_dict[loss] = {}
            for key, val in params.items():
                if key == "log_weight":
                    loss_weights_dict[loss] = torch.tensor(
                        val, dtype=torch.float, device=device
                    )
                elif key == "epsilon" and type(val) != torch.Tensor and val != None:
                    loss_params_dict[loss][key] = torch.tensor(
                        val, dtype=torch.float, device=device
                    )
                else:
                    loss_params_dict[loss][key] = val
    if to_parameters:
        loss_weights_dict = convert_loss_tensors_to_torch_nn_params(loss_weights_dict)
    print("loss weights at the end of convert_dict_entries_to_tensors:")
    print(loss_weights_dict)
    for key, val in loss_weights_dict.items():
        print(key, val)
    print("loss params dict:")
    print(loss_params_dict)

    return loss_weights_dict, loss_params_dict


@typechecked
def convert_loss_tensors_to_torch_nn_params(
    loss_weights: dict,
) -> torch.nn.ParameterDict:
    loss_weights_params = {}
    for loss, weight in loss_weights.items():  # loop over multiple different losses
        print(loss, weight)
        loss_weights_params[loss] = torch.nn.Parameter(weight, requires_grad=True)
    parameter_dict = torch.nn.ParameterDict(loss_weights_params)
    return parameter_dict
