"""Supervised and unsupervised losses implemented in pytorch."""

from omegaconf import ListConfig
import torch
from torch.nn import functional as F
from torchtyping import TensorType, patch_typeguard
from geomloss import SamplesLoss
from typeguard import typechecked
from typing import Any, Callable, Dict, Tuple, List, Literal, Optional, Union

from pose_est_nets.datasets.preprocessing import format_multiview_data_for_pca
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
    loss_type: str = "mse"
) -> TensorType[()]:
    """Computes heatmap (MSE or Wasserstein) loss between ground truth heatmap and predicted heatmap.

    Args:
        y: ground truth heatmaps
        y_hat: predicted heatmaps

    Returns:
        MSE loss

    """
    # apply mask, only computes loss on heatmaps where the ground truth heatmap
    # is not all zeros (i.e., not an occluded keypoint)
    max_vals = torch.amax(y, dim=(2, 3))
    zeros = torch.zeros(size=(y.shape[0], y.shape[1]), device=y_hat.device)
    non_zeros = ~torch.eq(max_vals, zeros)
    mask = torch.reshape(non_zeros, [non_zeros.shape[0], non_zeros.shape[1], 1, 1])
    # compute loss
    if loss_type == "mse":
        loss = F.mse_loss(torch.masked_select(y_hat, mask), torch.masked_select(y, mask))
    elif loss_type == "wasserstein":
        wass_loss = SamplesLoss(loss="sinkhorn") #maybe could speed up by creating this outside of the function
        loss = wass_loss(torch.masked_select(y_hat, mask).unsqueeze(0), torch.masked_select(y, mask).unsqueeze(0))
    else:
        NotImplementedError("Currently only mse and wasserstein are supported")

    return loss

# @typechecked
# def MaskedWassersteinLoss(
#     y: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
#     y_hat: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
# ) -> TensorType[()]:
#     max_vals = torch.amax(y, dim=(2, 3))
#     zeros = torch.zeros(size=(y.shape[0], y.shape[1]), device=y_hat.device)
#     non_zeros = ~torch.eq(max_vals, zeros)
#     mask = torch.reshape(non_zeros, [non_zeros.shape[0], non_zeros.shape[1], 1, 1])
#     wass_loss = SamplesLoss(loss="sinkhorn")
#     loss = wass_loss(torch.masked_select(y_hat, mask).unsqueeze(0), torch.masked_select(y, mask).unsqueeze(0))
#     return loss



# TODO: this won't work unless the inputs are right, not implemented yet.
# TODO: y_hat should be already reshaped? if so, change below
@typechecked
def MultiviewPCALoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints", float],
    discarded_eigenvectors: TensorType["num_discarded_evecs", "views_times_two", float],
    mean: TensorType[float]
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

    keypoint_preds = keypoint_preds - mean.unsqueeze(-1)
    
    abs_proj_discarded = torch.abs(
        torch.matmul(keypoint_preds.T, discarded_eigenvectors.T)
    )
    epsilon_masked_proj = abs_proj_discarded.masked_fill(
        mask=abs_proj_discarded < epsilon, value=0.0
    )
    # each element positive
    assert (epsilon_masked_proj >= 0.0).all()
    # the scalar loss should be smaller after zeroing out elements.
    assert torch.mean(epsilon_masked_proj) <= torch.mean(abs_proj_discarded)
    return torch.mean(epsilon_masked_proj)


@typechecked
def SingleviewPCALoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints", float],
    discarded_eigenvectors: TensorType[
        "num_discarded_evecs", "two_x_num_keypoints", float
    ],
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
        Projection of data onto discarded eigenvectors

    """

    abs_proj_discarded = torch.abs(
        torch.matmul(keypoint_preds, discarded_eigenvectors.T)
    )
    epsilon_masked_proj = abs_proj_discarded.masked_fill(
        mask=abs_proj_discarded < epsilon, value=0.0
    )
    # each element positive
    assert (epsilon_masked_proj >= 0.0).all()
    # the scalar loss should be smaller after zeroing out elements.
    assert torch.mean(epsilon_masked_proj) <= torch.mean(abs_proj_discarded)
    return torch.mean(epsilon_masked_proj)


@typechecked
def TemporalLoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints"],
    epsilon: TensorType[float] = 5,
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
    # returns tensor of shape (batch - 1, num_targets)
    diffs = torch.diff(keypoint_preds, dim=0)
    # returns tensor of shape (batch - 1, num_keypoints, 2)
    reshape = torch.reshape(diffs, (diffs.shape[0], -1, 2))
    # returns tensor of shape (batch - 1, num_keypoints)
    loss = torch.linalg.norm(reshape, ord=2, dim=2)
    # epsilon-insensitive loss
    loss = loss.masked_fill(mask=loss < epsilon, value=0.0)
    return torch.mean(loss)  # pixels


@typechecked
def UnimodalLoss(
    keypoint_preds: TensorType["batch", "two_x_num_keypoints"],
    heatmap_preds: TensorType[
        "batch", "num_keypoints", "heatmap_height", "heatmap_width"
    ],
    original_image_height: torch.Tensor,
    original_image_width: torch.Tensor,
    output_shape: tuple,
    heatmap_loss_type: str,
    **kwargs  # make loss robust to unneeded inputs
) -> TensorType[(), float]:
    keypoint_preds = keypoint_preds.reshape(keypoint_preds.shape[0], -1, 2)
    ideal_heatmaps = generate_heatmaps(  # this process doesn't compute gradients
        keypoints=keypoint_preds,
        height=int(original_image_height),
        width=int(original_image_width),
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
) -> Tuple[dict, dict]:
    """Set scalars in loss to torch tensors for use with unsupervised losses.

    Args:
        loss_params: dictionary of loss dictionaries, each containing weight, and other args.
        losses_to_use: a list of string with names of losses we'll use for training. these names will match the keys of loss_params
        device: device to send floats and ints to
        to_parameters: boolean saying whether we make the values into torch.nn.Parameter, allowing them to be recognized as by torch.nn.module as module.parameters() (and potentially be trained). if False, keep them as tensors.

    Returns:
        dict with updated values

    """
    loss_params_tensor = {} #for parameters that can be represented as a tensor
    loss_params_dict = {} #for parameters like a tuple or a string which should not be
    for loss, params in loss_params.items():
        print(loss)
        if loss in losses_to_use:
            loss_params_tensor[loss] = {}
            loss_params_dict[loss] = {}
            print("this loss is used")
            for key, val in params.items():
                # TODO: can be made more concise? assign dtype beforehand
                if type(val) == float:
                    loss_params_tensor[loss][key] = torch.tensor(
                        val, dtype=torch.float, device=device
                    )
                if type(val) == int:
                    loss_params_tensor[loss][key] = torch.tensor(
                        val, dtype=torch.int, device=device
                    )
                if val is None:
                    loss_params_tensor[loss][key] = torch.tensor(
                        0.0, dtype=torch.float, device=device
                    )
                elif type(val) != float and type(val) != int: #if it is a string, or a tuple, or some other type of parameter we don't want as a tensor
                    loss_params_dict[loss][key] = val
                    continue
                print("right before making it a parameter")
                if to_parameters:
                    requires_grad = True if key == "log_weight" else False
                    loss_params_tensor[loss][key] = torch.nn.Parameter(
                        data=loss_params_tensor[loss][key], requires_grad=requires_grad
                    )
    print("loss params at the end of convert_dict_entries_to_tensors")
    print(loss_params_tensor, loss_params_dict)
    return loss_params_tensor, loss_params_dict


@typechecked
def convert_loss_tensors_to_torch_nn_modules(loss_params: dict) -> torch.nn.ModuleDict:
    for loss, params in loss_params.items():  # loop over multiple different losses
        # params is a dict which we convert to ParameterDict
        print(loss, params)
        loss_params[loss] = torch.nn.ParameterDict(params)
        for key, val in params.items():  # loop over the entries of each loss
            if (
                key == "log_weight"
            ):  # we take derivatives only w.r.t the weight infront of the loss
                assert loss_params[loss][key].requires_grad == True
            else:  # treat it as a static tensor
                assert loss_params[loss][key].requires_grad == False
    return torch.nn.ModuleDict(loss_params)
