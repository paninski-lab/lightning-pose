"""Supervised and unsupervised losses implemented in pytorch."""

from omegaconf import ListConfig
import torch
from torch.nn import functional as F
from torchtyping import TensorType, patch_typeguard
from geomloss import SamplesLoss
from typeguard import typechecked
from typing import Any, Callable, Dict, Tuple, List, Literal, Optional, Union
import pytorch_lightning as pl
from pose_est_nets.utils.pca import (
    KeypointPCA,
    format_multiview_data_for_pca,
    compute_PCA_reprojection_error,
)
from pose_est_nets.utils.heatmaps import generate_heatmaps
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule

patch_typeguard()  # use before @typechecked

reduce_methods_dict = {"mean": torch.mean, "sum": torch.sum}

# TODO: Do we want to inherit from torch or ptl?
class Loss(pl.LightningModule):
    def __init__(self, epsilon: float = 0.0, log_weight: float = 0.0) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.log_weight = torch.tensor(log_weight, device=self.device)
        self.loss_name = "base"

    @property
    def weight(self) -> TensorType[(), float]:
        # weight = \sigma where our trainable parameter is \log(\sigma^2). i.e., we take the parameter as it is in the config and exponentiate it to enforce positivity
        weight = 1.0 / (2.0 * torch.exp(self.log_weight))
        return weight

    def rectify_epsilon(
        self, loss: torch.Tensor
    ) -> torch.Tensor:  # TODO: check if we can assert same in/out shapes here
        # loss values below epsilon as masked to zero
        loss = loss.masked_fill(mask=loss < self.epsilon, value=0.0)
        return loss

    def remove_nans(self, **kwargs):
        # find nans in the targets, and do a masked_select operation
        raise NotImplementedError

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    def reduce_loss(
        self, loss: torch.Tensor, method: str = "mean"
    ) -> TensorType[(), float]:
        return reduce_methods_dict[method](loss)

    def log_loss(self, loss: torch.Tensor) -> None:
        self.log(self.loss_name + "_loss", loss, prog_bar=True)
        self.log(self.loss_name + "_weight", self.weight)

    def __call__(self, **kwargs):
        raise NotImplementedError
        # # give us the flow of operations, and we overwrite the methods, and determine their arguments which are in buffer
        # self.remove_nans()
        # self.compute_loss()
        # self.rectify_epsilon()
        # self.reduce_loss()
        # self.log_loss()
        # return self.weight * scalar_loss


class HeatmapLoss(Loss):
    # TODO: check if we can safely eliminate the __init__()
    def __init__(
        self,
        epsilon: float = 0.0,
        log_weight: float = 0.0,
    ) -> None:
        super().__init__(epsilon=epsilon, log_weight=log_weight)

    def remove_nans(
        self,
        targets: TensorType[
            "batch", "num_keypoints", "heatmap_height", "heatmap_width"
        ],
        predictions: TensorType[
            "batch", "num_keypoints", "heatmap_height", "heatmap_width"
        ],
    ) -> Tuple[
        TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ]:

        squeezed_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
        all_zeroes = torch.all(squeezed_targets == 0.0, dim=-1)

        return targets[~all_zeroes], predictions[~all_zeroes]

    def __call__(
        self,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        logging: bool = True,
        **kwargs
    ):
        # give us the flow of operations, and we overwrite the methods, and determine their arguments which are in buffer
        clean_targets, clean_predictions = self.remove_nans(
            targets=targets, predictions=predictions
        )
        elementwise_loss = self.compute_loss(
            targets=clean_targets, predictions=clean_predictions
        )
        epsilon_insensitive_loss = self.rectify_epsilon(loss=elementwise_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")
        if logging:
            self.log_loss(loss=scalar_loss)
        return self.weight * scalar_loss


class HeatmapMSELoss(HeatmapLoss):
    def __init__(
        self, epsilon: float = 0.0, log_weight: float = 0.0, loss_name="heatmap_mse"
    ) -> None:
        super().__init__(epsilon=epsilon, log_weight=log_weight)
        self.loss_name = loss_name

    def compute_loss(self, targets: torch.Tensor, predictions: torch.Tensor):
        loss = F.mse_loss(targets, predictions, reduction="none")
        return loss


@typechecked
class HeatmapWassersteinLoss(HeatmapLoss):
    def __init__(
        self,
        epsilon: float = 0.0,
        log_weight: float = 0.0,
        reach: Union[float, str] = "none",
    ) -> None:
        super().__init__(epsilon=epsilon, log_weight=log_weight)
        self.wasserstein_loss = SamplesLoss(
            loss="sinkhorn", reach=None if (reach == "none") else reach
        )  # maybe could speed up by creating this outside of the function
        self.loss_name = "wasserstein"

    @typechecked
    def compute_loss(
        self,
        targets: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType[
            "num_valid_keypoints", "heatmap_height", "heatmap_width"
        ],
    ) -> TensorType["num_valid_keypoints"]:
        # we should divide the loss by batch size times number of keypoints so its on per heatmap scale
        return self.wasserstein_loss(targets, predictions)


@typechecked
class PCALoss(Loss):
    def __init__(
        self,
        loss_name: Literal["pca_singleview", "pca_multiview"],
        data_module: Union[BaseDataModule, UnlabeledDataModule],
        components_to_keep: Union[int, float] = 0.95,
        empirical_epsilon_percentile: float = 0.90,
        epsilon: float = 0,
        log_weight: float = 0,
    ) -> None:
        super().__init__(epsilon=epsilon, log_weight=log_weight)
        self.loss_name = loss_name
        # initialize keypoint pca module
        self.pca = KeypointPCA(
            loss_type=self.loss_name,
            data_module=data_module,
            components_to_keep=components_to_keep,
            empirical_epsilon_percentile=empirical_epsilon_percentile,
            device=self.device,
        )
        self.pca()  # computes all the parameters needed for the loss
        self.epsilon = self.pca.parameters[
            "epsilon"
        ]  # computed empirically in KeypointPCA

    def compute_loss(self, predictions: torch.Tensor):
        # compute reprojection error
        # TODO: preds have to be reshaped before
        compute_PCA_reprojection_error(
            clean_pca_arr=predictions,
            kept_eigenvectors=self.pca.parameters["kept_eigenvectors"],
            mean=self.pca.parameters["mean"],
        )
        pass

    def __call__(self, predictions: torch.Tensor, logging: bool = True):
        # different from heatmap's
        # if multiview, reshape the predictions first
        elementwise_loss = self.compute_loss(predictions=predictions)
        epsilon_insensitive_loss = self.rectify_epsilon(loss=elementwise_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")
        if logging:
            self.log_loss(loss=scalar_loss)
        return self.weight * scalar_loss


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
def get_loss_classes() -> Dict[str, Callable]:
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
    return loss_dict
