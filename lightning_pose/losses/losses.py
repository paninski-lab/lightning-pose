"""Supervised and unsupervised losses implemented in pytorch.

The lightning pose package defines each loss as its own class; an initialized loss
object, in addition to computing the loss, stores hyperparameters related to the loss
(weight in the final objective funcion, epsilon-insensitivity parameter, etc.)

A separate LossFactory class (defined in lightning_pose.losses.factory) collects all
losses for a given model and orchestrates their execution, logging, etc.

The general flow of each loss class is as follows:
- input: predicted and ground truth data
- step 0: remove ground truth samples containing nans if desired
- step 1: compute loss for each batch element/keypoint/etc
- step 2: epsilon-insensitivity: set loss to zero for any batch element with loss < epsilon
- step 3: reduce loss (usually mean)
- step 4: log values to a dict
- step 5: return loss

"""

import os
import warnings
from typing import Literal, Tuple, Type

import torch
from kornia.losses import js_div_loss_2d, kl_div_loss_2d
from omegaconf import ListConfig
from torch.nn import functional as F
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.cameras import project_3d_to_2d, project_camera_pairs_to_3d
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.utils import generate_heatmaps
from lightning_pose.utils.pca import KeypointPCA

# to ignore imports for sphix-autoapidoc
__all__ = [
    "Loss",
    "HeatmapLoss",
    "HeatmapMSELoss",
    "HeatmapKLLoss",
    "HeatmapJSLoss",
    "PCALoss",
    "TemporalLoss",
    "TemporalHeatmapLoss",
    "UnimodalLoss",
    "RegressionMSELoss",
    "RegressionRMSELoss",
    "PairwiseProjectionsLoss",
    "ReprojectionHeatmapLoss",
    "UnsupervisedReprojectionLoss",
    "get_loss_classes",
]

_DEFAULT_TORCH_DEVICE = "cpu"
if torch.cuda.is_available():
    # When running with multiple GPUs, the LOCAL_RANK variable correctly
    # contains the DDP Local Rank, which is also the cuda device index.
    _DEFAULT_TORCH_DEVICE = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"


# @typechecked
class Loss:
    """Parent class for all losses."""

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        epsilon: float | list[float] = 0.0,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """

        Args:
            data_module: give losses access to data for computing data-specific loss params
            epsilon: loss values below epsilon will be zeroed out
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__()
        self.data_module = data_module
        # epsilon can either by a float or a list of floats
        self.epsilon = torch.tensor(epsilon, dtype=torch.float)
        self.log_weight = torch.tensor(log_weight, dtype=torch.float)

        self.reduce_methods_dict = {"mean": torch.mean, "sum": torch.sum}

    @property
    def weight(self) -> TensorType[()]:
        # weight = \sigma where our trainable parameter is \log(\sigma^2).
        # i.e., we take the parameter as it is in the config and exponentiate it to
        # enforce positivity
        weight = 1.0 / (2.0 * torch.exp(self.log_weight))
        return weight

    def remove_nans(self, **kwargs):
        # find nans in the targets, and do a masked_select operation
        raise NotImplementedError

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    def rectify_epsilon(self, loss: torch.Tensor) -> torch.Tensor:
        # loss values below epsilon as masked to zero
        loss = F.relu(loss - self.epsilon)
        return loss

    def reduce_loss(self, loss: torch.Tensor, method: str = "mean") -> TensorType[()]:
        return self.reduce_methods_dict[method](loss)

    def log_loss(
        self,
        loss: torch.Tensor,
        stage: Literal["train", "val", "test"],
    ) -> list[dict]:
        loss_dict = {
            "name": "%s_%s_loss" % (stage, self.loss_name),
            "value": loss,
            "prog_bar": True,
        }
        weight_dict = {
            "name": "%s_weight" % self.loss_name,
            "value": self.weight,
        }
        return [loss_dict, weight_dict]

    def __call__(self, *args, **kwargs):
        # give us the flow of operations, and we overwrite the methods, and determine
        # their arguments which are in buffer

        # self.remove_nans()
        # self.compute_loss()
        # self.rectify_epsilon()
        # self.reduce_loss()
        # self.log_loss()

        # return scalar_loss, logs
        raise NotImplementedError


# @typechecked
class HeatmapLoss(Loss):
    """Parent class for different heatmap losses (MSE, Wasserstein, etc)."""

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize HeatmapLoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, log_weight=log_weight)

    def remove_nans(
        self,
        targets: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ) -> Tuple[
        TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ]:

        squeezed_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
        idxs_ignore = torch.all(squeezed_targets == 0.0, dim=-1)

        return targets[~idxs_ignore], predictions[~idxs_ignore]

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        heatmaps_targ: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        heatmaps_pred: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:
        # give us the flow of operations, and we overwrite the methods, and determine
        # their arguments which are in buffer
        clean_targets, clean_predictions = self.remove_nans(
            targets=heatmaps_targ, predictions=heatmaps_pred
        )
        elementwise_loss = self.compute_loss(
            targets=clean_targets, predictions=clean_predictions
        )
        scalar_loss = self.reduce_loss(elementwise_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)

        return scalar_loss, logs


# @typechecked
class HeatmapMSELoss(HeatmapLoss):
    """MSE loss between heatmaps."""

    loss_name = "heatmap_mse"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize HeatmapMSELoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, log_weight=log_weight)

    def compute_loss(
        self,
        targets: TensorType["batch_x_num_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["batch_x_num_keypoints", "heatmap_height", "heatmap_width"],
    ) -> TensorType["batch_x_num_keypoints", "heatmap_height", "heatmap_width"]:
        h = targets.shape[1]
        w = targets.shape[2]
        # multiply by number of pixels in heatmap to standardize loss range
        loss = F.mse_loss(targets, predictions, reduction="none") * h * w
        return loss


# @typechecked
class HeatmapKLLoss(HeatmapLoss):
    """Kullback-Leibler loss between heatmaps."""

    loss_name = "heatmap_kl"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize HeatmapKLLoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, log_weight=log_weight)
        self.loss = kl_div_loss_2d

    def compute_loss(
        self,
        targets: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ) -> TensorType["num_valid_keypoints"]:
        loss = self.loss(
            predictions.unsqueeze(0) + 1e-10,
            targets.unsqueeze(0) + 1e-10,
            reduction="none",
        )
        return loss[0]


# @typechecked
class HeatmapJSLoss(HeatmapLoss):
    """Jensen-Shannon loss between heatmaps."""

    loss_name = "heatmap_js"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize HeatmapJSLoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, log_weight=log_weight)
        self.loss = js_div_loss_2d

    def compute_loss(
        self,
        targets: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ) -> TensorType["num_valid_keypoints"]:
        loss = self.loss(
            predictions.unsqueeze(0) + 1e-10,
            targets.unsqueeze(0) + 1e-10,
            reduction="none",
        )
        return loss[0]


# @typechecked
class PCALoss(Loss):
    """Penalize predictions that fall outside a low-dimensional subspace."""

    # define all valid loss names as class constants
    LOSS_NAME_MULTIVIEW = "pca_multiview"
    LOSS_NAME_SINGLEVIEW = "pca_singleview"

    def __init__(
        self,
        loss_name: Literal["pca_singleview", "pca_multiview"],
        components_to_keep: int | float = 0.95,
        empirical_epsilon_percentile: float = 99.0,
        epsilon: float | None = None,
        empirical_epsilon_multiplier: float = 1.0,
        mirrored_column_matches: ListConfig | list | None = None,
        columns_for_singleview_pca: ListConfig | list | None = None,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        log_weight: float = 0.0,
        device: Literal["cuda", "cpu"] | torch.device = _DEFAULT_TORCH_DEVICE,
        centering_method: Literal["mean", "median"] | None = None,
        **kwargs,
    ) -> None:
        """Initialize PCALoss.

        Fits a :class:`KeypointPCA` object on the training data and uses the resulting
        low-dimensional subspace to penalize out-of-subspace predictions at training time.

        Args:
            loss_name: ``"pca_singleview"`` penalizes single-camera predictions;
                ``"pca_multiview"`` penalizes predictions that are inconsistent across views.
            components_to_keep: passed to :class:`KeypointPCA`; see its docstring for details.
            empirical_epsilon_percentile: percentile of the training-data reprojection error
                used to set epsilon when ``epsilon`` is ``None``; in ``[0, 100]``.
            epsilon: if not ``None``, use this fixed epsilon value and ignore
                ``empirical_epsilon_percentile``.
            empirical_epsilon_multiplier: scalar multiplier applied to the empirically computed
                epsilon before use.
            mirrored_column_matches: required for ``"pca_multiview"``; see :class:`KeypointPCA`
                for details.
            columns_for_singleview_pca: subset of keypoint indices to use for singleview PCA;
                ``None`` uses all keypoints.
            data_module: data module used by :class:`KeypointPCA` to extract training data.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.
            device: device on which PCA parameters are stored and loss is computed.
            centering_method: if not ``None``, subtract the per-frame keypoint centroid before
                fitting PCA. ``"mean"`` uses the arithmetic mean; ``"median"`` uses the median.

        """
        super().__init__(data_module=data_module, log_weight=log_weight)
        self.device = device

        # validate against class constants
        if loss_name not in (self.LOSS_NAME_MULTIVIEW, self.LOSS_NAME_SINGLEVIEW):
            raise ValueError(f"Invalid loss_name: {loss_name}")
        self.loss_name = loss_name

        if loss_name == "pca_multiview":
            if mirrored_column_matches is None:
                raise ValueError("must provide mirrored_column_matches in data config")

        # the current data_module contains datasets that are loaded using augmentations. the
        # current solution is to pass the data module to KeypointPCA, which then passes it to
        # DataExtractor; we will also pass a "no_augmentation" arg to DataExtractor which will
        # rebuild the data module with only resizing augmentations, then extract the data.

        # initialize keypoint pca module
        # this module will fit pca on training data, and will define the error metric
        # and fuction to be used in model training.
        self.pca = KeypointPCA(
            loss_type=self.loss_name,
            data_module=data_module,
            components_to_keep=components_to_keep,
            empirical_epsilon_percentile=empirical_epsilon_percentile,
            mirrored_column_matches=mirrored_column_matches,
            columns_for_singleview_pca=columns_for_singleview_pca,
            device=device,
            centering_method=centering_method,
        )
        # compute all the parameters needed for the loss
        self.pca()
        # select epsilon based on constructor inputs
        if epsilon is not None:
            warnings.warn(
                "Using absolute epsilon=%.2f for pca loss; empirical epsilon ignored"
                % epsilon
            )
            self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=self.device)
        else:
            # empirically compute epsilon, already converted to tensor
            self.epsilon = self.pca.parameters["epsilon"] * empirical_epsilon_multiplier

            warnings.warn(
                "Using empirical epsilon=%.3f * multiplier=%.3f -> total=%.3f for %s loss"
                % (
                    float(self.pca.parameters["epsilon"]),
                    float(empirical_epsilon_multiplier),
                    float(self.epsilon),
                    self.loss_name,
                )
            )

    def remove_nans(self, **kwargs):
        # find nans in the targets, and do a masked_select operation
        pass

    def compute_loss(
        self,
        predictions: TensorType["num_samples", "sample_dim"],
    ) -> TensorType["num_samples", -1]:
        assert predictions.device == torch.device(self.device), (
            predictions.device,
            torch.device(self.device),
        )
        # compute either reprojection error or projection onto discarded evecs.
        # they will vary in the last dim, hence -1.
        return self.pca.compute_reprojection_error(data_arr=predictions)

    def __call__(
        self,
        keypoints_pred: torch.Tensor,
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:
        assert keypoints_pred.device == torch.device(self.device), (
            keypoints_pred.device,
            torch.device(self.device),
        )
        keypoints_pred = self.pca._format_data(data_arr=keypoints_pred)
        elementwise_loss = self.compute_loss(predictions=keypoints_pred)
        epsilon_insensitive_loss = self.rectify_epsilon(loss=elementwise_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)
        return scalar_loss, logs


# @typechecked
class TemporalLoss(Loss):
    """Penalize temporal differences for each target.

    Motion model: x_t = x_(t-1) + e_t, e_t ~ N(0, s)

    """

    loss_name = "temporal"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        epsilon: float | list[float] = 0.0,
        prob_threshold: float = 0.0,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize TemporalLoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            epsilon: loss values below this threshold are zeroed out. May be a scalar or a list
                with one value per keypoint.
            prob_threshold: predictions whose confidence is below this value are excluded from
                the loss computation.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, epsilon=epsilon, log_weight=log_weight)
        self.prob_threshold = torch.tensor(prob_threshold, dtype=torch.float)

    def rectify_epsilon(
        self, loss: TensorType["batch_minus_one", "num_keypoints"]
    ) -> TensorType["batch_minus_one", "num_keypoints"]:
        """Rectify supporting a list of epsilons, one per bodypart.
        Not implemented in Loss class, because shapes of broadcasting may vary"""
        # self.epsilon is a tensor initialized in parent class
        # repeating for broadcasting.
        # note: this unsqueezing doesn't affect anything if epsilon is a scalar tensor,
        # but it does if it's a tensor with multiple elements.
        epsilon = self.epsilon.unsqueeze(0).repeat(loss.shape[0], 1).to(loss.device)
        return F.relu(loss - epsilon)

    def remove_nans(
        self,
        loss: TensorType["batch_minus_one", "num_keypoints"],
        confidences: TensorType["batch", "num_keypoints"],
    ) -> TensorType["batch_minus_one", "num_keypoints"]:
        # find nans in the targets, and do a masked_select operation
        # get rid of unsupervised targets with extremely uncertain predictions or likely occlusions
        idxs_ignore = confidences < self.prob_threshold
        # ignore the loss values in the diff where one of the heatmaps is 'nan'
        union_idxs_ignore = torch.zeros(
            (confidences.shape[0] - 1, confidences.shape[1]),
            dtype=torch.bool,
            device=loss.device,
        )
        for i in range(confidences.shape[0] - 1):
            union_idxs_ignore[i] = torch.logical_or(idxs_ignore[i], idxs_ignore[i + 1])

        # clone loss and zero out the nan values
        clean_loss = loss.clone()
        clean_loss[union_idxs_ignore] = 0.0
        return clean_loss

    def compute_loss(
        self,
        predictions: TensorType["batch", "two_x_num_keypoints"],
    ) -> TensorType["batch_minus_one", "num_keypoints"]:

        #  return shape: (batch - 1, num_targets)
        diffs = torch.diff(predictions, dim=0)

        # return shape: (batch - 1, num_keypoints, 2)
        reshape = torch.reshape(diffs, (diffs.shape[0], -1, 2))

        # return shape (batch - 1, num_keypoints)
        loss = torch.linalg.norm(reshape, ord=2, dim=2)

        return loss

    def __call__(
        self,
        keypoints_pred: TensorType["batch", "two_x_num_keypoints"],
        confidences: TensorType["batch", "num_keypoints"] = None,
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:

        elementwise_loss = self.compute_loss(predictions=keypoints_pred)
        # do remove nans with loss to remove temporal difference values
        clean_loss = (
            self.remove_nans(loss=elementwise_loss, confidences=confidences)
            if confidences is not None
            else elementwise_loss
        )
        epsilon_insensitive_loss = self.rectify_epsilon(loss=clean_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)
        return scalar_loss, logs


# @typechecked
class TemporalHeatmapLoss(Loss):
    """Penalize temporal differences for each heatmap.

    Motion model: x_t = x_(t-1) + e_t, e_t ~ N(0, s)

    """

    LOSS_NAME_MSE = "temporal_heatmap_mse"
    LOSS_NAME_KL = "temporal_heatmap_kl"

    def __init__(
        self,
        loss_name: Literal["temporal_heatmap_mse", "temporal_heatmap_kl"],
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        epsilon: float | list[float] = 0.0,
        prob_threshold: float = 0.0,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize TemporalHeatmapLoss.

        Args:
            loss_name: ``"temporal_heatmap_mse"`` uses pixel-wise MSE between consecutive
                heatmaps; ``"temporal_heatmap_kl"`` uses the KL divergence.
            data_module: data module providing access to datasets; passed to the parent class.
            epsilon: loss values below this threshold are zeroed out. May be a scalar or a list
                with one value per keypoint.
            prob_threshold: predictions whose confidence is below this value are excluded from
                the loss computation.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, epsilon=epsilon, log_weight=log_weight)

        if loss_name not in (self.LOSS_NAME_MSE, self.LOSS_NAME_KL):
            raise ValueError(f"Invalid loss_name: {loss_name}")
        self.loss_name = loss_name

        if self.loss_name == "temporal_heatmap_mse":
            self.hmloss = None
        elif self.loss_name == "temporal_heatmap_kl":
            self.hmloss = kl_div_loss_2d
        else:
            raise NotImplementedError

        self.prob_threshold = torch.tensor(prob_threshold, dtype=torch.float)

    def rectify_epsilon(
        self, loss: TensorType["batch_minus_one", "num_valid_keypoints"]
    ) -> TensorType["batch_minus_one", "num_valid_keypoints"]:
        """Rectify supporting a list of epsilons, one per bodypart.
        Not implemented in Loss class, because shapes of broadcasting may vary"""
        # self.epsilon is a tensor initialized in parent class
        # repeating for broadcasting.
        # note: this unsqueezing doesn't affect anything if epsilon is a scalar tensor,
        # but it does if it's a tensor with multiple elements.
        epsilon = self.epsilon.unsqueeze(0).repeat(loss.shape[0], 1).to(loss.device)
        return F.relu(loss - epsilon)

    def remove_nans(
        self,
        confidences: TensorType["batch", "num_keypoints"],
        loss: TensorType["batch_minus_one", "num_keypoints"],
    ) -> TensorType["batch_minus_one", "num_keypoints"]:
        # find nans in the targets, and do a masked_select operation
        # get rid of unsupervised targets with extremely uncertain predictions or likely occlusions
        idxs_ignore = confidences < self.prob_threshold
        # ignore the loss values in the diff where one of the heatmaps is 'nan'
        union_idxs_ignore = torch.zeros(
            (confidences.shape[0] - 1, confidences.shape[1]), dtype=torch.bool
        ).to(loss.device)
        for i in range(confidences.shape[0] - 1):
            union_idxs_ignore[i] = torch.logical_or(idxs_ignore[i], idxs_ignore[i + 1])

        loss[union_idxs_ignore] = 0.0
        return loss

    def compute_loss(
        self,
        predictions: TensorType["batch", "num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ) -> TensorType["batch_minus_one", "num_valid_keypoints"]:
        # compute the differences between matching heatmaps for each keypoint

        diffs = torch.zeros(
            (predictions.shape[0] - 1, predictions.shape[1]), device=predictions.device
        )

        for i in range(diffs.shape[0]):
            if self.loss_name == "temporal_heatmap_mse":
                curr_mse = F.mse_loss(
                    predictions[i], predictions[i + 1], reduction="none"
                ).reshape(predictions.shape[1], -1)
                diffs[i] = torch.mean(curr_mse, dim=-1)
            elif self.loss_name == "temporal_heatmap_kl":
                diffs[i] = self.hmloss(
                    predictions[i].unsqueeze(0) + 1e-10,
                    predictions[i + 1].unsqueeze(0) + 1e-10,
                    reduction="none",
                )

        return diffs

    def __call__(
        self,
        heatmaps_pred: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        confidences: TensorType["batch", "num_keypoints"],
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:

        elementwise_loss = self.compute_loss(predictions=heatmaps_pred)
        # remove nan after loss is computed to get rid of diff vals with a bad heatmap
        clean_loss = self.remove_nans(confidences=confidences, loss=elementwise_loss)
        epsilon_insensitive_loss = self.rectify_epsilon(loss=clean_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)

        return scalar_loss, logs


# @typechecked
class UnimodalLoss(Loss):
    """Encourage heatmaps to be unimodal using various measures."""

    LOSS_NAME_MSE = "unimodal_mse"
    LOSS_NAME_KL = "unimodal_kl"
    LOSS_NAME_JS = "unimodal_js"

    def __init__(
        self,
        loss_name: Literal["unimodal_mse", "unimodal_kl", "unimodal_js"],
        original_image_height: int,
        original_image_width: int,
        downsampled_image_height: int,
        downsampled_image_width: int,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        prob_threshold: float = 0.0,
        log_weight: float = 0.0,
        uniform_heatmaps: bool = False,
        **kwargs,
    ) -> None:
        """Initialize UnimodalLoss.

        Generates an ideal unimodal heatmap from each predicted keypoint coordinate and
        penalizes the difference between that ideal heatmap and the network's predicted heatmap.

        Args:
            loss_name: divergence measure to use. ``"unimodal_mse"`` uses pixel-wise MSE;
                ``"unimodal_kl"`` uses KL divergence; ``"unimodal_js"`` uses Jensen-Shannon
                divergence.
            original_image_height: height of the full-resolution input image in pixels, used
                when generating ideal heatmaps.
            original_image_width: width of the full-resolution input image in pixels.
            downsampled_image_height: height of the heatmap output (after backbone downsampling).
            downsampled_image_width: width of the heatmap output.
            data_module: data module providing access to datasets; passed to the parent class.
            prob_threshold: predictions whose confidence is below this value are excluded from
                the loss computation.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.
            uniform_heatmaps: if ``True``, generate uniform (flat) target heatmaps for NaN
                ground truth keypoints instead of ignoring them in the loss.

        """

        super().__init__(data_module=data_module, log_weight=log_weight)

        if loss_name not in (self.LOSS_NAME_MSE, self.LOSS_NAME_KL, self.LOSS_NAME_JS):
            raise ValueError(f"Invalid loss_name: {loss_name}")
        self.loss_name = loss_name

        self.original_image_height = original_image_height
        self.original_image_width = original_image_width
        self.downsampled_image_height = downsampled_image_height
        self.downsampled_image_width = downsampled_image_width
        self.uniform_heatmaps = uniform_heatmaps

        self.prob_threshold = torch.tensor(prob_threshold, dtype=torch.float)

        if self.loss_name == "unimodal_mse":
            self.loss = None
        elif self.loss_name == "unimodal_kl":
            self.loss = kl_div_loss_2d
        elif self.loss_name == "unimodal_js":
            self.loss = js_div_loss_2d
        else:
            raise NotImplementedError

    def remove_nans(
        self,
        targets: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        confidences: TensorType["batch", "num_keypoints"],
    ) -> Tuple[
        TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Remove nans from targets and predictions.
        Args:
            targets: (batch, num_keypoints, heatmap_height, heatmap_width)
            predictions: (batch, num_keypoints, heatmap_height, heatmap_width)
            confidences: (batch, num_keypoints)
        Returns:
            clean targets: concatenated across different images and keypoints
            clean predictions: concatenated across different images and keypoints
        """
        # use confidences to get rid of unsupervised targets with likely occlusions
        idxs_ignore = confidences < self.prob_threshold

        return targets[~idxs_ignore], predictions[~idxs_ignore]

    def compute_loss(
        self,
        targets: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["num_valid_keypoints", "heatmap_height", "heatmap_width"],
    ) -> torch.Tensor:

        if self.loss_name == "unimodal_mse":
            return F.mse_loss(targets, predictions, reduction="none")
        elif self.loss_name == "unimodal_kl":
            return self.loss(
                predictions.unsqueeze(0) + 1e-10,
                targets.unsqueeze(0) + 1e-10,
                reduction="none",
            )
        elif self.loss_name == "unimodal_js":
            return self.loss(
                predictions.unsqueeze(0) + 1e-10,
                targets.unsqueeze(0) + 1e-10,
                reduction="none",
            )
        else:
            raise NotImplementedError

    def __call__(
        self,
        keypoints_pred_augmented: TensorType["batch", "two_x_num_keypoints"],
        heatmaps_pred: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        confidences: TensorType["batch", "num_keypoints"],
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:
        """Compute unimodal loss.

        Args:
            keypoints_pred_augmented: these are in the augmented image space
            heatmaps_pred: also in the augmented space, matching the keypoints_pred_augmented

        """

        # turn keypoint predictions into unimodal heatmaps
        keypoints_pred = keypoints_pred_augmented.reshape(keypoints_pred_augmented.shape[0], -1, 2)
        heatmaps_ideal = generate_heatmaps(  # this process doesn't compute gradients
            keypoints=keypoints_pred,
            height=self.original_image_height,
            width=self.original_image_width,
            output_shape=(self.downsampled_image_height, self.downsampled_image_width),
            uniform_heatmaps=self.uniform_heatmaps,
        )

        # remove invisible keypoints according to confidences
        clean_targets, clean_predictions = self.remove_nans(
            targets=heatmaps_ideal, predictions=heatmaps_pred, confidences=confidences
        )
        # compute loss just on the valid heatmaps
        elementwise_loss = self.compute_loss(
            targets=clean_targets, predictions=clean_predictions
        )
        scalar_loss = self.reduce_loss(elementwise_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)

        return scalar_loss, logs


# @typechecked
class RegressionMSELoss(Loss):
    """MSE loss between ground truth and predicted coordinates."""

    loss_name = "regression"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        epsilon: float = 0.0,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize RegressionMSELoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            epsilon: loss values below this threshold are zeroed out.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, epsilon=epsilon, log_weight=log_weight)

    def remove_nans(
        self,
        targets: TensorType["batch", "two_x_num_keypoints"],
        predictions: TensorType["batch", "two_x_num_keypoints"],
    ) -> Tuple[TensorType["num_valid_keypoints"], TensorType["num_valid_keypoints"]]:
        mask = targets == targets  # keypoints is not none, bool
        targets_masked = torch.masked_select(targets, mask)
        predictions_masked = torch.masked_select(predictions, mask)
        return targets_masked, predictions_masked

    def compute_loss(
        self,
        targets: TensorType["batch_x_two_x_num_keypoints"],
        predictions: TensorType["batch_x_two_x_num_keypoints"],
    ) -> TensorType["batch_x_two_x_num_keypoints"]:
        loss = F.mse_loss(targets, predictions, reduction="none")
        return loss

    def __call__(
        self,
        keypoints_targ: TensorType["batch", "two_x_num_keypoints"],
        keypoints_pred: TensorType["batch", "two_x_num_keypoints"],
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:

        clean_targets, clean_predictions = self.remove_nans(
            targets=keypoints_targ, predictions=keypoints_pred
        )
        elementwise_loss = self.compute_loss(
            targets=clean_targets, predictions=clean_predictions
        )
        scalar_loss = self.reduce_loss(elementwise_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)

        return scalar_loss, logs


# @typechecked
class RegressionRMSELoss(RegressionMSELoss):
    """Root MSE loss between ground truth and predicted coordinates."""

    loss_name = "rmse"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        epsilon: float = 0.0,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize RegressionRMSELoss.

        Args:
            data_module: data module providing access to datasets; passed to the parent class.
            epsilon: loss values below this threshold are zeroed out.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(data_module=data_module, epsilon=epsilon, log_weight=log_weight)

    def compute_loss(
        self,
        targets: TensorType["batch_x_two_x_num_keypoints"],
        predictions: TensorType["batch_x_two_x_num_keypoints"],
    ) -> TensorType["batch_x_num_keypoints"]:
        targs = targets.reshape(-1, 2)
        preds = predictions.reshape(-1, 2)
        loss = torch.mean(F.mse_loss(targs, preds, reduction="none"), dim=1)
        return torch.sqrt(loss)


class PairwiseProjectionsLoss(Loss):
    """Penalize projections from each pair of cameras into 3D world space."""

    loss_name = "supervised_pairwise_projections"

    def __init__(self, log_weight: float = 0.0, **kwargs) -> None:
        """Initialize PairwiseProjectionsLoss.

        Args:
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.

        """
        super().__init__(log_weight=log_weight)

    def remove_nans(
        self,
        loss: TensorType["batch", "cam_pairs", "num_keypoints"],
    ) -> TensorType["valid_losses"]:
        mask = ~torch.isnan(loss)
        valid_losses = torch.masked_select(loss, mask)
        if valid_losses.numel() == 0:
            # No valid losses, return zero that preserves gradients
            # Use torch.where to avoid nan*0.0 issues
            dummy_loss = torch.where(mask, loss, torch.zeros_like(loss))
            return dummy_loss.sum()  # This will be 0.0 and preserve gradients
        else:
            return valid_losses

    def compute_loss(
        self,
        targets: TensorType["batch", "num_keypoints", 3],
        predictions: TensorType["batch", "cam_pairs", "num_keypoints", 3],
    ) -> TensorType["batch", "cam_pairs", "num_keypoints"]:

        # Check for NaN targets AND predictions
        nan_targets = torch.isnan(targets).any(dim=-1)  # [batch, num_keypoints]
        nan_predictions = torch.isnan(predictions).any(dim=-1)  # [batch, cam_pairs, num_keypoints]

        # Expand target NaN mask to match prediction dimensions
        nan_targets_expanded = nan_targets.unsqueeze(1)  # [batch, 1, num_keypoints]

        # Combined NaN mask
        combined_nan_mask = \
            nan_targets_expanded | nan_predictions  # [batch, cam_pairs, num_keypoints]

        # Create clean targets and predictions - replace NaNs with zeros and detach
        clean_targets = torch.where(
            nan_targets.unsqueeze(-1),  # [batch, num_keypoints, 1]
            torch.zeros_like(targets).detach(),
            targets,
        )

        clean_predictions = torch.where(
            combined_nan_mask.unsqueeze(-1),  # [batch, cam_pairs, num_keypoints, 1]
            torch.zeros_like(predictions).detach(),
            predictions,
        )

        # Compute loss with clean tensors
        loss = torch.linalg.norm(clean_targets.unsqueeze(1) - clean_predictions, ord=2, dim=-1)

        # Set loss to NaN where either targets or predictions were originally NaN
        loss = torch.where(
            combined_nan_mask,
            torch.tensor(float('nan'), device=loss.device, dtype=loss.dtype),
            loss,
        )

        return loss

    def __call__(
        self,
        keypoints_targ_3d: TensorType["batch", "num_keypoints", 3],
        keypoints_pred_3d: TensorType["batch", "cam_pairs", "num_keypoints", 3],
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:

        # check if 3D keypoints are available
        if keypoints_targ_3d is None or keypoints_pred_3d is None:
            raise ValueError(
                f"3D keypoints not available for {stage} stage. "
                "Camera params file is required but not found;"
                "Turn off supervised_pairwise_projections loss to avoid this error."
            )

        elementwise_loss = self.compute_loss(
            targets=keypoints_targ_3d,
            predictions=keypoints_pred_3d,
        )
        clean_loss = self.remove_nans(loss=elementwise_loss)
        scalar_loss = self.reduce_loss(clean_loss, method="mean")

        logs = self.log_loss(loss=scalar_loss, stage=stage)

        return scalar_loss, logs


class ReprojectionHeatmapLoss(Loss):
    """Penalize error between predicted 2D->3D->2D->heatmap and ground truth heatmap."""

    loss_name = "supervised_reprojection_heatmap_mse"

    def __init__(
        self,
        original_image_height: int,
        original_image_width: int,
        downsampled_image_height: int,
        downsampled_image_width: int,
        log_weight: float = 0.0,
        uniform_heatmaps: bool = False,
        **kwargs,
    ) -> None:
        """Initialize ReprojectionHeatmapLoss.

        Converts 2D reprojected keypoints (obtained by projecting 3D triangulated predictions
        back into each camera's image plane) into heatmaps and compares them with the ground
        truth heatmaps using pixel-wise MSE.

        Args:
            original_image_height: height of the full-resolution input image in pixels.
            original_image_width: width of the full-resolution input image in pixels.
            downsampled_image_height: height of the heatmap output (after backbone downsampling).
            downsampled_image_width: width of the heatmap output.
            log_weight: final weight in front of the loss term in the objective function is
                computed as ``1.0 / (2.0 * exp(log_weight))``.
            uniform_heatmaps: if ``True``, generate uniform (flat) target heatmaps for NaN
                ground truth keypoints instead of ignoring them in the loss.

        """
        super().__init__(log_weight=log_weight)
        self.original_image_height = original_image_height
        self.original_image_width = original_image_width
        self.downsampled_image_height = downsampled_image_height
        self.downsampled_image_width = downsampled_image_width
        self.uniform_heatmaps = uniform_heatmaps

    def remove_nans(
        self,
        loss: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        targets: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ) -> TensorType["valid_losses"]:
        # Create mask for valid keypoints (non-zero targets)
        squeezed_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
        valid_keypoints = ~torch.all(squeezed_targets == 0.0, dim=-1)  # [batch, num_keypoints]

        # Expand mask to match loss dimensions
        valid_mask = valid_keypoints.unsqueeze(-1).unsqueeze(-1)  # [batch, num_keypoints, 1, 1]
        valid_mask = valid_mask.expand_as(loss)  # [batch, num_keypoints, h, w]

        valid_losses = torch.masked_select(loss, valid_mask)

        if valid_losses.numel() == 0:
            # No valid losses, return zero that preserves gradients
            dummy_loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
            return dummy_loss.sum()  # This will be 0.0 and preserve gradients
        else:
            return valid_losses

    def compute_loss(
        self,
        targets: TensorType["batch_x_num_keypoints", "heatmap_height", "heatmap_width"],
        predictions: TensorType["batch_x_num_keypoints", "heatmap_height", "heatmap_width"],
    ) -> TensorType["batch_x_num_keypoints", "heatmap_height", "heatmap_width"]:
        h = targets.shape[1]
        w = targets.shape[2]
        # multiply by number of pixels in heatmap to standardize loss range
        loss = F.mse_loss(targets, predictions, reduction="none") * h * w
        return loss

    def __call__(
        self,
        heatmaps_targ: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        keypoints_pred_2d_reprojected: TensorType["batch", "num_keypoints", 2],
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:

        # check if reprojected keypoints are available
        if keypoints_pred_2d_reprojected is None:
            raise ValueError(
                f"Reprojected keypoints not available for {stage} stage. "
                "Camera params file is required but not found;"
                "Turn off supervised_reprojection_heatmap loss to avoid this error."
            )

        # create heatmaps from 2d reprojections
        heatmaps_pred = generate_heatmaps(
            keypoints=keypoints_pred_2d_reprojected,
            height=self.original_image_height,
            width=self.original_image_width,
            output_shape=(self.downsampled_image_height, self.downsampled_image_width),
            uniform_heatmaps=self.uniform_heatmaps,
            keep_gradients=True,
        )

        elementwise_loss = self.compute_loss(targets=heatmaps_targ, predictions=heatmaps_pred)
        clean_loss = self.remove_nans(loss=elementwise_loss, targets=heatmaps_targ)
        scalar_loss = self.reduce_loss(clean_loss, method="mean")

        logs = self.log_loss(loss=scalar_loss, stage=stage)

        return scalar_loss, logs


class UnsupervisedReprojectionLoss(Loss):
    """Cycle-reprojection consistency loss for unlabeled multiview data.

    Triangulates predicted 2D keypoints to 3D via pairwise camera triangulation,
    aggregates pairs with ``nanmedian`` for robustness, then reprojects the 3D point
    back to each view and penalizes the per-keypoint L2 distance between the original
    prediction and the reprojection.

    Path A v1: a single session's calibration is loaded at init time from
    ``data_module.dataset.cam_params_file_to_camgroup`` and reused for every unlabeled
    batch. For fly-anipose with multi-session calibration, pass the session key via
    the ``session`` kwarg; otherwise the first entry is used.
    """

    loss_name = "cycle_reprojection"

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        log_weight: float = 0.0,
        epsilon: float = 0.0,
        prob_threshold: float = 0.05,
        session: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Initialize UnsupervisedReprojectionLoss.

        Args:
            data_module: labeled data module; must expose a dataset with
                ``cam_params_file_to_camgroup`` populated.
            log_weight: final weight = ``1.0 / (2.0 * exp(log_weight))``.
            epsilon: per-keypoint L2 reprojection error (pixels) below which loss
                contributions are zeroed out.
            prob_threshold: minimum confidence for a keypoint to contribute to the
                loss. Applied per (view, keypoint).
            session: optional key from the ``file`` column of ``camera_params_file``
                selecting which session's calibration to use. If ``None``, the first
                session is used.
            verbose: if True, prints a one-line reprojection-error summary every call.
                A full init-time diagnostic and a first-forward diagnostic print are
                emitted regardless of this flag.

        """
        super().__init__(data_module=data_module, epsilon=epsilon, log_weight=log_weight)
        self.prob_threshold = torch.tensor(prob_threshold, dtype=torch.float)
        self.verbose = verbose
        self._did_first_call_debug = False

        cam_params_dict = getattr(
            getattr(data_module, "dataset", None), "cam_params_file_to_camgroup", None,
        )
        if not cam_params_dict:
            raise ValueError(
                "cycle_reprojection loss requires camera calibration. Set "
                "data.camera_params_file in the config."
            )

        if session is not None:
            if session not in cam_params_dict:
                raise ValueError(
                    f"cycle_reprojection: session '{session}' not in camera_params_file. "
                    f"Available: {list(cam_params_dict.keys())}"
                )
        else:
            session = next(iter(cam_params_dict))
        camgroup = cam_params_dict[session]
        self.session_used = session

        # store calibration as plain tensors (Loss is not an nn.Module, so register_buffer
        # is unavailable); they are moved to the input device inside __call__, matching the
        # pattern used by TemporalLoss.rectify_epsilon / PCALoss.compute_loss.
        self.cal_intrinsic = torch.stack(
            [torch.tensor(cam.get_camera_matrix()) for cam in camgroup.cameras], dim=0,
        ).float()
        self.cal_extrinsic = torch.stack(
            [torch.tensor(cam.get_extrinsics_mat()[:3]) for cam in camgroup.cameras], dim=0,
        ).float()
        self.cal_distortions = torch.stack(
            [torch.tensor(cam.get_distortions()) for cam in camgroup.cameras], dim=0,
        ).float()
        self.num_views = self.cal_intrinsic.shape[0]

        self._print_init_diagnostics(cam_params_dict)

    def _print_init_diagnostics(self, cam_params_dict: dict) -> None:
        """Dump calibration summary + a self-test triangulation at init time."""
        prefix = "[cycle_reprojection]"
        print(f"{prefix} Path A v1 — single-session calibration (does NOT vary per batch).")
        print(f"{prefix} sessions in camera_params_file ({len(cam_params_dict)}):")
        for k in cam_params_dict:
            marker = "  <-- USED" if k == self.session_used else ""
            print(f"{prefix}   - {k}{marker}")
        if len(cam_params_dict) > 1:
            print(
                f"{prefix} WARNING: {len(cam_params_dict)} sessions found but only "
                f"'{self.session_used}' is used. If unlabeled videos come from other "
                f"sessions, their reprojection residuals will be computed against the "
                f"WRONG camera geometry. See Path B to fix per-session."
            )
        print(f"{prefix} num_views: {self.num_views}")
        for v in range(self.num_views):
            K = self.cal_intrinsic[v]
            E = self.cal_extrinsic[v]
            fx, fy = K[0, 0].item(), K[1, 1].item()
            cx, cy = K[0, 2].item(), K[1, 2].item()
            t_norm = E[:, 3].norm().item()
            print(
                f"{prefix}   view {v}: fx={fx:.1f} fy={fy:.1f} "
                f"cx={cx:.1f} cy={cy:.1f} |t|={t_norm:.2f}"
            )
        self._self_test_roundtrip()

    def _self_test_roundtrip(self) -> None:
        """3D -> 2D -> 3D round-trip on the scene center; residual should be ~0."""
        prefix = "[cycle_reprojection]"
        try:
            # scene anchor: centroid of camera centers (C_v = -R_v^T t_v)
            R = self.cal_extrinsic[:, :, :3]                  # (V, 3, 3)
            t = self.cal_extrinsic[:, :, 3]                   # (V, 3)
            centers = -torch.einsum(
                "vij,vj->vi", R.transpose(-1, -2), t,
            )                                                  # (V, 3)
            anchor = centers.mean(dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

            intr = self.cal_intrinsic.unsqueeze(0)            # (1, V, 3, 3)
            extr = self.cal_extrinsic.unsqueeze(0)            # (1, V, 3, 4)
            dst = self.cal_distortions.unsqueeze(0)           # (1, V, D)

            proj_2d = project_3d_to_2d(
                points_3d=anchor, intrinsics=intr, extrinsics=extr, dist=dst,
            )                                                  # (1, V, 1, 2)
            if torch.isnan(proj_2d).any():
                print(
                    f"{prefix} self-test: anchor projection produced NaN "
                    f"(anchor behind some camera). Skipping round-trip."
                )
                return
            tri_3d = project_camera_pairs_to_3d(
                points=proj_2d, intrinsics=intr, extrinsics=extr, dist=dst,
            )                                                  # (1, pairs, 1, 3)
            median_3d = torch.nanmedian(tri_3d, dim=1).values   # (1, 1, 3)
            residual = (median_3d - anchor).norm(dim=-1).item()
            print(
                f"{prefix} self-test 3D->2D->3D residual: {residual:.4e} "
                f"(expect ~1e-3 or smaller for a well-calibrated session)"
            )
            if residual > 0.1:
                print(
                    f"{prefix} WARNING: large self-test residual — check that "
                    "camera_params_file units match keypoint units (world coords "
                    "vs pixels, meters vs millimeters, etc.)."
                )
        except Exception as e:
            print(f"{prefix} self-test FAILED with {type(e).__name__}: {e}")

    def compute_loss(
        self,
        keypoints_pred: TensorType["batch", "num_views", "num_keypoints", 2],
        intrinsics: TensorType["batch", "num_views", 3, 3],
        extrinsics: TensorType["batch", "num_views", 3, 4],
        dist: TensorType["batch", "num_views", "num_params"],
    ) -> TensorType["batch", "num_views", "num_keypoints"]:
        points_3d_pairs = project_camera_pairs_to_3d(
            points=keypoints_pred, intrinsics=intrinsics, extrinsics=extrinsics, dist=dist,
        )
        # median over cam_pairs; nanmedian ignores failed triangulations
        points_3d = torch.nanmedian(points_3d_pairs, dim=1).values
        reprojected_2d = project_3d_to_2d(
            points_3d=points_3d, intrinsics=intrinsics, extrinsics=extrinsics, dist=dist,
        )
        return torch.linalg.norm(keypoints_pred - reprojected_2d, ord=2, dim=-1)

    def remove_nans(
        self,
        loss: TensorType["batch", "num_views", "num_keypoints"],
        confidences: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = ~torch.isnan(loss)
        if confidences is not None:
            conf = confidences.reshape(loss.shape)
            valid_mask = valid_mask & (conf >= self.prob_threshold.to(conf.device))
        valid_losses = torch.masked_select(loss, valid_mask)
        if valid_losses.numel() == 0:
            # zero that preserves gradients (mirrors PairwiseProjectionsLoss.remove_nans)
            dummy = torch.where(valid_mask, loss, torch.zeros_like(loss))
            return dummy.sum()
        return valid_losses

    def __call__(
        self,
        keypoints_pred: TensorType["batch", "two_x_num_views_x_num_keypoints"],
        confidences: TensorType["batch", "num_views_x_num_keypoints"] | None = None,
        stage: Literal["train", "val", "test"] | None = None,
        **kwargs,
    ) -> Tuple[TensorType[()], list[dict]]:
        batch = keypoints_pred.shape[0]
        num_keypoints = keypoints_pred.shape[1] // 2 // self.num_views
        pred = keypoints_pred.reshape(batch, self.num_views, num_keypoints, 2)

        device = keypoints_pred.device
        intrinsics = self.cal_intrinsic.to(device).unsqueeze(0).expand(batch, -1, -1, -1)
        extrinsics = self.cal_extrinsic.to(device).unsqueeze(0).expand(batch, -1, -1, -1)
        dist = self.cal_distortions.to(device).unsqueeze(0).expand(batch, -1, -1)

        elementwise_loss = self.compute_loss(
            keypoints_pred=pred, intrinsics=intrinsics, extrinsics=extrinsics, dist=dist,
        )

        if not self._did_first_call_debug:
            self._did_first_call_debug = True
            self._print_first_call_debug(
                keypoints_pred=keypoints_pred,
                pred=pred,
                elementwise_loss=elementwise_loss,
                confidences=confidences,
            )

        clean_loss = self.remove_nans(loss=elementwise_loss, confidences=confidences)
        epsilon_insensitive_loss = self.rectify_epsilon(loss=clean_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")

        if self.verbose:
            with torch.no_grad():
                not_nan = ~torch.isnan(elementwise_loss)
                n_not_nan = int(not_nan.sum().item())
                n_total = int(not_nan.numel())
                conf_str = ""
                if confidences is not None:
                    conf_reshaped = confidences.reshape(elementwise_loss.shape)
                    conf_mask = conf_reshaped >= self.prob_threshold.to(confidences.device)
                    n_conf = int((not_nan & conf_mask).sum().item())
                    conf_mean = conf_reshaped[not_nan].mean().item() if n_not_nan > 0 else float("nan")
                    conf_str = f" conf_mean={conf_mean:.4f} conf_pass={n_conf}/{n_not_nan}"
                    valid_final = not_nan & conf_mask
                else:
                    valid_final = not_nan
                    n_conf = n_not_nan
                raw_mean = (
                    elementwise_loss[valid_final].mean().item()
                    if n_conf > 0 else float("nan")
                )
                print(
                    f"[cycle_reprojection] stage={stage} batch={pred.shape[0]} "
                    f"not_nan={n_not_nan}/{n_total}{conf_str} "
                    f"raw_mean_pix={raw_mean:.4f} "
                    f"post_eps_scalar={scalar_loss.item():.4f}"
                )

        logs = self.log_loss(loss=scalar_loss, stage=stage)
        return scalar_loss, logs

    def _print_first_call_debug(
        self,
        keypoints_pred: torch.Tensor,
        pred: torch.Tensor,
        elementwise_loss: torch.Tensor,
        confidences: torch.Tensor | None,
    ) -> None:
        """One-shot diagnostic on the first forward call."""
        prefix = "[cycle_reprojection]"
        with torch.no_grad():
            print(f"{prefix} FIRST FORWARD")
            print(
                f"{prefix}   keypoints_pred: shape={tuple(keypoints_pred.shape)} "
                f"device={keypoints_pred.device} dtype={keypoints_pred.dtype}"
            )
            print(
                f"{prefix}   calibration (stored on {self.cal_intrinsic.device}, "
                f"moved to {keypoints_pred.device} for compute)"
            )
            if confidences is not None:
                print(
                    f"{prefix}   confidences: shape={tuple(confidences.shape)} "
                    f"min={confidences.min().item():.3f} "
                    f"mean={confidences.mean().item():.3f} "
                    f"max={confidences.max().item():.3f}"
                )
            # pred stats (detect obvious scale/units mismatch vs calibration)
            finite = torch.isfinite(pred)
            if finite.any():
                vals = pred[finite]
                print(
                    f"{prefix}   pred coords (pixels?): "
                    f"min={vals.min().item():.1f} "
                    f"max={vals.max().item():.1f} "
                    f"mean={vals.mean().item():.1f}"
                )
            # reprojection error distribution pre-epsilon, pre-conf-mask
            finite_loss = elementwise_loss[~torch.isnan(elementwise_loss)]
            if finite_loss.numel() > 0:
                q = torch.quantile(
                    finite_loss.float(), torch.tensor([0.5, 0.95], device=finite_loss.device),
                )
                print(
                    f"{prefix}   reprojection err (pixels, pre-epsilon): "
                    f"mean={finite_loss.mean().item():.2f} "
                    f"median={q[0].item():.2f} "
                    f"p95={q[1].item():.2f} "
                    f"max={finite_loss.max().item():.2f}"
                )
                if finite_loss.mean().item() > 1e3:
                    print(
                        f"{prefix}   WARNING: reprojection error >1000 px suggests "
                        "geometry/units mismatch (wrong session calibration for these "
                        "unlabeled frames, or pixel vs normalized-coord mismatch)."
                    )
            nan_frac_pred = torch.isnan(pred).any(dim=-1).float().mean().item()
            nan_frac_loss = torch.isnan(elementwise_loss).float().mean().item()
            print(
                f"{prefix}   NaN fraction: pred={nan_frac_pred:.3f} "
                f"elementwise_loss={nan_frac_loss:.3f}"
            )


@typechecked
def get_loss_classes() -> dict[str, Type[Loss]]:
    """Get a dict with all the loss classes.

    Returns:
        dict[str, Callable]: [description]

    """
    loss_dict = {
        RegressionMSELoss.loss_name: RegressionMSELoss,
        HeatmapMSELoss.loss_name: HeatmapMSELoss,
        HeatmapKLLoss.loss_name: HeatmapKLLoss,
        HeatmapJSLoss.loss_name: HeatmapJSLoss,
        PCALoss.LOSS_NAME_MULTIVIEW: PCALoss,
        PCALoss.LOSS_NAME_SINGLEVIEW: PCALoss,
        TemporalLoss.loss_name: TemporalLoss,
        TemporalHeatmapLoss.LOSS_NAME_MSE: TemporalHeatmapLoss,
        TemporalHeatmapLoss.LOSS_NAME_KL: TemporalHeatmapLoss,
        UnimodalLoss.LOSS_NAME_MSE: UnimodalLoss,
        UnimodalLoss.LOSS_NAME_KL: UnimodalLoss,
        UnimodalLoss.LOSS_NAME_JS: UnimodalLoss,
        PairwiseProjectionsLoss.loss_name: PairwiseProjectionsLoss,
        ReprojectionHeatmapLoss.loss_name: ReprojectionHeatmapLoss,
        UnsupervisedReprojectionLoss.loss_name: UnsupervisedReprojectionLoss,
    }
    return loss_dict
