"""Models that produce (x, y) coordinates of keypoints from images."""

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing_extensions import Literal

from pose_est_nets.losses.factory import LossFactory
from pose_est_nets.losses.losses import RegressionRMSELoss
from pose_est_nets.models.base import (
    BaseBatchDict,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)

patch_typeguard()  # use before @typechecked


class RegressionTracker(BaseSupervisedTracker):
    """Base model that produces (x, y) predictions of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: bool = True,
        representation_dropout_rate: float = 0.2,
        last_resnet_layer_to_get: int = -2,
        torch_seed: int = 123,
    ) -> None:
        """Base model that produces (x, y) coordinates of keypoints from images.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            pretrained: True to load pretrained imagenet weights
            representation_dropout_rate: dropout in the final fully connected
                layers
            last_resnet_layer_to_get: skip final layers of backbone model
            torch_seed: make weight initialization reproducible

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(
            resnet_version=resnet_version,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
        )
        self.num_keypoints = num_keypoints
        self.num_targets = self.num_keypoints * 2
        self.loss_factory = loss_factory
        self.resnet_version = resnet_version
        self.final_layer = nn.Linear(self.num_fc_input_features, self.num_targets)
        # TODO: consider removing dropout
        self.representation_dropout = nn.Dropout(p=representation_dropout_rate)
        self.torch_seed = torch_seed

        # use this to log auxiliary information: rmse on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters(ignore="loss_factory")  # cannot be pickled

    @staticmethod
    @typechecked
    def reshape_representation(
        representation: TensorType["batch", "features", "rep_height", "rep_width"]
    ) -> TensorType["batch", "features"]:
        return representation.reshape(representation.shape[0], representation.shape[1])

    @typechecked
    def forward(
        self,
        images: TensorType["batch", "channels":3, "image_height", "image_width"],
    ) -> TensorType["batch", "two_x_num_keypoints"]:
        """Forward pass through the network.

        Args:
            images: images

        Returns:
            heatmap per keypoint

        """
        representation = self.get_representations(images)
        out = self.final_layer(self.reshape_representation(representation))
        return out

    @typechecked
    def get_loss_inputs_labeled(self, batch_dict: BaseBatchDict) -> dict:
        """Return predicted coordinates."""

        representation = self.get_representations(batch_dict["images"])
        predicted_keypoints = self.final_layer(
            self.reshape_representation(representation)
        )

        return {
            "keypoints_targ": batch_dict["keypoints"],
            "keypoints_pred": predicted_keypoints,
        }


class SemiSupervisedRegressionTracker(SemiSupervisedTrackerMixin, RegressionTracker):
    """Model produces vectors of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        loss_factory_unsupervised: LossFactory,
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: bool = True,
        representation_dropout_rate: float = 0.2,
        last_resnet_layer_to_get: int = -2,
        torch_seed: int = 123,
    ) -> None:
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            pretrained: True to load pretrained imagenet weights
            representation_dropout_rate: dropout in the final fully connected
                layers
            last_resnet_layer_to_get: skip final layers of original model
            torch_seed: make weight initialization reproducible

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            resnet_version=resnet_version,
            pretrained=pretrained,
            representation_dropout_rate=representation_dropout_rate,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            torch_seed=torch_seed,
        )
        self.loss_factory_unsup = loss_factory_unsupervised
        loss_names = loss_factory_unsupervised.loss_instance_dict.keys()
        if "unimodal_mse" in loss_names or "unimodal_wasserstein" in loss_names:
            raise ValueError("cannot use unimodal loss in regression tracker")

        # this attribute will be modified by AnnealWeight callback during training
        self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))

    @typechecked
    def get_loss_inputs_unlabeled(self, batch: torch.Tensor) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""

        representation = self.get_representations(batch)
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )  # TODO: consider removing representation dropout?

        return {"keypoints_pred": predicted_keypoints}
