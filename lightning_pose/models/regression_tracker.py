"""Models that produce (x, y) coordinates of keypoints from images."""

from omegaconf import DictConfig
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal

from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    BaseBatchDict,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)

patch_typeguard()  # use before @typechecked


@typechecked
class RegressionTracker(BaseSupervisedTracker):
    """Base model that produces (x, y) predictions of keypoints from images."""

    @typechecked
    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnet50_3d", "resnet50_contrastive",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"] = "resnet50",
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
        representation_dropout_rate: float = 0.2,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = True,
    ) -> None:
        """Base model that produces (x, y) coordinates of keypoints from images.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of backbone model
            representation_dropout_rate: dropout in the final fully connected
                layers
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.num_keypoints = num_keypoints
        self.num_targets = self.num_keypoints * 2
        self.loss_factory = loss_factory
        self.final_layer = nn.Linear(self.num_fc_input_features, self.num_targets)
        # TODO: consider removing dropout
        self.representation_dropout = nn.Dropout(p=representation_dropout_rate)
        self.torch_seed = torch_seed
        self.do_context = do_context
        if self.mode == "2d":
            self.representation_fc = torch.nn.Linear(5, 1, bias=False)
        elif self.mode == "3d":
            self.representation_fc = torch.nn.Linear(8, 1, bias=False)

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
        images: Union[
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"]]
        ) -> TensorType["batch", "two_x_num_keypoints"]:
        """Forward pass through the network."""
        representations = self.get_representations(images)
        if self.do_context:
            # output of line below is of shape (batch, features, height, width, 1)
            representations = self.representation_fc(representations)
            # output of line below is of shape (batch, features, height, width)
            representations = torch.squeeze(representations, 4)
        out = self.final_layer(self.reshape_representation(representations))
        return out

    @typechecked
    def get_loss_inputs_labeled(self, batch_dict: BaseBatchDict) -> dict:
        """Return predicted coordinates for a batch of data."""
        representation = self.get_representations(batch_dict["images"])
        predicted_keypoints = self.final_layer(
            self.reshape_representation(representation)
        )
        return {
            "keypoints_targ": batch_dict["keypoints"],
            "keypoints_pred": predicted_keypoints,
        }


@typechecked
class SemiSupervisedRegressionTracker(SemiSupervisedTrackerMixin, RegressionTracker):
    """Model produces vectors of keypoints from labeled/unlabeled images."""

    @typechecked
    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        loss_factory_unsupervised: LossFactory,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnet50_3d", "resnet50_contrastive",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"] = "resnet50",
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
        representation_dropout_rate: float = 0.2,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
    ) -> None:
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of original model
            representation_dropout_rate: dropout in the final fully connected
                layers
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            pretrained=pretrained,
            representation_dropout_rate=representation_dropout_rate,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.loss_factory_unsup = loss_factory_unsupervised
        loss_names = loss_factory_unsupervised.loss_instance_dict.keys()
        if "unimodal_mse" in loss_names or "unimodal_wasserstein" in loss_names:
            raise ValueError("cannot use unimodal loss in regression tracker")

        # this attribute will be modified by AnnealWeight callback during training
        self.total_unsupervised_importance = torch.tensor(1.0)
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))

    @typechecked
    def get_loss_inputs_unlabeled(self, batch: Union[TensorType[
        "sequence_length", "RGB":3, "image_height", "image_width", float
        ], TensorType[
            "sequence_length", "context":5, "RGB":3, "image_height", "image_width", float
    ]]) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        representation = self.get_representations(batch)
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )  # TODO: consider removing representation dropout?
        return {"keypoints_pred": predicted_keypoints}
