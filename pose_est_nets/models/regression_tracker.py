"""Models that produce (x, y) coordinates of keypoints from images."""
# TODO: support weight tuning here too
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
from typing_extensions import Literal

from pose_est_nets.losses.factory import LossFactory
from pose_est_nets.losses.losses import RegressionRMSELoss
from pose_est_nets.models.base_resnet import BaseFeatureExtractor

patch_typeguard()  # use before @typechecked


class BaseBatchDict(TypedDict):
    images: TensorType["batch", "RGB":3, "image_height", "image_width", float]
    keypoints: TensorType["batch", "num_targets", float]
    idxs: TensorType["batch", int]


class SemiSupervisedBatchDict(TypedDict):
    labeled: BaseBatchDict
    unlabeled: TensorType[
        "sequence_length", "RGB":3, "image_height", "image_width", float
    ]


class RegressionTracker(BaseFeatureExtractor):
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

        super().__init__(  # execute BaseFeatureExtractor.__init__()
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
        self.save_hyperparameters()

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
    def training_step(self, data_batch: BaseBatchDict, batch_idx: int) -> dict:

        # forward pass
        representation = self.get_representations(data_batch["images"])
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )

        # compute and log loss
        loss = self.loss_factory(
            keypoints_true=data_batch["keypoints"],
            keypoints_pred=predicted_keypoints,
            stage="train",  # for logging purposes
        )
        self.log("train_loss", loss, prog_bar=True)

        # for additional info: compute and log supervised rmse
        rmse_loss = RegressionRMSELoss()
        supervised_rmse = rmse_loss(
            keypoints_true=data_batch["keypoints"],
            keypoints_pred=predicted_keypoints,
            logging=False,
        )
        self.log("train_rmse_supervised", supervised_rmse, prog_bar=True)

        return {"loss": loss}

    @typechecked
    def evaluate(
        self,
        data_batch: BaseBatchDict,
        stage: Optional[Literal["val", "test"]] = None,
    ) -> None:

        # forward_pass
        representation = self.get_representations(data_batch["images"])
        predicted_keypoints = self.final_layer(
            self.reshape_representation(representation)
        )

        # compute loss
        loss = self.loss_factory(
            keypoints_targ=data_batch["keypoints"],
            keypoints_pred=predicted_keypoints,
            stage=stage,  # for logging purposes
        )

        # log loss + rmse
        if stage:
            rmse_loss = RegressionRMSELoss()
            supervised_rmse = rmse_loss(
                keypoints_targ=data_batch["keypoints"],
                keypoints_pred=predicted_keypoints,
                logging=False,
            )
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_supervised_rmse", supervised_rmse, prog_bar=True)

    def validation_step(self, validation_batch: BaseBatchDict, batch_idx: int):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: BaseBatchDict, batch_idx: int):
        self.evaluate(test_batch, "test")


class SemiSupervisedRegressionTracker(RegressionTracker):
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
        loss_names = loss_factory_unsupervised.loss_classes_dict.keys()
        if "unimodal_mse" in loss_names or "unimodal_wasserstein" in loss_names:
            raise ValueError("cannot use unimodal loss in regression tracker")

        # this attribute will be modified by AnnealWeight callback during training
        self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))

    @typechecked
    def training_step(
        self, data_batch: SemiSupervisedBatchDict, batch_idx: int
    ) -> dict:

        # on each epoch, self.total_unsupervised_importance is modified by the
        # AnnealWeight callback
        self.log(
            "total_unsupervised_importance",
            self.total_unsupervised_importance,
            prog_bar=True,
        )

        # forward pass labeled
        # --------------------
        representation = self.get_representations(data_batch["labeled"]["images"])
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )  # TODO: consider removing representation dropout?

        # compute and log supervised loss
        loss_super = self.loss_factory(
            keypoints_targ=data_batch["labeled"]["keypoints"],
            keypoints_pred=predicted_keypoints,
            stage="train",  # for logging purposes
        )
        self.log("train_loss_supervised", loss_super, prog_bar=True)

        # for additional info: compute and log supervised rmse
        rmse_loss = RegressionRMSELoss()
        supervised_rmse = rmse_loss(
            keypoints_targ=data_batch["labeled"]["keypoints"],
            keypoints_pred=predicted_keypoints,
            logging=False,
        )
        self.log("train_rmse_supervised", supervised_rmse, prog_bar=True)

        # forward pass unlabeled
        # ----------------------
        representation_ul = self.get_representations(data_batch["unlabeled"])
        predicted_keypoints_ul = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation_ul))
        )  # TODO: consider removing representation dropout?

        # compute and log unsupervised loss
        loss_unsuper = self.loss_factory_unsup(
            keypoints_pred=predicted_keypoints_ul,
            anneal_weight=self.total_unsupervised_importance,
            stage="train",  # for logging purposes
        )

        # log total loss
        total_loss = loss_super + loss_unsuper
        self.log("total_loss", total_loss, prog_bar=True)

        return {"loss": total_loss}
