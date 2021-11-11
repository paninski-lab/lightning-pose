"""Models that produce (x, y) coordinates of keypoints from images."""

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
from typing_extensions import Literal

from pose_est_nets.losses.losses import (
    convert_dict_entries_to_tensors,
    get_losses_dict,
    MaskedRegressionMSELoss,
    MaskedRMSELoss,
)
from pose_est_nets.models.base_resnet import BaseFeatureExtractor

patch_typeguard()  # use before @typechecked


class BaseBatchDict(TypedDict):
    images: TensorType["batch", "RGB":3, "image_height", "image_width"]
    keypoints: TensorType["batch", "num_targets"]
    idxs: TensorType["batch"]


# TODO: Add the semisuper case


class RegressionTracker(BaseFeatureExtractor):
    """Base model that produces (x, y) predictions of keypoints from images."""

    def __init__(
        self,
        num_targets: int,  # TODO: decide whether targets or keypoints is the quantity of interest
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: bool = True,
        representation_dropout_rate: float = 0.2,
        last_resnet_layer_to_get: int = -2,
        torch_seed: int = 123,
    ) -> None:
        """Base model that produces (x, y) coordinates of keypoints from images.

        Args:
            num_targets: number of body parts times 2 (x,y) coords
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
        self.num_targets = num_targets
        self.resnet_version = resnet_version
        self.final_layer = nn.Linear(self.base.fc.in_features, self.num_targets)
        # TODO: consider removing dropout
        self.representation_dropout = nn.Dropout(p=representation_dropout_rate)
        self.torch_seed = torch_seed
        self.save_hyperparameters()

    @property
    def num_keypoints(self):
        return self.num_targets // 2

    @staticmethod
    @typechecked
    def reshape_representation(
        representation: TensorType[
            "batch",
            "features",
            "rep_height",
            "rep_width",
            float,
        ]
    ) -> TensorType["batch", "features", float]:
        return representation.reshape(representation.shape[0], representation.shape[1])

    @typechecked
    def forward(
        self,
        images: TensorType["batch", "channels":3, "image_height", "image_width", float],
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
    def training_step(self, data_batch: Dict, batch_idx: int) -> Dict:

        # forward pass
        representation = self.get_representations(data_batch["images"])
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )

        # compute loss
        loss = MaskedRegressionMSELoss(data_batch["keypoints"], predicted_keypoints)
        supervised_rmse = MaskedRMSELoss(data_batch["keypoints"], predicted_keypoints)

        # log training loss + rmse
        self.log("train_loss", loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)

        return {"loss": loss}

    @typechecked
    def evaluate(
        self,
        data_batch: Dict,
        stage: Optional[Literal["val", "test"]] = None,
    ) -> None:
        representation = self.get_representations(data_batch["images"])
        predicted_keypoints = self.final_layer(
            self.reshape_representation(representation)
        )
        loss = MaskedRegressionMSELoss(data_batch["keypoints"], predicted_keypoints)
        supervised_rmse = MaskedRMSELoss(data_batch["keypoints"], predicted_keypoints)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_supervised_rmse", supervised_rmse, prog_bar=True)

    def validation_step(self, validation_batch: Dict, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: Dict, batch_idx):
        self.evaluate(test_batch, "test")


class SemiSupervisedRegressionTracker(RegressionTracker):
    """Model produces vectors of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_targets: int,  # TODO: decide whether targets or keypoints is the quantity of interest
        loss_params: dict,
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: bool = True,
        representation_dropout_rate: float = 0.2,
        last_resnet_layer_to_get: int = -2,
        torch_seed: int = 123,
        semi_super_losses_to_use: Optional[list] = None,
    ) -> None:
        """

        Args:
            num_targets: number of body parts times 2 (x,y) coords
            loss_params:
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            pretrained: True to load pretrained imagenet weights
            representation_dropout_rate: dropout in the final fully connected
                layers
            last_resnet_layer_to_get: skip final layers of original model
            torch_seed: make weight initialization reproducible
            semi_super_losses_to_use: TODO

        """
        super().__init__(
            num_targets=num_targets,
            resnet_version=resnet_version,
            pretrained=pretrained,
            representation_dropout_rate=representation_dropout_rate,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            torch_seed=torch_seed,
        )
        print(semi_super_losses_to_use)
        if semi_super_losses_to_use is None:
            raise ValueError("must specify losses for unlabeled frames")
        if "unimodal" in semi_super_losses_to_use:
            raise ValueError("cannot use unimodal loss in regression tracker")
        self.loss_function_dict = get_losses_dict(semi_super_losses_to_use)
        self.loss_params = loss_params

    @typechecked
    def training_step(self, data_batch: dict, batch_idx: int) -> dict:

        # forward pass labeled
        representation = self.get_representations(data_batch["labeled"]["images"])
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )  # TODO: consider removing representation dropout?

        # compute loss labeled
        supervised_loss = MaskedRegressionMSELoss(
            data_batch["labeled"]["keypoints"], predicted_keypoints
        )  # for training
        supervised_rmse = MaskedRMSELoss(
            data_batch["labeled"]["keypoints"], predicted_keypoints
        )  # for logging

        # forward pass unlabeled
        us_representation = self.get_representations(data_batch["unlabeled"])
        predicted_us_keypoints = self.final_layer(
            self.representation_dropout(
                self.reshape_representation(us_representation)
            )  # Do we need dropout
        )

        # loop over unsupervised losses
        tot_loss = 0.0
        tot_loss += supervised_loss
        self.loss_params = convert_dict_entries_to_tensors(
            self.loss_params, self.device
        )
        for loss_name, loss_func in self.loss_function_dict.items():
            add_loss = self.loss_params[loss_name]["weight"] * loss_func(
                predicted_us_keypoints, **self.loss_params[loss_name]
            )
            tot_loss += add_loss
            # log individual unsupervised losses
            self.log(loss_name + "_loss", add_loss, prog_bar=True)

        # log other losses
        self.log("total_loss", tot_loss, prog_bar=True)
        self.log("supervised_loss", supervised_loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)

        return {"loss": tot_loss}
