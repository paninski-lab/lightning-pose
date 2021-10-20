import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from typing import Any, Callable, Optional, Tuple, List, Dict
from torchtyping import TensorType, patch_typeguard
from typing_extensions import Literal
from typeguard import typechecked
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pose_est_nets.models.base_resnet import BaseFeatureExtractor
from pose_est_nets.losses.losses import (
    MaskedRMSELoss,
    MaskedRegressionMSELoss,
    get_losses_dict,
    convert_dict_entries_to_tensors
)

patch_typeguard()  # use before @typechecked


class RegressionTracker(BaseFeatureExtractor):
    def __init__(
        self,
        num_targets: int,  # TODO: decide whether targets or keypoints is the quantity of interest
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: Optional[bool] = True,
        representation_dropout_rate: Optional[float] = 0.2,
        last_resnet_layer_to_get: Optional[int] = -2,
    ) -> None:
        """
        Initializes regression tracker model with resnet backbone
        :param num_targets: number of body parts
        :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
            large the resnet will be.
        :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
            meaning the entire model will be trained unless this flag is provided
        """
        super().__init__(
            resnet_version=resnet_version,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
        )
        self.num_targets = num_targets
        self.resnet_version = resnet_version
        self.final_layer = nn.Linear(self.base.fc.in_features, self.num_targets)
        self.representation_dropout = nn.Dropout(
            p=representation_dropout_rate
        )  # TODO: consider removing dropout
        self.save_hyperparameters()

    @staticmethod
    @typechecked
    def reshape_representation(
        representation: TensorType[
            "Batch_Size",
            "Features",
            "Representation_Height",
            "Representation_Width",
            float,
        ]
    ) -> TensorType["Batch_Size", "Features", float]:
        return representation.reshape(representation.shape[0], representation.shape[1])

    @typechecked
    def forward(
        self,
        images: TensorType[
            "Batch_Size", "Image_Channels":3, "Image_Height", "Image_Width", float
        ],
    ) -> TensorType["Batch_Size", "Num_Targets"]:
        """
        Forward pass through the network
        :param x: input
        :return: output of network
        """
        representation = self.get_representations(images)
        out = self.final_layer(self.reshape_representation(representation))
        return out

    @typechecked
    def training_step(
        self,
        data_batch: list,
        batch_idx: int,
    ) -> Dict:
        images, keypoints = data_batch
        # forward pass
        representation = self.get_representations(images)
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )
        # compute loss
        loss = MaskedRegressionMSELoss(keypoints, predicted_keypoints)
        supervised_rmse = MaskedRMSELoss(keypoints, predicted_keypoints)
        # log training loss + rmse
        self.log("train_loss", loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)
        return {"loss": loss}

    @typechecked
    def evaluate(
        self, data_batch: list, stage: Optional[Literal["val", "test"]] = None
    ):
        images, keypoints = data_batch
        representation = self.get_representations(images)
        predicted_keypoints = self.final_layer(
            self.reshape_representation(representation)
        )
        loss = MaskedRegressionMSELoss(keypoints, predicted_keypoints)
        supervised_rmse = MaskedRMSELoss(keypoints, predicted_keypoints)

        # TODO: do we need other metrics?
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_supervised_rmse", supervised_rmse, prog_bar=True)

    def validation_step(self, validation_batch: list, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: list, batch_idx):
        self.evaluate(test_batch, "test")


class SemiSupervisedRegressionTracker(RegressionTracker):
    def __init__(
        self,
        num_targets: int,  # TODO: decide whether targets or keypoints is the quantity of interest
        loss_params: dict,
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: Optional[bool] = True,
        representation_dropout_rate: Optional[float] = 0.2,
        last_resnet_layer_to_get: Optional[int] = -2,
        semi_super_losses_to_use: Optional[list] = None,
    ) -> None:
        super().__init__(
            num_targets,
            resnet_version,
            pretrained,
            representation_dropout_rate,
            last_resnet_layer_to_get,
        )
        self.loss_function_dict = get_losses_dict(semi_super_losses_to_use)
        self.loss_params = convert_dict_entries_to_tensors(loss_params, self.device)

    @typechecked
    def training_step(self, data_batch: dict, batch_idx: int) -> dict:
        labeled_imgs, true_keypoints = data_batch["labeled"]
        unlabeled_imgs = data_batch["unlabeled"]
        representation = self.get_representations(labeled_imgs)
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )  # TODO: consider removing representation dropout?
        # compute loss
        supervised_loss = MaskedRegressionMSELoss(
            true_keypoints, predicted_keypoints
        )  # for training
        supervised_rmse = MaskedRMSELoss(
            true_keypoints, predicted_keypoints
        )  # for logging
        us_representation = self.get_representations(unlabeled_imgs)
        predicted_us_keypoints = self.final_layer(
            self.representation_dropout(
                self.reshape_representation(us_representation)
            )  # Do we need dropout
        )
        tot_loss = 0.0
        tot_loss += supervised_loss
        # loop over unsupervised losses
        for loss_name, loss_func in self.loss_function_dict.items():
            add_loss = self.loss_params[loss_name]["weight"] * loss_func(
                predicted_us_keypoints, **self.loss_params[loss_name]
            )
            tot_loss += add_loss
            # log individual unsupervised losses
            self.log(
                loss_name + "_loss",
                add_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        # log the total loss
        self.log(
            "total_loss",
            tot_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # log the supervised loss
        self.log(
            "supervised_loss",
            supervised_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "supervised_rmse",
            supervised_rmse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": tot_loss}
