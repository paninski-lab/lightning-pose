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
from pose_est_nets.losses.losses import MaskedRegressionMSELoss, get_losses_dict

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
        self.final_layer = nn.Linear(self.backbone.fc.in_features, self.num_targets)
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
        )  # TODO: consider removing representation dropout?
        # compute loss
        loss = MaskedRegressionMSELoss(keypoints, predicted_keypoints)
        # log training loss
        self.log("train_loss", loss)
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
        # TODO: do we need other metrics?
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, validation_batch: list, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: list, batch_idx):
        self.evaluate(test_batch, "test")

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=20, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


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
        self.loss_params = loss_params
        self.loss_function_dict = get_losses_dict(semi_super_losses_to_use)
        
    @typechecked
    def training_step(self, data_batch: dict, batch_idx: int) -> dict:
        labeled_imgs, true_keypoints = data_batch["labeled"]
        unlabeled_imgs = data_batch["unlabeled"]
        representation = self.get_representations(labeled_imgs)
        predicted_keypoints = self.final_layer(
            self.representation_dropout(self.reshape_representation(representation))
        )  # TODO: consider removing representation dropout?
        # compute loss
        supervised_loss = MaskedRegressionMSELoss(true_keypoints, predicted_keypoints)
        us_representation = self.get_representations(unlabeled_imgs)
        predicted_us_keypoints = self.final_layer(
            self.representation_dropout(
                self.reshape_representation(us_representation)
            )  # Do we need dropout
        )
        tot_loss = 0.0
        tot_loss += supervised_loss
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
        return {"loss": tot_loss}


# # that was the previous version that worked
# class RegressionTracker(LightningModule):
#     def __init__(
#         self,
#         num_targets: int,  # TODO: decide whether targets or keypoints is the quantity of interest
#         resnet_version: Optional[int] = 18,
#         transfer: Optional[bool] = False,
#         representation_dropout_rate: Optional[float] = 0.2,
#     ) -> None:
#         """
#         Initializes regression tracker model with resnet backbone
#         :param num_targets: number of body parts
#         :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
#             large the resnet will be.
#         :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
#             meaning the entire model will be trained unless this flag is provided
#         """
#         super(RegressionTracker, self).__init__()
#         self.__dict__.update(locals())  # todo: what is this?
#         self.resnet_version = resnet_version
#         self.num_targets = num_targets
#         self.backbone = grab_resnet_backbone(
#             resnet_version=self.resnet_version, pretrained=transfer
#         )
#         self.feature_extractor = grab_layers_sequential(
#             model=self.backbone, last_layer_ind=-2
#         )
#         self.final_layer = nn.Linear(self.backbone.fc.in_features, self.num_targets)
#         self.representation_dropout = nn.Dropout(
#             p=representation_dropout_rate
#         )  # TODO: consider removing

#     @staticmethod
#     @typechecked
#     def reshape_representation(
#         representation: TensorType["batch", "features", 1, 1]
#     ) -> TensorType["batch", "features"]:
#         return representation.reshape(representation.shape[0], representation.shape[1])

#     @typechecked
#     def forward(
#         self, x: TensorType["batch", 3, "height", "width"]
#     ) -> TensorType["batch", "num_targets"]:
#         """
#         Forward pass through the network
#         :param x: input
#         :return: output of network
#         """
#         with torch.no_grad():
#             representation = self.feature_extractor(x)
#             out = self.final_layer(self.reshape_representation(representation))
#         return out

#     @staticmethod
#     @typechecked
#     def regression_loss(
#         labels: TensorType["batch", "num_targets"],
#         preds: TensorType["batch", "num_targets"],
#     ) -> TensorType[()]:
#         """
#         Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
#         :param y: ground truth. shape=(batch, num_targets)
#         :param y_hat: prediction. shape=(batch, num_targets)
#         :return: mse loss
#         """
#         mask = labels == labels  # labels is not none, bool.
#         loss = F.mse_loss(
#             torch.masked_select(labels, mask), torch.masked_select(preds, mask)
#         )

#         return loss

#     def training_step(self, data, batch_idx):
#         x, y = data
#         # forward pass
#         representation = self.feature_extractor(x)
#         y_hat = self.final_layer(
#             self.representation_dropout(self.reshape_representation(representation))
#         )  # TODO: consider removing representation dropout?
#         # compute loss
#         loss = self.regression_loss(y, y_hat)
#         # log training loss
#         self.log(
#             "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
#         )
#         return {"loss": loss}

#     def validation_step(self, data, batch_idx):
#         x, y = data
#         # forward pass
#         representation = self.feature_extractor(x)
#         y_hat = self.final_layer(self.reshape_representation(representation))
#         # compute loss
#         loss = self.regression_loss(y, y_hat)
#         # log validation loss
#         self.log("val_loss", loss, prog_bar=True, logger=True)

#     def test_step(self, data, batch_idx):
#         self.validation_step(data, batch_idx)

#     def configure_optimizers(self):
#         return Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
