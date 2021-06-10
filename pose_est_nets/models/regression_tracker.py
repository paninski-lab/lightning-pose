import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from typing import Any, Callable, Optional, Tuple, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

def grab_resnet_backbone(resnet_version: Optional[int] = 18,
                         pretrained: Optional[bool] = True) -> models.resnet.ResNet:
    resnets = {
        18: models.resnet18, 34: models.resnet34,
        50: models.resnet50, 101: models.resnet101,
        152: models.resnet152
    }
    return resnets[resnet_version](pretrained)


def grab_layers_sequential(model: models.resnet.ResNet,
                           last_layer_ind: Optional[int] = None) -> torch.nn.modules.container.Sequential:
    layers = list(model.children())[:last_layer_ind + 1]
    return nn.Sequential(*layers)


# TODO: verify that the forward pass makes sense, add callback for freeze unfreeze.
class RegressionTracker(LightningModule):
    def __init__(self,
                 num_targets: int,
                 resnet_version: int = 18,
                 transfer: Optional[bool] = False
                 ) -> None:
        """
        Initializes regression tracker model with resnet backbone
        :param num_targets: number of body parts
        :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
            large the resnet will be.
        :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
            meaning the entire model will be trained unless this flag is provided
        """
        super(RegressionTracker, self).__init__()
        self.__dict__.update(locals())  # todo: what is this?
        self.resnet_version = resnet_version
        self.num_targets = num_targets
        self.backbone = grab_resnet_backbone(resnet_version=self.resnet_version,
                                             pretrained=transfer)
        # num_filters = backbone.fc.in_features  # number of inputs to final linear layer
        # layers = list(backbone.children())[:-1]  # keeping all layers but the last
        # self.feature_extractor = nn.Sequential(*layers)
        # self.final_layer = nn.Linear(num_filters, num_targets)

    @property
    def feature_extractor(self):
        return grab_layers_sequential(model=self.backbone, last_layer_ind=-2)

    @property
    def final_layer(self):
        return nn.ModuleList(nn.Linear(self.backbone.fc.in_features, self.num_targets)) # to be able to send to cuda

    @staticmethod
    @typechecked
    def reshape_representation(representation: TensorType["batch", "features", 1, 1]) -> TensorType[
        "batch", "features"]:
        return representation.reshape(representation.shape[0], representation.shape[1])

    @typechecked
    def forward(self,
                x: TensorType["batch", 3, "height", "width"]
                ) -> TensorType["batch", "num_targets"]:
        """
        Forward pass through the network
        :param x: input
        :return: output of network
        """
        with torch.no_grad():
            representation = self.feature_extractor(x)
            out = self.final_layer(self.reshape_representation(representation))
        return out

    @staticmethod
    @typechecked
    def regression_loss(labels: TensorType["batch", "num_targets"],
                        preds: TensorType["batch", "num_targets"]
                        ) -> TensorType[()]:
        """
        Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
        :param y: ground truth. shape=(batch, num_targets)
        :param y_hat: prediction. shape=(batch, num_targets)
        :return: mse loss
        """
        mask = labels == labels  # labels is not none, bool.
        loss = F.mse_loss(torch.masked_select(labels, mask),
                          torch.masked_select(preds, mask))

        return loss

    def training_step(self, data, batch_idx):
        x, y = data
        # forward pass
        representation = self.feature_extractor(x)
        y_hat = self.final_layer(self.reshape_representation(representation))
        # compute loss
        loss = self.regression_loss(y, y_hat)
        # log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, data, batch_idx):
        x, y = data
        # forward pass
        representation = self.feature_extractor(x)
        y_hat = self.final_layer(self.reshape_representation(representation))
        # compute loss
        loss = self.regression_loss(y, y_hat)
        # log validation loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        self.validation_step(data, batch_idx)

    def configure_optimizers(self):
        return Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
