import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from typing import Any, Callable, Optional, Tuple, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch.optim.lr_scheduler import ReduceLROnPlateau


patch_typeguard()  # use before @typechecked


@typechecked
def grab_resnet_backbone(
    resnet_version: Optional[int] = 18, pretrained: Optional[bool] = True
) -> models.resnet.ResNet:
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    return resnets[resnet_version](pretrained)


@typechecked
def grab_layers_sequential(
    model: models.resnet.ResNet, last_layer_ind: Optional[int] = None
) -> torch.nn.modules.container.Sequential:
    layers = list(model.children())[: last_layer_ind + 1]
    return nn.Sequential(*layers)


class BaseFeatureExtractor(LightningModule):
    def __init__(
        self,
        resnet_version: Optional[int] = 18,
        pretrained: Optional[bool] = False,
        last_resnet_layer_to_get: Optional[int] = -2,
    ) -> None:
        super(BaseFeatureExtractor, self).__init__()
        self.__dict__.update(locals())  # TODO: what is this?
        self.resnet_version = resnet_version
        self.backbone = grab_resnet_backbone(
            resnet_version=self.resnet_version, pretrained=pretrained
        )
        self.feature_extractor = grab_layers_sequential(
            model=self.backbone,
            last_layer_ind=last_resnet_layer_to_get,
        )

    @typechecked
    def forward(
        self, x: TensorType["batch", 3, "height", "width"]
    ) -> TensorType["batch", "features", 1, 1]:
        # TODO: the [1,1] shape assertion depends on when we truncate the ResNet
        """
        Forward pass from images to representations
        :param x: image
        :return: representation
        """
        with torch.no_grad():
            return self.feature_extractor(x)
