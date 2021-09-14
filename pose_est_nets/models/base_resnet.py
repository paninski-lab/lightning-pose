import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from typing import Any, Callable, Optional, Tuple, List
from typing_extensions import Literal
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch.optim.lr_scheduler import ReduceLROnPlateau


patch_typeguard()  # use before @typechecked


@typechecked
def grab_resnet_backbone(
    resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
    pretrained: Optional[bool] = True,
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
def get_resnet_features(resnet_version: Literal[18, 34, 50, 101, 152]) -> int:
    if resnet_version < 50:
        _num_features_in_representation = 512
    else:
        _num_features_in_representation = 2048
    return _num_features_in_representation


@typechecked
def grab_layers_sequential(
    model: models.resnet.ResNet, last_layer_ind: Optional[int] = None
) -> torch.nn.modules.container.Sequential:
    layers = list(model.children())[: last_layer_ind + 1]
    return nn.Sequential(*layers)


class BaseFeatureExtractor(LightningModule):
    def __init__(
        self,
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        pretrained: Optional[bool] = False,
        last_resnet_layer_to_get: Optional[int] = -2,
    ) -> None:
        """A ResNet model that takes in images and generates features.
        ResNets will be loaded from torchvision and can be either pre-trained on ImageNet or randomly initialized.
        These were originally used for classification tasks, so we truncate their final fully connected layer.

        Args:
            resnet_version (Optional[int], optional): Which ResNet version to use. Defaults to 18.
            pretrained (Optional[bool], optional): [description]. Defaults to False.
            last_resnet_layer_to_get (Optional[int], optional): [description]. Defaults to -2.
        """
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

    def get_representations(
        self,
        images: TensorType[
            "Batch_Size", "Image_Channels":3, "Image_Height", "Image_Width", float
        ],
    ) -> TensorType[
        "Batch_Size",
        "Features",
        "Representation_Height",
        "Representation_Width",
        float,
    ]:
        """a wrapper around self.feature_extractor for typechecking purposes. see tests/test_base_resnet.py for example shapes.

        Args:
            images (torch.tensor(float)): a batch of images

        Returns:
            torch.tensor(float): a representation of the images. Features differ as a function of resnet version. Representation height and width differ as a function of image dimensions, and are not necessarily equal.
        """
        return self.feature_extractor(images)

    def forward(self, images):
        """Forward pass from images to representations. Just a wrapper over get_representations.
        Fancier childern models will use get_representations in their forward methods.

        Args:
            images (torch.tensor(float)): a batch of images.

        Returns:
            torch.tensor(float): a representation of the images.
        """
        return self.get_representations(images)
