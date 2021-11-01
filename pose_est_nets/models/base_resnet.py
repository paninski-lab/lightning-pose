"""Base class for resnet backbone that acts as a feature extractor."""

from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torchtyping import TensorType, patch_typeguard
import torchvision.models as models
from typeguard import typechecked
from typing import Any, Callable, List, Optional, Tuple
from typing_extensions import Literal


patch_typeguard()  # use before @typechecked


@typechecked
def grab_resnet_backbone(
    resnet_version: Literal[18, 34, 50, 101, 152] = 18,
    pretrained: bool = True,
) -> models.resnet.ResNet:
    """Load resnet architecture from torchvision.

    Args:
        resnet_version: choose network depth
        pretrained: True to load weights pretrained on imagenet

    Returns:
        selected resnet architecture as a model object

    """
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
    """Package selected number of layers into a nn.Sequential object.

    Args:
        model: original resnet model
        last_layer_ind: final layer to pass data through

    Returns:
        potentially reduced backbone model

    """
    layers = list(model.children())[: last_layer_ind + 1]
    return nn.Sequential(*layers)


class BaseFeatureExtractor(LightningModule):
    """Object that contains the base resnet feature extractor."""

    def __init__(
        self,
        resnet_version: Literal[18, 34, 50, 101, 152] = 18,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
    ) -> None:
        """A ResNet model that takes in images and generates features.

        ResNets will be loaded from torchvision and can be either pre-trained
        on ImageNet or randomly initialized. These were originally used for
        classification tasks, so we truncate their final fully connected layer.

        Args:
            resnet_version: which ResNet version to use; defaults to 18
            pretrained: True to load weights pretrained on imagenet
            last_resnet_layer_to_get: Defaults to -2.

        """
        super().__init__()
        print("\n Initializing a {} instance.".format(self._get_name()))

        self.resnet_version = resnet_version
        self.base = grab_resnet_backbone(
            resnet_version=self.resnet_version, pretrained=pretrained
        )
        self.backbone = grab_layers_sequential(
            model=self.base,
            last_layer_ind=last_resnet_layer_to_get,
        )

    def get_representations(
        self,
        images: TensorType["batch", "channels":3, "image_height", "image_width", float],
    ) -> TensorType["batch", "features", "rep_height", "rep_width", float]:
        """Forward pass from images to feature maps.

        Wrapper around the backbone's feature_extractor() method for
        typechecking purposes.
        See tests/test_base_resnet.py for example shapes.

        Args:
            images: a batch of images

        Returns:
            a representation of the images; features differ as a function of
            resnet version. Representation height and width differ as a
            function of image dimensions, and are not necessarily equal.
        """
        return self.backbone(images)

    def forward(self, images):
        """Forward pass from images to representations.

        Wrapper around self.get_representations().
        Fancier childern models will use get_representations() in their forward
        methods.

        Args:
            images (torch.tensor(float)): a batch of images.

        Returns:
            torch.tensor(float): a representation of the images.
        """
        return self.get_representations(images)

    def configure_optimizers(self):
        """Select optimizer, lr scheduler, and metric for monitoring."""

        # standard adam optimizer
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)

        # define a scheduler that reduces the base learning rate
        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        #     factor=0.2,
        #     patience=20,
        #     verbose=True
        # )
        # scheduler = StepLR(
        #     optimizer,
        #     step_size=50,
        #     gamma=0.5
        # )
        scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
