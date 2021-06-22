import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from typing import Any, Callable, Optional, Tuple, List
from torchtyping import TensorType, patch_typeguard
import numpy as np
#from tensorflow.keras.applications.resnet50 import preprocess_input #might want to change this
#from deepposekit.models.layers.convolutional import SubPixelUpscaling


class HeatmapTracker(LightningModule):
    def __init__(
        self,
        num_targets: int,
        resnet_version: int = 18,
        transfer: Optional[bool] = False,
    ) -> None:
        """
        Initializes regression tracker model with resnet backbone
        :param num_targets: number of body parts
        :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
            large the resnet will be.
        :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
            meaning the entire model will be trained unless this flag is provided
        """
        super(HeatmapTracker, self).__init__()
        self.__dict__.update(locals())  # todo: what is this?
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        # Using a pretrained ResNet backbone
        backbone = resnets[resnet_version](pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.upsampling_layers = []
        # TODO: See if a big image ends up with different spatial resolution
        in_dim = num_filters // 16
        out_dim = 64

        # TODO: Add normalization
        # TODO: Should depend on input size
        for i in range(8):
            self.upsampling_layers += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_dim),
                nn.Upsample(scale_factor=2, mode="bilinear"),
            ]
            in_dim = out_dim

        # TODO: Move sigmoid after final upsampling
        self.upsampling_layers += [
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_dim, num_targets, kernel_size=3, stride=1, padding=1),
        ]
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)
        self.sigmoid = nn.Sigmoid()
        self.batch_size = 16
        self.num_workers = 0

    # TODO: Separate from training step
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass through the network
        :param x: input
        :return: output of network
        """
        #self.feature_extractor.eval()
        #with torch.no_grad():
        representations = torch.reshape(
            self.feature_extractor(x), (x.shape[0], -1, 4, 4)
        )
        out = self.upsampling_layers(representations)
        #upsample_final = nn.Upsample(size=(x.shape[-2], x.shape[-1]), mode="bilinear")
        out = self.sigmoid(upsample_final(out))
        return out

    @staticmethod
    def heatmap_loss(y: torch.tensor, y_hat: torch.tensor) -> torch.tensor:
        """
        Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
        :param y: ground truth. shape=(num_targets, 2)
        :param y_hat: prediction. shape=(num_targets, 2)
        :return: mse loss
        """
        # apply mask
        # compute loss
        # loss = F.mse_loss(y_hat, y)
        loss = F.binary_cross_entropy(y_hat, y)

        return loss

    def training_step(self, data, batch_idx):
        x, y = data
        # forward pass
        y_hat = self.forward(x)
        # compute loss
        loss = self.heatmap_loss(y, y_hat)
        # log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, data, batch_idx):
        x, y = data
        y_hat = self.forward(x)
        # compute loss
        loss = self.heatmap_loss(y, y_hat)
        # log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        self.validation_step(data, batch_idx)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)



class DLC(LightningModule):
    def __init__(
        self,
        num_targets: int,
        resnet_version: int = 18,
        transfer: Optional[bool] = False,
    ) -> None:
        """
        Initializes regression tracker model with resnet backbone
        :param num_targets: number of body parts
        :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
            large the resnet will be.
        :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
            meaning the entire model will be trained unless this flag is provided
        """
        super(DLC, self).__init__()
        self.__dict__.update(locals())  # todo: what is this?
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        # Using a pretrained ResNet backbone
        backbone = resnets[resnet_version](pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.upsampling_layers = []
        # TODO: Add normalization
        # TODO: Should depend on input size
        self.num_keypoints = 17
        padding_dims = compute_same_padding((384/(2**2), 384/(2**2)))
        self.upsampling_layers += [
            nn.PixelShuffle(2),
            F.pad(padding_dims), #aims to mimic "same" padding from tensorflow
            nn.ConvTranspose2d(in_channels = num_filters, out_channels = self.num_keypoints, kernel_size = (3, 3), stride = (2,2))
        ]
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)
        self.batch_size = 16
        self.num_workers = 0

     def forward(self, x: TensorType["batch", 3, "Height", "Width"]) -> TensorType["batch", self.num_keypoints, "Out_Height", "Out_Width"]:
        """
        Forward pass through the network
        :param x: input
        :return: output of network
        """
        #self.feature_extractor.eval()
        #with torch.no_grad():
        #x = ImageNetPreprocess(network="resnet50", mode="torch")(x) #hopefully works
        representations = self.feature_extractor(x)
        out = self.upsampling_layers(representations)
        return out

    @staticmethod
    def heatmap_loss(y: TensorType["batch", self.num_keypoints, "Out_Height", "Out_Width"], y_hat: TensorType["batch", self.num_keypoints, "Out_Height", "Out_Width"]) -> TensorType[()]:
        """
        Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
        :param y: ground truth. shape=(num_targets, 2)
        :param y_hat: prediction. shape=(num_targets, 2)
        :return: mse loss
        """
        # apply mask
        # compute loss
        loss = F.mse_loss(y_hat, y)
        #loss = F.binary_cross_entropy(y_hat, y)

        return loss

    def training_step(self, data, batch_idx):
        x, y = data
        # forward pass
        y_hat = self.forward(x)
        # compute loss
        loss = self.heatmap_loss(y, y_hat)
        # log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, data, batch_idx):
        x, y = data
        y_hat = self.forward(x)
        # compute loss
        loss = self.heatmap_loss(y, y_hat)
        # log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        self.validation_step(data, batch_idx)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


def compute_same_padding(img_dims, kernel_dims, stride):
    in_height, in_width = img_dims
    filter_height, filter_width = kernel_dims
    strides=stride
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width  = np.ceil(float(in_width) / float(strides[2]))

    #The total padding applied along the height and width is computed as:

    if (in_height % strides[1] == 0):
      pad_along_height = max(filter_height - strides[1], 0)
    else:
      pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if (in_width % strides[2] == 0):
      pad_along_width = max(filter_width - strides[2], 0)
    else:
      pad_along_width = max(filter_width - (in_width % strides[2]), 0)

    #print(pad_along_height, pad_along_width)
      
    #Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)
