from pose_est_nets.models.base_resnet import BaseFeatureExtractor
import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Callable, Optional, Tuple, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import numpy as np
from pose_est_nets.utils.heatmap_tracker_utils import (
    find_subpixel_maxima,
    largest_factor,
    format_mouse_data,
)
from pose_est_nets.losses.heatmap_loss import MaskedMSEHeatmapLoss

patch_typeguard()  # use before @typechecked


class HeatmapTracker(BaseFeatureExtractor):
    def __init__(
        self,
        num_targets: int,
        resnet_version: Optional[int] = 18,
        downsample_factor: Optional[
            int
        ] = 2,  # TODO: downsample_factor may be in mismatch between datamodule and model
        pretrained: Optional[bool] = False,
        last_resnet_layer_to_get: Optional[int] = -3,
    ) -> None:
        """
        TODO: edit this. note, last_resnet_layer_to_get is different from the regression net, on purpose.
        Initializes a DLC-like model with resnet backbone inherited from BaseFeatureExtractor
        :param num_targets: number of body parts times 2 (x,y) coords
        :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
            large the resnet will be.
        :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
            meaning the entire model will be trained unless this flag is provided
        """
        super().__init__(  # execute BaseFeatureExtractor.__init__()
            resnet_version=resnet_version,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
        )
        self.__dict__.update(locals())  # TODO: what is this?
        print("running this in init heatmap tracker")

        self.num_filters_for_upsampling = self.backbone.fc.in_features
        self.num_targets = num_targets
        self.num_keypoints = num_targets // 2
        self.downsample_factor = downsample_factor
        self.coordinate_scale = torch.tensor(2 ** downsample_factor, device=self.device)
        self.upsampling_layers = self.make_upsampling_layers()
        self.initialize_upsampling_layers()

        # TODO: review with Nick. Do we agree on dims? how to do this?
        global _num_features_in_representation
        if resnet_version < 50:
            _num_features_in_representation = 512
        else:
            _num_features_in_representation = 2048
        global _num_keypoints
        _num_keypoints = np.copy(self.num_keypoints)
        # TODO: check that the 5 thing is general? not sure, it's only right for 384X384?
        # TODO: this avoids the need for outputshape? we don't need to know image size in the current setup.
        global _heatmap_dims
        _heatmap_dims = 12 * (2 ** (5 - downsample_factor))
        # I'm up to this line. the below is olde
        # self.batch_size = 16 #for autoscale batchsize
        # self.num_workers = 0

    def initialize_upsampling_layers(self):
        # TODO: test that running this method changes the weights and biases
        """loop over the Conv2DTranspose layers and initialize them"""
        for index, layer in enumerate(self.upsampling_layers):
            if index > 0:  # we ignore the PixelShuffle
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    # TODO: Should return nn.sequential!
    @typechecked
    def make_upsampling_layers(self) -> list:
        """input shape = [batch, resnet_version_filters, 12, 12]"""
        upsampling_layers = [nn.PixelShuffle(2)]
        upsampling_layers += nn.ConvTranspose2d(
            in_channels=self.num_filters_for_upsampling // 4,
            out_channels=self.num_keypoints,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )  # [batch, self.num_keypoints, 48, 48]. stopping here if downsample_factor=3

        if self.downsample_factor == 2:  # make the heatmaps bigger
            upsampling_layers += nn.ConvTranspose2d(
                in_channels=self.num_keypoints,
                out_channels=self.num_keypoints,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            )  # [batch, self.num_keypoints, 96, 96]

        return nn.Sequential(*upsampling_layers)

    @staticmethod
    @typechecked
    def create_double_upsampling_layer(
        in_channels: int, out_channels: int
    ) -> torch.nn.ConvTranspose2d:
        """taking in/out channels, and performs ConvTranspose2d to double the output shape"""
        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )

    @typechecked
    def heatmaps_from_representation(
        self,
        representations: TensorType[
            "Batch_Size",
            "Features":_num_features_in_representation,
            "Representation_Height":12,
            "Representation_Width":12,
        ],
    ) -> TensorType[
        "Batch_Size",
        "Num_Keypoints":_num_keypoints,
        "Heatmap_Height":_heatmap_dims,
        "Heatmap_Width":_heatmap_dims,
    ]:
        return self.upsampling_layers(representations)

    @typechecked
    def forward(
        self, images: TensorType["Batch_Size", 3, "Image_Height", "Image_Width"]
    ) -> TensorType[
        "Batch_Size",
        "Num_Keypoints":_num_keypoints,
        "Heatmap_Height":_heatmap_dims,
        "Heatmap_Width":_heatmap_dims,
    ]:
        """
        Forward pass through the network
        :param x: images
        :return: heatmap per keypoint
        """
        """TODO: I have good assertions in the old heatmap_tracker.py
        currently the heatmaps and image shapes are decoupled, consider changing"""
        representations = self.get_representations(images)
        out = self.heatmaps_from_representation(representations)
        return out

    def training_step(self, data, batch_idx):
        # load batch
        images, true_heatmaps = data
        # forward pass: images -> heatmaps
        predicted_heatmaps = self.forward(images)
        # compute loss
        heatmap_loss = MaskedMSEHeatmapLoss(true_heatmaps, predicted_heatmaps)

        self.log(
            "train_loss",
            heatmap_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": heatmap_loss}

    def validation_step(self, data: Tuple, batch_idx: int) -> None:
        images, true_heatmaps = data
        predicted_heatmaps = self.forward(images)
        # compute loss
        loss = MaskedMSEHeatmapLoss(true_heatmaps, predicted_heatmaps)
        # log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        # TODO: check that the logging names are fine
        self.validation_step(data, batch_idx)

    # TODO: can we define subpixmax class? that gets all the inputs at init and then operates given just heatmaps?
    def computeSubPixMax(self, heatmaps_pred, heatmaps_y, threshold):
        assert hasattr(self, "output_shape")
        kernel_size = np.min(self.output_shape)
        kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
        pred_keypoints = find_subpixel_maxima(
            heatmaps_pred.detach(),
            torch.tensor(kernel_size, device=heatmaps_pred.device),
            torch.tensor(self.output_sigma, device=heatmaps_pred.device),
            self.upsample_factor,
            self.coordinate_scale,
            self.confidence_scale,
        )
        y_keypoints = find_subpixel_maxima(
            heatmaps_y.detach(),
            torch.tensor(kernel_size, device=heatmaps_pred.device),
            torch.tensor(self.output_sigma, device=heatmaps_pred.device),
            self.upsample_factor,
            self.coordinate_scale,
            self.confidence_scale,
        )
        pred_keypoints = pred_keypoints[0]
        y_keypoints = y_keypoints[0]
        if threshold:
            # for i in range(pred_keypoints.shape[0]): # pred_keypoints is shape(num_keypoints, 3) the last entry being (x,y, confidence)
            #     if pred_keypoints[i, 2] > 0.008: #threshold for low confidence predictions
            #         pred_kpts_list.append(pred_keypoints[i, :2].cpu().numpy())
            #     if y_keypoints[i, 2] > 0.008:
            #         y_kpts_list.append(y_keypoints[i, :2].cpu().numpy())
            # print(pred_kpts_list, y_kpts_list)
            num_threshold = torch.tensor(0.001, device=heatmaps_pred.device)
            pred_mask = torch.gt(pred_keypoints[:, 2], num_threshold)
            pred_mask = pred_mask.unsqueeze(-1)
            y_mask = torch.gt(y_keypoints[:, 2], num_threshold)
            y_mask = y_mask.unsqueeze(-1)
            pred_keypoints = torch.masked_select(pred_keypoints, pred_mask).reshape(
                -1, 3
            )
            y_keypoints = torch.masked_select(y_keypoints, y_mask).reshape(-1, 3)

        pred_keypoints = pred_keypoints[:, :2]  # getting rid of the actual max value
        y_keypoints = y_keypoints[:, :2]
        return pred_keypoints, y_keypoints

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=20, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
