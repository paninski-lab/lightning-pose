from pose_est_nets.models.base_resnet import BaseFeatureExtractor
import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Callable, Optional, Tuple, List
from typing_extensions import Literal
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
        resnet_version: Optional[Literal[18, 34, 50, 101, 152]] = 18,
        downsample_factor: Optional[
            Literal[2, 3]
        ] = 2,  # TODO: downsample_factor may be in mismatch between datamodule and model. consider adding support for more types
        pretrained: Optional[bool] = False,
        last_resnet_layer_to_get: Optional[int] = -3,
    ) -> None:
        """
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
        self.num_targets = num_targets
        self.downsample_factor = downsample_factor
        self.upsampling_layers = self.make_upsampling_layers()
        self.initialize_upsampling_layers()

    @property
    def num_keypoints(self):
        return self.num_targets // 2

    @property
    def num_filters_for_upsampling(self):
        return self.backbone.fc.in_features

    @property
    def coordinate_scale(self):
        return torch.tensor(2 ** self.downsample_factor, device=self.device)

    def initialize_upsampling_layers(self) -> None:
        # TODO: test that running this method changes the weights and biases
        """loop over the Conv2DTranspose upsampling layers and initialize them"""
        for index, layer in enumerate(self.upsampling_layers):
            if index > 0:  # we ignore the PixelShuffle
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    @typechecked
    def make_upsampling_layers(self) -> torch.nn.Sequential:
        upsampling_layers = [nn.PixelShuffle(2)]
        upsampling_layers.append(
            self.create_double_upsampling_layer(
                in_channels=self.num_filters_for_upsampling // 4,
                out_channels=self.num_keypoints,
            )
        )  # running up to here results in downsample_factor=3 for [384,384] images
        if self.downsample_factor == 2:
            upsampling_layers.append(
                self.create_double_upsampling_layer(
                    in_channels=self.num_keypoints,
                    out_channels=self.num_keypoints,
                )
            )

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
    def heatmaps_from_representations(
        self,
        representations: TensorType[
            "Batch_Size",
            "Features",
            "Representation_Height",
            "Representation_Width",
            float,
        ],
    ) -> TensorType[
        "Batch_Size", "Num_Keypoints", "Heatmap_Height", "Heatmap_Width", float
    ]:
        """a wrapper around self.upsampling_layers for type and shape assertion.
        Args:
            representations (torch.tensor(float)): the output of the Resnet feature extractor.
        Returns:
            (torch.tensor(float)): the result of applying the upsampling layers to the representations.
        """
        return self.upsampling_layers(representations)

    @typechecked
    def forward(
        self, images: TensorType["Batch_Size", 3, "Image_Height", "Image_Width"]
    ) -> TensorType["Batch_Size", "Num_Keypoints", "Heatmap_Height", "Heatmap_Width",]:
        """
        Forward pass through the network
        :param x: images
        :return: heatmap per keypoint
        """
        """TODO: I have good assertions in the old heatmap_tracker.py
        currently the heatmaps and image shapes are decoupled, consider changing"""
        representations = self.get_representations(images)
        heatmaps = self.heatmaps_from_representations(representations)
        return heatmaps

    @typechecked
    def training_step(self, data_batch: Tuple, batch_idx: int) -> dict:
        images, true_heatmaps = data_batch  # read batch
        predicted_heatmaps = self.forward(images)  # images -> heatmaps
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

    @typechecked
    def evaluate(
        self, data_batch: Tuple, stage: Optional[Literal["val", "test"]] = None
    ):
        images, true_heatmaps = data_batch  # read batch
        predicted_heatmaps = self.forward(images)  # images -> heatmaps
        loss = MaskedMSEHeatmapLoss(true_heatmaps, predicted_heatmaps)
        # TODO: do we need other metrics?
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)

    def validation_step(self, validation_batch: Tuple, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: Tuple, batch_idx):
        self.evaluate(test_batch, "test")

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
