"""Heads that produce heatmap predictions for heatmap regression."""

from typing import Tuple

import torch
from kornia.geometry.subpix import spatial_softmax2d
from torch import Tensor, nn
from torchtyping import TensorType
from typing_extensions import Literal

from lightning_pose.models.heads import HeatmapHead
from lightning_pose.models.heads.heatmap import run_subpixelmaxima

# to ignore imports for sphix-autoapidoc
__all__ = [
    "MultiviewHeatmapCNNHead",
    "MultiviewHeatmapCNNMultiHead",
    "ResidualBlock",
]


class MultiviewHeatmapCNNHead(nn.Module):
    """Multi-view convolutional neural network head that operates on heatmaps.

    This head takes a set of 2D feature maps corresponding to different views, and fuses them
    together.

    """

    def __init__(
        self,
        backbone_arch: str,
        num_views: int,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
    ):
        """

        Args:
            backbone_arch: string denoting backbone architecture; to remove in future release
            num_views: number of camera views in each batch
            in_channels: number of channels in the input feature map
            out_channels: number of channels in the output heatmap (i.e. number of keypoints)
            deconv_out_channels: output channel number for each intermediate deconv layer; defaults
                to number of keypoints
            downsample_factor: make heatmaps smaller than input frames by this factor; subpixel
                operations are performed for increased precision

        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor
        self.temperature = torch.tensor(1000.0)  # soft argmax temp

        # create upsampling head
        self.upsample = HeatmapHead(
            backbone_arch=backbone_arch,
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            downsample_factor=downsample_factor,
            final_softmax=False,
        )

        # create multi-view fusion head
        self.num_views = num_views
        self.fusion = ResidualBlock(
            in_channels=num_views,
            intermediate_channels=32,
            out_channels=num_views,
            final_relu=False,
            final_softmax=True,
        )

    def forward(
        self,
        features: TensorType["view x batch", "features", "rep_height", "rep_width"],
        num_views: torch.tensor,
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Upsample and run multiview head to get final heatmaps.

        Args:
            features: outputs of backbone
            num_views: number of camera views for each batch element

        """

        batch_size_combined = features.shape[0]
        batch_size = int(batch_size_combined // num_views)

        # upsample features
        heatmaps = self.upsample(features)
        # heatmaps = [view * batch, num_keypoints, heatmap_height, heatmap_width]
        heat_height = heatmaps.shape[-2]
        heat_width = heatmaps.shape[-1]

        # now we will process the heatmaps for each set of corresponding keypoints with the
        # multiview head; this requires lots of reshaping

        heatmaps = heatmaps.reshape(batch_size, num_views, -1, heat_height, heat_width)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps = heatmaps.permute(0, 2, 1, 3, 4)
        # heatmaps = [batch, num_keypoints, views, heatmap_height, heatmap_width]
        heatmaps = heatmaps.reshape(-1, num_views, heat_height, heat_width)
        # heatmaps = [num_keypoints * batch, views, heatmap_height, heatmap_width]
        heatmaps = self.fusion(heatmaps)
        # heatmaps = [num_keypoints * batch, views, heatmap_height, heatmap_width]

        # reshape heatmaps back to their original shape
        heatmaps = heatmaps.reshape(batch_size, -1, num_views, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints, views, heatmap_height, heatmap_width]
        heatmaps = heatmaps.permute(0, 2, 1, 3, 4)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps = heatmaps.reshape(batch_size, -1, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]

        return heatmaps

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


class MultiviewHeatmapCNNMultiHead(nn.Module):
    """Multi-head, multi-view convolutional neural network head.

    This head takes a set of 2D feature maps corresponding to different views, and fuses them
    together.

    The head is composed of two heads:
    - single view head: several deconvolutional layers followed by a 2D spatial softmax to
      generate normalized heatmaps from low-resolution feature maps for each single view.
    - multi-view head: several deconvolutional layers are applied to each set of features; the
      resulting heatmaps are fed into a convolutional neural network to produce fused heatmaps.
      CNN parameters for the fusion head are shared across keypoints.

    """

    def __init__(
        self,
        backbone_arch: str,
        num_views: int,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        upsampling_factor: int = 2,
    ):
        """

        Args:
            backbone_arch: string denoting backbone architecture; to remove in future release
            num_views: number of camera views in each batch
            in_channels: number of channels in the input feature map
            out_channels: number of channels in the output heatmap (i.e. number of keypoints)
            deconv_out_channels: output channel number for each intermediate deconv layer; defaults
                to number of keypoints
            downsample_factor: make heatmaps smaller than input frames by this factor; subpixel
                operations are performed for increased precision
            upsampling_factor: upsample features before feeding to crnn

        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor
        self.upsampling_factor = upsampling_factor
        self.temperature = torch.tensor(1000.0)  # soft argmax temp

        # create single-view head
        self.head_sv = HeatmapHead(
            backbone_arch=backbone_arch,
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            downsample_factor=downsample_factor,
            final_softmax=True,
        )

        # create multi-view upsampling head
        self.head_mv = HeatmapHead(
            backbone_arch=backbone_arch,
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            downsample_factor=downsample_factor,
            final_softmax=False,
        )

        # create multi-view fusion head
        self.num_views = num_views
        self.head_fusion = ResidualBlock(
            in_channels=num_views,
            intermediate_channels=32,
            out_channels=num_views,
            final_relu=False,
            final_softmax=True,
        )

    def forward(
        self,
        features: TensorType["view x batch", "features", "rep_height", "rep_width"],
        num_views: torch.tensor,
    ) -> Tuple[
        TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Upsample and run multiview head to get final heatmaps.

        Args:
            features: outputs of backbone
            num_views: number of camera views for each batch element

        """

        batch_size_combined = features.shape[0]
        batch_size = int(batch_size_combined // num_views)

        # ----------------------------------------------------------------------------
        # process single view head (upsampling only)
        # ----------------------------------------------------------------------------
        heatmaps_sv = self.head_sv(features)
        # heatmaps = [view * batch, num_keypoints, heatmap_height, heatmap_width]
        heat_height = heatmaps_sv.shape[-2]
        heat_width = heatmaps_sv.shape[-1]

        heatmaps_sv = heatmaps_sv.reshape(batch_size, num_views, -1, heat_height, heat_width)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps_sv = heatmaps_sv.reshape(batch_size, -1, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]

        # ----------------------------------------------------------------------------
        # process multi-view head (upsampling + multiview)
        # ----------------------------------------------------------------------------
        heatmaps_mv = self.head_mv(features)
        # heatmaps = [view * batch, num_keypoints, heatmap_height, heatmap_width]

        # now we will process the heatmaps for each set of corresponding keypoints with the
        # multiview head; this requires lots of reshaping

        heatmaps_mv = heatmaps_mv.reshape(batch_size, num_views, -1, heat_height, heat_width)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.permute(0, 2, 1, 3, 4)
        # heatmaps = [batch, num_keypoints, views, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.reshape(-1, num_views, heat_height, heat_width)
        # heatmaps = [num_keypoints * batch, views, heatmap_height, heatmap_width]
        heatmaps_mv = self.head_fusion(heatmaps_mv)
        # heatmaps = [num_keypoints * batch, views, heatmap_height, heatmap_width]

        # reshape heatmaps back to their original shape
        heatmaps_mv = heatmaps_mv.reshape(batch_size, -1, num_views, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints, views, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.permute(0, 2, 1, 3, 4)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.reshape(batch_size, -1, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]

        return heatmaps_sv, heatmaps_mv

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


class ResidualBlock(nn.Module):
    """Resnet residual block module.

    Adapted from:
    https://github.com/pytorch/vision/blob/4249b610811b290ea9ac9e445260be195ce52ae1/torchvision/models/resnet.py#L59  # noqa

    """
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        final_relu: bool = False,
        final_softmax: bool = False,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.final_relu = final_relu
        self.final_softmax = final_softmax

        self.initialize_layers()

    def initialize_layers(self):

        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=0.01)
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.constant_(self.bn1.weight, 1.0)
        torch.nn.init.constant_(self.bn1.bias, 0.0)

        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=0.01)
        torch.nn.init.zeros_(self.conv2.bias)

        torch.nn.init.constant_(self.bn2.weight, 1.0)
        torch.nn.init.constant_(self.bn2.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        # first layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # second layer
        out = self.conv2(out)
        out = self.bn2(out)
        # residual connection
        out += x
        if self.final_relu:
            out = self.relu(out)
        if self.final_softmax:
            out = spatial_softmax2d(out, temperature=torch.tensor([1.0]))
        return out
