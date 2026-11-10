"""Heatmap head with a nonlinear (per-pixel MLP) readout."""

import torch
from torch import nn

from lightning_pose.models.heads.heatmap import HeatmapHead

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels at every pixel of a (B, C, H, W) map (ConvNeXt style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class HeatmapNonlinearHead(HeatmapHead):
    """Deconvolution head with a nonlinear readout before the spatial softmax.

    Same upsampling stack as :class:`HeatmapHead`, except that the last transposed convolution
    outputs ``hidden_channels`` channels instead of one per keypoint. These pass through
    LayerNorm over channels -> ReLU -> 1x1 convolution to one channel per keypoint, i.e. a
    per-pixel MLP readout. The layers live in ``upsampling_layers`` like in the linear head.

    """

    def __init__(
        self,
        backbone_arch: str,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        final_softmax: bool = True,
        hidden_channels: int = 256,
    ) -> None:
        """

        Args:
            backbone_arch: string denoting backbone architecture; to remove in future release
            in_channels: number of channels in the input feature map
            out_channels: number of channels in the output heatmap (i.e. number of keypoints)
            deconv_out_channels: output channel number for each intermediate deconv layer; defaults
                to ``hidden_channels``
            downsample_factor: make heatmaps smaller than input frames by this factor; subpixel
                operations are performed for increased precision
            final_softmax: pass final heatmaps through a 2D softmax with temperature 1.0
            hidden_channels: width of the readout: output channels of the last transposed
                convolution and input channels of the final 1x1 convolution

        """
        # the parent builds the upsampling stack ending in ``hidden_channels`` channels
        super().__init__(
            backbone_arch=backbone_arch,
            in_channels=in_channels,
            out_channels=hidden_channels,
            deconv_out_channels=deconv_out_channels,
            downsample_factor=downsample_factor,
            final_softmax=final_softmax,
        )
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        # same small init as the transposed convolutions of the linear head
        nn.init.xavier_uniform_(readout.weight, gain=0.01)
        nn.init.zeros_(readout.bias)
        self.upsampling_layers = nn.Sequential(
            *self.upsampling_layers,
            LayerNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            readout,
        )
