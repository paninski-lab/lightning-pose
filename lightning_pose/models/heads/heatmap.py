"""Heads that produce heatmap predictions for heatmap regression."""


import math

import torch
from jaxtyping import Float
from kornia.filters import filter2d
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d
from kornia.geometry.transform.pyramid import _get_pyramid_gaussian_kernel
from torch import nn

from lightning_pose.data.heatmaps import evaluate_heatmaps_at_location
from lightning_pose.models.backbones import BACKBONE_STRIDES

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def make_upsampling_layers(
    in_channels: int,
    out_channels: int,
    int_channels: int,
    n_layers: int,
) -> torch.nn.Sequential:
    """Build a sequential upsampling head composed of a PixelShuffle and ConvTranspose2d layers.

    Args:
        in_channels: number of input feature-map channels.
        out_channels: number of output channels (i.e. number of keypoints).
        int_channels: number of channels in each intermediate convolutional layer.
        n_layers: total number of ConvTranspose2d layers after the PixelShuffle.

    Returns:
        ``nn.Sequential`` starting with a ``PixelShuffle(2)`` followed by ``n_layers``
        ConvTranspose2d layers.
    """
    # Note:
    # https://github.com/jgraving/DeepPoseKit/blob/
    # cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/DeepLabCut.py#L131
    # in their model, the pixel shuffle happens only for downsample_factor=2

    upsampling_layers = []
    upsampling_layers.append(nn.PixelShuffle(2))
    for layer in range(n_layers):

        if layer == 0:
            in_ = in_channels // 4  # division by 4 to account for PixelShuffle layer
            if n_layers == 1:
                out_ = out_channels
            else:
                out_ = int_channels
        elif layer == n_layers - 1:
            in_ = int_channels if n_layers > 1 else in_channels // 4
            out_ = out_channels
        else:
            in_ = int_channels
            out_ = int_channels

        upsampling_layers.append(
            nn.ConvTranspose2d(
                in_channels=in_,
                out_channels=out_,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            )
        )

    return nn.Sequential(*upsampling_layers)


def initialize_upsampling_layers(layers: nn.Sequential) -> None:
    """Intialize the Conv2DTranspose upsampling layers."""
    for index, layer in enumerate(layers):
        if index > 0:  # we ignore the PixelShuffle
            if isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                torch.nn.init.zeros_(layer.bias)  # type: ignore[arg-type]
            elif isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1.0)
                torch.nn.init.constant_(layer.bias, 0.0)


def upsample(
    inputs: Float[torch.Tensor, "batch num_keypoints heatmap_height heatmap_width"],
) -> Float[torch.Tensor, "batch num_keypoints two_x_heatmap_height two_x_heatmap_width"]:
    """Upsample batch of heatmaps by a factor of two using interpolation (no learned weights).

    This is a copy of kornia's pyrup function but with better defaults.
    """
    kernel = _get_pyramid_gaussian_kernel()
    _, _, height, width = inputs.shape
    # align_corners=False is important!! otherwise the offsets below don't hold
    inputs_up = nn.functional.interpolate(
        inputs, size=(height * 2, width * 2), mode="bicubic", align_corners=False,
    )
    inputs_up = filter2d(inputs_up, kernel, border_type="constant")
    return inputs_up


def run_subpixelmaxima(
    heatmaps: Float[torch.Tensor, "batch num_keypoints heatmap_height heatmap_width"],
    downsample_factor: int,
    temperature: torch.Tensor,
) -> tuple[Float[torch.Tensor, "batch num_targets"], Float[torch.Tensor, "batch num_keypoints"]]:
    """Use soft argmax on heatmaps.

    Args:
        heatmaps: output of upsampling layers
        downsample_factor: controls how many times upsampling needs to be performed
        temperature: temperature parameter of softmax; higher leads to tighter peaks

    Returns:
        tuple
            - soft argmax of shape (batch, num_targets)
            - confidences of shape (batch, num_keypoints)

    """

    # upsample heatmaps
    for _ in range(downsample_factor):
        heatmaps = upsample(heatmaps)
    # find soft argmax
    softmaxes = spatial_softmax2d(heatmaps, temperature=temperature)
    preds = spatial_expectation2d(softmaxes, normalized_coordinates=False)
    # compute confidences as softmax value pooled around prediction
    confidences = evaluate_heatmaps_at_location(heatmaps=softmaxes, locs=preds)
    # fix grid offsets from upsampling
    if downsample_factor == 1:
        preds -= 0.5
    elif downsample_factor == 2:
        preds -= 1.5
    elif downsample_factor == 3:
        preds -= 2.5

    # NOTE: we cannot use
    # `preds.reshape(-1, self.num_targets)`
    # This works fine for the non-multiview case
    # This works fine for multiview training
    # This fails during multiview inference when we might have an arbitrary number of views
    # that we are processing (self.num_targets is tied to the labeled data)
    return preds.reshape(-1, heatmaps.shape[1] * 2), confidences


class HeatmapHead(nn.Module):
    """Simple deconvolution head that converts 2D feature maps to per-keypoint heatmaps.

    This is the standard heatmap head used in the Lightning Pose package. The head is composed of
    several deconvolutional layers followed by a 2D spatial softmax to generate normalized heatmaps
    from low-resolution feature maps.

    """

    def __init__(
        self,
        backbone_arch: str,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        final_softmax: bool = True,
    ) -> None:
        """

        Args:
            backbone_arch: string denoting backbone architecture; to remove in future release
            in_channels: number of channels in the input feature map
            out_channels: number of channels in the output heatmap (i.e. number of keypoints)
            deconv_out_channels: output channel number for each intermediate deconv layer; defaults
                to number of keypoints
            downsample_factor: make heatmaps smaller than input frames by this factor; subpixel
                operations are performed for increased precision
            final_softmax: pass final heatmaps through a 2D softmax with temperature 1.0

        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor
        self.final_softmax = final_softmax
        # TODO: temp=1000 works for 64x64 heatmaps, need to generalize to other shapes
        self.temperature = torch.tensor(1000.0)  # soft argmax temp

        # n_layers = log2(stride) - downsample_factor - 1, derived from the requirement that
        # PixelShuffle(2) × ConvTranspose2d(stride=2)^n_layers must upsample the backbone
        # feature map (at 1/stride of input) to the target heatmap (at 1/2^downsample_factor).
        stride = BACKBONE_STRIDES.get(self.backbone_arch, 32)
        n_layers = int(math.log2(stride)) - self.downsample_factor - 1

        self.upsampling_layers = make_upsampling_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            int_channels=deconv_out_channels or out_channels,
            n_layers=n_layers,
        )
        initialize_upsampling_layers(self.upsampling_layers)

    def forward(
        self,
        features: Float[torch.Tensor, "batch features features_height features_width"],
    ) -> Float[torch.Tensor, "batch num_keypoints heatmap_height heatmap_width"]:
        """Upsample representations and normalize to get final heatmaps."""
        heatmaps = self.upsampling_layers(features)
        if self.final_softmax:
            # softmax temp stays 1 here; to modify for model predictions, see constructor
            heatmaps = spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))
        return heatmaps

    def run_subpixelmaxima(
        self,
        heatmaps: Float[torch.Tensor, "batch num_keypoints height width"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply soft argmax to heatmaps to obtain subpixel keypoint predictions.

        Args:
            heatmaps: predicted heatmaps of shape ``(batch, num_keypoints, height, width)``.

        Returns:
            Tuple of ``(keypoints, confidences)`` where keypoints has shape
            ``(batch, num_targets)`` and confidences has shape ``(batch, num_keypoints)``.
        """
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)
