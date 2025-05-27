"""Heads that produce heatmap predictions for heatmap regression."""

from typing import Tuple

import torch
from kornia.filters import filter2d
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d
from kornia.geometry.transform.pyramid import _get_pyramid_gaussian_kernel
from torch import nn
from torchtyping import TensorType

from lightning_pose.data.utils import evaluate_heatmaps_at_location

# to ignore imports for sphix-autoapidoc
__all__ = [
    "make_upsampling_layers",
    "initialize_upsampling_layers",
    "upsample",
    "run_subpixelmaxima",
    "HeatmapHead",
    "HeatmapHeadNoShuffle",
    # "HybridUpsamplingHead",
    # "ResidualBlock",
    # "TransformerBlock",

]


def make_upsampling_layers(
    in_channels: int,
    out_channels: int,
    int_channels: int,
    n_layers: int,
    pixel_shuffle: bool = True,
) -> torch.nn.Sequential:
    # Note:
    # https://github.com/jgraving/DeepPoseKit/blob/
    # cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/DeepLabCut.py#L131
    # in their model, the pixel shuffle happens only for downsample_factor=2

    
    upsampling_layers = []
    if pixel_shuffle:
        upsampling_layers.append(nn.PixelShuffle(2))
    else:
        n_layers +=1 

    for layer in range(n_layers):
        if layer == 0:
            if pixel_shuffle:
                in_ = in_channels // 4  # division by 4 to account for PixelShuffle layer
            else:
                in_ = in_channels

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


def initialize_upsampling_layers(layers) -> None:
    """Intialize the Conv2DTranspose upsampling layers."""
    for index, layer in enumerate(layers):
        if index > 0:  # we ignore the PixelShuffle
            if isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1.0)
                torch.nn.init.constant_(layer.bias, 0.0)


def upsample(
    inputs: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
) -> TensorType["batch", "num_keypoints", "two_x_heatmap_height", "two_x_heatmap_width"]:
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
    heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    downsample_factor: int,
    temperature: torch.tensor,
) -> Tuple[TensorType["batch", "num_targets"], TensorType["batch", "num_keypoints"]]:
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
    ):
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

        n_layers = 4 - self.downsample_factor
        if self.backbone_arch in ["vit_h_sam", "vit_b_sam"]:
            n_layers -= 1

        self.upsampling_layers = make_upsampling_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            int_channels=deconv_out_channels or out_channels,
            n_layers=n_layers,
            pixel_shuffle=True,
        )
        initialize_upsampling_layers(self.upsampling_layers)

    def forward(
        self,
        features: TensorType["batch", "features", "features_height", "features_width"],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Upsample representations and normalize to get final heatmaps."""
        heatmaps = self.upsampling_layers(features)
        if self.final_softmax:
            # softmax temp stays 1 here; to modify for model predictions, see constructor
            heatmaps = spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))
        return heatmaps

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


class HeatmapHeadNoShuffle(nn.Module):
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
    ):
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

        n_layers = 4 - self.downsample_factor
        if self.backbone_arch in ["vit_h_sam", "vit_b_sam"]:
            n_layers -= 1

        self.upsampling_layers = make_upsampling_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            int_channels=deconv_out_channels or out_channels,
            n_layers=n_layers,
            pixel_shuffle=False,
        )
        initialize_upsampling_layers(self.upsampling_layers)

    def forward(
        self,
        features: TensorType["batch", "features", "features_height", "features_width"],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Upsample representations and normalize to get final heatmaps."""
        heatmaps = self.upsampling_layers(features)
        if self.final_softmax:
            # softmax temp stays 1 here; to modify for model predictions, see constructor
            heatmaps = spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))
        return heatmaps

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)



# class ResidualBlock(nn.Module):
#     """Residual block with batch normalization for improved gradient flow."""
    
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels)
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#         return out


# class TransformerBlock(nn.Module):
#     """Lightweight spatial transformer block for feature refinement."""
    
#     def __init__(self, channels, nhead=8):
#         super().__init__()
#         self.norm = nn.LayerNorm([channels, 16, 16])  # Assuming 8x8 becomes 16x16 after initial upsampling
        
#         # Efficient multi-head self-attention
#         self.attention = nn.MultiheadAttention(
#             embed_dim=channels,
#             num_heads=nhead,
#             batch_first=True
#         )
        
#         # Feed-forward network
#         self.ffn = nn.Sequential(
#             nn.Linear(channels, channels * 2),
#             nn.ReLU(),
#             nn.Linear(channels * 2, channels)
#         )
        
#         self.norm1 = nn.LayerNorm(channels)
#         self.norm2 = nn.LayerNorm(channels)
        
#     def forward(self, x):
#         b, c, h, w = x.shape
        
#         # Reshape for self-attention
#         x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
#         # Self-attention with skip connection
#         attn_out, _ = self.attention(x_flat, x_flat, x_flat)
#         x_flat = x_flat + attn_out
#         x_flat = self.norm1(x_flat)
        
#         # Feed-forward with skip connection
#         ffn_out = self.ffn(x_flat)
#         x_flat = x_flat + ffn_out
#         x_flat = self.norm2(x_flat)
        
#         # Reshape back to spatial
#         x_out = x_flat.permute(0, 2, 1).reshape(b, c, h, w)
        
#         return x_out


# class HybridUpsamplingHead(nn.Module):
#     """Hybrid CNN-Transformer upsampling head for multiview pose estimation.
    
#     This head combines the best of CNN and transformer approaches:
#     1. Initial bilinear upsampling to increase spatial resolution efficiently
#     2. Channel reduction with 1x1 convolution to manage computational complexity
#     3. Spatial transformer block to refine features with self-attention
#     4. Progressive upsampling with residual blocks to maintain gradient flow
#     5. Final prediction layer with optional softmax for heatmap generation
#     """

#     def __init__(
#         self,
#         backbone_arch: str,
#         in_channels: int,
#         out_channels: int,
#         downsample_factor: int = 2,
#         final_softmax: bool = True,
#     ):
#         """
#         Args:
#             backbone_arch: string denoting backbone architecture
#             in_channels: number of channels in the input feature map (from transformer)
#             out_channels: number of channels in the output heatmap (i.e. number of keypoints)
#             downsample_factor: make heatmaps smaller than input frames by this factor
#             final_softmax: pass final heatmaps through a 2D softmax
#         """
#         super().__init__()

#         self.backbone_arch = backbone_arch
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.downsample_factor = downsample_factor
#         self.final_softmax = final_softmax
#         self.temperature = torch.tensor(1000.0)  # soft argmax temp
        
#         # Initial upsampling with bilinear interpolation (8x8 -> 16x16)
#         self.initial_upsample = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels // 2),
#             nn.ReLU(inplace=True)
#         )
        
#         # Spatial transformer block for feature refinement
#         self.spatial_transformer = TransformerBlock(in_channels // 2)
        
#         # Progressive upsampling with residual blocks
#         # For downsample_factor=2, we need 2 more upsampling stages (16x16 -> 64x64)
#         n_progressive_layers = 4 - downsample_factor
#         if backbone_arch in ["vit_h_sam", "vit_b_sam"]:
#             n_progressive_layers -= 1
            
#         self.progressive_upsampling = nn.ModuleList()
#         curr_channels = in_channels // 2
        
#         for i in range(n_progressive_layers):
#             # Determine output channels for this stage
#             out_channels_i = curr_channels // 2 if i < n_progressive_layers - 1 else out_channels
            
#             # Create upsampling block
#             block = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(curr_channels, out_channels_i, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels_i),
#                 nn.ReLU(inplace=True),
#                 ResidualBlock(out_channels_i)
#             )
            
#             self.progressive_upsampling.append(block)
#             curr_channels = out_channels_i

#         # Initialize weights
#         self._initialize_weights()
        
#     def _initialize_weights(self):
#         """Initialize all convolution layers with Xavier uniform."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 torch.nn.init.constant_(m.weight, 1.0)
#                 torch.nn.init.constant_(m.bias, 0.0)

#     def forward(
#         self,
#         features: TensorType["batch", "features", "features_height", "features_width"],
#     ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
#         """Upsample representations and normalize to get final heatmaps."""
#         # Initial upsampling (8x8 -> 16x16)
#         x = self.initial_upsample(features)
        
#         # Apply spatial transformer for feature refinement
#         x = self.spatial_transformer(x)
        
#         # Progressive upsampling
#         for upsample_block in self.progressive_upsampling:
#             x = upsample_block(x)
        
#         # Apply final softmax if needed
#         if self.final_softmax:
#             # softmax temp stays 1 here; to modify for model predictions, see constructor
#             x = spatial_softmax2d(x, temperature=torch.tensor([1.0]))
            
#         return x

#     def run_subpixelmaxima(self, heatmaps):
#         """Use soft argmax to get subpixel precise keypoint locations."""
#         return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)