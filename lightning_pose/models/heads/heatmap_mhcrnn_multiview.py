"""Multiview MHCRNN head for efficient spatiotemporal processing."""

from typing import Tuple
import torch
from torch import nn
from torchtyping import TensorType
from kornia.geometry.subpix import spatial_softmax2d

from lightning_pose.models.heads import HeatmapHead
from lightning_pose.models.heads.heatmap import run_subpixelmaxima

# to ignore imports for sphix-autoapidoc
__all__ = []


class HeatmapMHCRNNHeadMultiview(nn.Module):
    """Efficient MHCRNN head for multiview data with single backbone pass."""

    def __init__(
        self,
        backbone_arch: str,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        upsampling_factor: int = 2,
        num_views: int = 2,
    ):
        """
        Args:
            backbone_arch: backbone architecture name
            in_channels: input feature channels
            out_channels: number of keypoints
            deconv_out_channels: intermediate deconv channels
            downsample_factor: spatial downsampling factor
            upsampling_factor: temporal upsampling factor
            num_views: number of camera views
        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor
        self.upsampling_factor = upsampling_factor
        self.num_views = num_views
        self.temperature = torch.tensor(1000.0)
        
        # Single-frame head for comparison
        self.head_sf = HeatmapHead(
            backbone_arch=backbone_arch,
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            downsample_factor=downsample_factor,
        )

        # Efficient multiview temporal processor
        # hidden_dim should match channels after pixel shuffle for efficiency
        # but we allow projection if needed
        channels_after_shuffle = in_channels // 4
        self.head_mf = UpsamplingCRNNMultiview(
            num_filters_for_upsampling=in_channels,
            num_keypoints=out_channels,
            upsampling_factor=upsampling_factor,
            hidden_dim=min(256, channels_after_shuffle),  # Use smaller of the two
            nfilters_channel=16,
        )

    def forward(
        self,
        features: TensorType["batch*views", "features", "rep_height", "rep_width", "frames"],
        batch_shape: torch.tensor,
        is_multiview: bool,
    ) -> Tuple[
        TensorType["batch*views", "num_keypoints", "heatmap_height", "heatmap_width"],
        TensorType["batch*views", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Efficient forward pass with single temporal processing."""
        
        if not is_multiview:
            raise ValueError("HeatmapMHCRNNHeadMultiview requires is_multiview=True")
        
        # Extract dimensions
        if len(batch_shape) == 6:
            batch_size, num_views, num_frames = batch_shape[0:3].tolist()
        elif len(batch_shape) == 5:
            batch_size, num_views = batch_shape[0:2].tolist()
            num_frames = features.shape[-1]
        else:
            raise ValueError(f"Expected batch_shape with 5 or 6 dimensions, got {len(batch_shape)}")

        assert features.shape[0] == batch_size * num_views, \
            f"Features batch dimension {features.shape[0]} != batch_size * num_views {batch_size * num_views}"

        # Permute to (frames, batch*views, features, rep_height, rep_width)
        features = features.permute(4, 0, 1, 2, 3)
        
        # Get middle frame for single-frame processing
        middle_frame_idx = num_frames // 2
        heatmaps_sf = self.head_sf(features[middle_frame_idx])
        
        # Process all frames through efficient multiview CRNN
        heatmaps_mf = self.head_mf(features)

        return heatmaps_sf, heatmaps_mf

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


class UpsamplingCRNNMultiview(nn.Module):
    """Efficient bidirectional CRNN for multiview temporal processing."""

    def __init__(
        self,
        num_filters_for_upsampling: int,
        num_keypoints: int,
        upsampling_factor: int = 2,
        hidden_dim: int = 256,
        nfilters_channel: int = 16,
    ):
        super().__init__()

        self.upsampling_factor = upsampling_factor
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.dropout = nn.Dropout2d(0.25)

        # Feature adapter to reduce dimensionality
        self.feature_adapter = nn.Conv2d(
            num_filters_for_upsampling // 4, 
            hidden_dim,
            kernel_size=3, 
            padding=1
        )

        # Pre-upsampling layers
        if self.upsampling_factor == 2:
            self.W_pre = nn.ConvTranspose2d(
                in_channels=hidden_dim,
                out_channels=num_keypoints,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            )
            in_channels_rnn = num_keypoints
        else:
            in_channels_rnn = hidden_dim

        # Forward RNN
        self.W_f = nn.ConvTranspose2d(
            in_channels=in_channels_rnn,
            out_channels=num_keypoints,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        
        self.H_f = nn.Sequential(
            nn.Conv2d(
                in_channels=num_keypoints,
                out_channels=num_keypoints * nfilters_channel,
                kernel_size=2, stride=2, padding=0,
                groups=num_keypoints,
            ),
            nn.ConvTranspose2d(
                in_channels=num_keypoints * nfilters_channel,
                out_channels=num_keypoints,
                kernel_size=2, stride=2, padding=0,
                groups=num_keypoints,
            ),
        )

        # Backward RNN
        self.W_b = nn.ConvTranspose2d(
            in_channels=in_channels_rnn,
            out_channels=num_keypoints,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        
        self.H_b = nn.Sequential(
            nn.Conv2d(
                in_channels=num_keypoints,
                out_channels=num_keypoints * nfilters_channel,
                kernel_size=2, stride=2, padding=0,
                groups=num_keypoints,
            ),
            nn.ConvTranspose2d(
                in_channels=num_keypoints * nfilters_channel,
                out_channels=num_keypoints,
                kernel_size=2, stride=2, padding=0,
                groups=num_keypoints,
            ),
        )

        self._initialize_layers()

    def _initialize_layers(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: TensorType["frames", "batch*views", "features", "rep_height", "rep_width"]
    ) -> TensorType["batch*views", "num_keypoints", "heatmap_height", "heatmap_width"]:
        
        frames, batch_views, n_features, rep_height, rep_width = features.shape

        # Process each frame: pixel shuffle + adapt + dropout
        processed_frames = []
        for frame_idx in range(frames):
            if self.upsampling_factor == 2:
                shuffled = self.pixel_shuffle(features[frame_idx])
                adapted = self.feature_adapter(shuffled)
                adapted = self.dropout(adapted)
                upsampled = self.W_pre(adapted)
                processed_frames.append(upsampled)
            else:
                shuffled = self.pixel_shuffle(features[frame_idx])
                adapted = self.feature_adapter(shuffled)
                adapted = self.dropout(adapted)
                processed_frames.append(adapted)
        
        x_tensor = torch.stack(processed_frames, dim=0)

        # Bidirectional RNN with dropout
        x_f = self.W_f(x_tensor[0])
        for frame_batch in x_tensor[1:]:
            x_f = self.W_f(frame_batch) + self.H_f(x_f)
            x_f = self.dropout(x_f)

        x_tensor_b = torch.flip(x_tensor, dims=[0])
        x_b = self.W_b(x_tensor_b[0])
        for frame_batch in x_tensor_b[1:]:
            x_b = self.W_b(frame_batch) + self.H_b(x_b)
            x_b = self.dropout(x_b)

        heatmaps = (x_f + x_b) / 2

        return spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))