"""Heads that produce heatmap predictions for heatmap regression."""

from typing import Tuple

import torch
from kornia.geometry.subpix import spatial_softmax2d
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from lightning_pose.models.heads import HeatmapHead
from lightning_pose.models.heads.heatmap import run_subpixelmaxima

# to ignore imports for sphix-autoapidoc
__all__ = []


class HeatmapMHCRNNHead(nn.Module):
    """Multi-head convolutional recurrent neural network head.

    This head converts a sequence of 2D feature maps to per-keypoint heatmaps for the center frame.
    The head is composed of two heads:
    - single frame head: several deconvolutional layers followed by a 2D spatial softmax to
      generate normalized heatmaps from low-resolution feature maps for a single frame.
    - multi-frame head: several deconvolutional layers are applied to each set of features in a
      temporal sequence; the resulting heatmaps are fed into a convolutional recurrent neural
      network to produce heatmaps for the center frame

    """

    def __init__(
        self,
        backbone_arch: str,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        upsampling_factor: int = 2,
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

        # create single-frame head
        self.head_sf = HeatmapHead(
            backbone_arch=backbone_arch,
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            downsample_factor=downsample_factor,
        )

        # create multi-frame head
        self.head_mf = UpsamplingCRNN(
            num_filters_for_upsampling=self.head_sf.in_channels,
            num_keypoints=self.head_sf.out_channels,
            upsampling_factor=upsampling_factor,
        )

    def forward(
        self,
        features: TensorType["batch", "features", "rep_height", "rep_width", "frames"],
        batch_shape: torch.tensor,
        is_multiview: bool,
    ) -> Tuple[
        TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
        TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Handle context frames then upsample to get final heatmaps.

        Args:
            features: outputs of backbone
            batch_shape: identifies whether or not we need to do some reshaping
            is_multiview: if batch has a view dimension

        """

        num_frames = batch_shape[0]

        if len(batch_shape) == 5 and is_multiview:
            # put view info back in batch so we can properly extract heatmaps
            shape_r = features.shape
            num_frames -= 4  # we lose the first/last 2 frames of unlabeled batch due to context
            features = features.reshape(
                num_frames * batch_shape[1], -1, shape_r[-3], shape_r[-2], shape_r[-1],
            )

        # permute to shape (frames, batch, features, rep_height, rep_width)
        features = torch.permute(features, (4, 0, 1, 2, 3))
        heatmaps_sf = self.head_sf(features[2])  # index 2 == middle frame
        heatmaps_mf = self.head_mf(features)

        if len(batch_shape) == 6 or len(batch_shape) == 5:
            # reshape the outputs to extract the view dimension
            heatmaps_sf = heatmaps_sf.reshape(
                num_frames, -1, heatmaps_sf.shape[-2], heatmaps_sf.shape[-1]
            )
            heatmaps_mf = heatmaps_mf.reshape(
                num_frames, -1, heatmaps_mf.shape[-2], heatmaps_mf.shape[-1]
            )

        return heatmaps_sf, heatmaps_mf

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


class UpsamplingCRNN(nn.Module):
    """Bidirectional Convolutional RNN network that handles heatmaps of context frames.

    The input to the CRNN is a set of heatmaps at times t-k, ..., t, ...t+k, one heatmap for each
    timepoint/keypoint

    The output of the CRNN is a single heatmap for each keypoint

    """

    def __init__(
        self,
        num_filters_for_upsampling: int,
        num_keypoints: int,
        upsampling_factor: Literal[1, 2] = 2,
        hkernel: int = 2,
        hstride: int = 2,
        hpad: int = 0,
        nfilters_channel: int = 16,
    ) -> None:
        """Upsampling Convolutional RNN - initialize input and hidden weights."""

        super().__init__()

        self.upsampling_factor = upsampling_factor
        self.pixel_shuffle = nn.PixelShuffle(2)

        if self.upsampling_factor == 2:
            self.W_pre = torch.nn.ConvTranspose2d(
                in_channels=num_filters_for_upsampling // 4,
                out_channels=num_keypoints,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1),
            )
            in_channels_rnn = num_keypoints
        else:
            in_channels_rnn = num_filters_for_upsampling // 4

        self.W_f = torch.nn.ConvTranspose2d(
            in_channels=in_channels_rnn,
            out_channels=num_keypoints,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )

        H_f_layers = []
        H_f_layers.append(
            nn.Conv2d(
                in_channels=num_keypoints,
                out_channels=num_keypoints * nfilters_channel,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        H_f_layers.append(
            nn.ConvTranspose2d(
                in_channels=num_keypoints * nfilters_channel,
                out_channels=num_keypoints,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                output_padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        self.H_f = nn.Sequential(*H_f_layers)

        self.W_b = torch.nn.ConvTranspose2d(
            in_channels=in_channels_rnn,
            out_channels=num_keypoints,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )
        H_b_layers = []
        H_b_layers.append(
            nn.Conv2d(
                in_channels=num_keypoints,
                out_channels=num_keypoints * nfilters_channel,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        H_b_layers.append(
            nn.ConvTranspose2d(
                in_channels=num_keypoints * nfilters_channel,
                out_channels=num_keypoints,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                output_padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        self.H_b = nn.Sequential(*H_b_layers)

        self._initialize_layers()

        if self.upsampling_factor == 2:
            self.layers = torch.nn.ModuleList([self.W_pre, self.W_f, self.H_f, self.W_b, self.H_b])
        else:
            self.layers = torch.nn.ModuleList([self.W_f, self.H_f, self.W_b, self.H_b])

    def _initialize_layers(self):
        if self.upsampling_factor == 2:
            torch.nn.init.xavier_uniform_(self.W_pre.weight, gain=1.0)
            torch.nn.init.zeros_(self.W_pre.bias)

        torch.nn.init.xavier_uniform_(self.W_f.weight, gain=1.0)
        torch.nn.init.zeros_(self.W_f.bias)
        for index, layer in enumerate(self.H_f):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.xavier_uniform_(self.W_b.weight, gain=1.0)
        torch.nn.init.zeros_(self.W_b.bias)
        for index, layer in enumerate(self.H_b):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
            torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        features: TensorType["frames", "batch", "features", "rep_height", "rep_width"]
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:

        # expand representations in spatial domain using pixel shuffle to create heatmaps
        if self.upsampling_factor == 2:
            # upsample once more before passing through RNN
            # need to reshape to push through convolution layers
            frames, batch, n_features, rep_height, rep_width = features.shape
            frames_batch_shape = batch * frames
            representations_batch_frames: TensorType[
                "batch*frames", "features", "rep_height", "rep_width"
            ] = features.reshape(frames_batch_shape, n_features, rep_height, rep_width)
            x_tensor = self.W_pre(self.pixel_shuffle(representations_batch_frames))
            x_tensor = x_tensor.reshape(
                frames,
                batch,
                x_tensor.shape[1],
                x_tensor.shape[2],
                x_tensor.shape[3],
            )
        else:
            x_tensor = self.pixel_shuffle(features)

        # push heatmaps through CRNN
        x_f = self.W_f(x_tensor[0])
        for frame_batch in x_tensor[1:]:  # forward pass
            x_f = self.W_f(frame_batch) + self.H_f(x_f)
        x_tensor_b = torch.flip(x_tensor, dims=[0])
        x_b = self.W_b(x_tensor_b[0])
        for frame_batch in x_tensor_b[1:]:  # backwards pass
            x_b = self.W_b(frame_batch) + self.H_b(x_b)

        # average forward/backward heatmaps
        heatmaps = (x_f + x_b) / 2

        # softmax temp stays 1; to modify for model predictions, see HeatmapMHCRNNHead constructor
        return spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))
