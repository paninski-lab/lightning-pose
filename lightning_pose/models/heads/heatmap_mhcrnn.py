# """Heads that produce heatmap predictions for heatmap regression."""
#
# from typing import Tuple
# from typing_extensions import Literal
#
# import torch
# from kornia.filters import filter2d
# from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d
# from kornia.geometry.transform.pyramid import _get_pyramid_gaussian_kernel
# from lightning.pytorch import LightningModule
# from torch import nn
# from torchtyping import TensorType
#
# from lightning_pose.data.utils import evaluate_heatmaps_at_location
#
# # to ignore imports for sphix-autoapidoc
# __all__ = [
#     "HeatmapMHCRNNHead",
#     "UpsamplingCRNN",
# ]
#
#
# class HeatmapMHCRNNHead(LightningModule):
#     """Multi-head convolutional recurrent neural network head.
#
#     This head converts a sequence of 2D feature maps to per-keypoint heatmaps for the center frame.
#     The head is composed of two heads:
#     - single frame head: several deconvolutional layers followed by a 2D spatial softmax to
#       generate normalized heatmaps from low-resolution feature maps for a single frame.
#     - multi-frame head: several deconvolutional layers are applied to each set of features in a
#       temporal sequence; the resulting heatmaps are fed into a convolutional recurrent neural
#       network to produce heatmaps for the center frame
#
#     """
#
#     def __init__(
#         self,
#         backbone_arch: str,
#         in_channels: int,
#         out_channels: int,
#         deconv_out_channels: int | None = None,
#         downsample_factor: int = 2,
#     ):
#         """
#
#         Args:
#             backbone_arch: string denoting backbone architecture; to remove in future release
#             in_channels: number of channels in the input feature map
#             out_channels: number of channels in the output heatmap (i.e. number of keypoints)
#             deconv_out_channels: output channel number for each intermediate deconv layer; defaults
#                 to number of keypoints
#             downsample_factor: make heatmaps smaller than input frames by this factor; subpixel
#                 operations are performed for increased precision
#
#         """
#         super().__init__()
#
#         self.backbone_arch = backbone_arch
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.deconv_out_channels = deconv_out_channels
#         self.downsample_factor = downsample_factor
#
#         self.upsampling_layers = self._make_upsampling_layers(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             int_channels=deconv_out_channels or out_channels,
#         )
#         self._initialize_upsampling_layers()
#
#         # TODO: temp=1000 works for 64x64 heatmaps, need to generalize to other shapes
#         self.temperature = torch.tensor(1000.0, device=self.device)  # soft argmax temp
#
#     def _make_upsampling_layers(
#         self,
#         in_channels: int,
#         out_channels: int,
#         int_channels: int,
#     ) -> torch.nn.Sequential:
#         # Note:
#         # https://github.com/jgraving/DeepPoseKit/blob/
#         # cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/DeepLabCut.py#L131
#         # in their model, the pixel shuffle happens only for downsample_factor=2
#
#         n_layers_to_build = 4 - self.downsample_factor
#         if self.backbone_arch in ["vit_h_sam", "vit_b_sam"]:
#             n_layers_to_build = -1
#
#         upsampling_layers = []
#         upsampling_layers.append(nn.PixelShuffle(2))
#         for layer in range(n_layers_to_build):
#
#             if layer == 0:
#                 in_ = in_channels // 4  # division by 4 to account for PixelShuffle layer
#                 out_ = int_channels
#             elif layer == n_layers_to_build - 1:
#                 in_ = int_channels if n_layers_to_build > 1 else in_channels // 4
#                 out_ = out_channels
#             else:
#                 in_ = int_channels
#                 out_ = int_channels
#
#             upsampling_layers.append(
#                 nn.ConvTranspose2d(
#                     in_channels=in_,
#                     out_channels=out_,
#                     kernel_size=(3, 3),
#                     stride=(2, 2),
#                     padding=(1, 1),
#                     output_padding=(1, 1),
#                 )
#             )
#
#         return nn.Sequential(*upsampling_layers)
#
#     def _initialize_upsampling_layers(self) -> None:
#         """Intialize the Conv2DTranspose upsampling layers."""
#         for index, layer in enumerate(self.upsampling_layers):
#             if index > 0:  # we ignore the PixelShuffle
#                 if isinstance(layer, nn.ConvTranspose2d):
#                     torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
#                     torch.nn.init.zeros_(layer.bias)
#                 elif isinstance(layer, nn.BatchNorm2d):
#                     torch.nn.init.constant_(layer.weight, 1.0)
#                     torch.nn.init.constant_(layer.bias, 0.0)
#
#     def forward(
#         self,
#         features: TensorType["batch", "features", "features_height", "features_width"],
#     ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
#         """Upsample representations and normalize to get final heatmaps."""
#         heatmaps = self.upsampling_layers(features)
#         # softmax temp stays 1 here; to modify for model predictions, see constructor
#         return spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))
#
#     def run_subpixelmaxima(
#         self,
#         heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
#     ) -> Tuple[TensorType["batch", "num_targets"], TensorType["batch", "num_keypoints"]]:
#         """Use soft argmax on heatmaps.
#
#         Args:
#             heatmaps: output of upsampling layers
#
#         Returns:
#             tuple
#                 - soft argmax of shape (batch, num_targets)
#                 - confidences of shape (batch, num_keypoints)
#
#         """
#
#         # upsample heatmaps
#         for _ in range(self.downsample_factor):
#             heatmaps = upsample(heatmaps)
#         # find soft argmax
#         softmaxes = spatial_softmax2d(heatmaps, temperature=self.temperature)
#         preds = spatial_expectation2d(softmaxes, normalized_coordinates=False)
#         # compute confidences as softmax value pooled around prediction
#         confidences = evaluate_heatmaps_at_location(heatmaps=softmaxes, locs=preds)
#         # fix grid offsets from upsampling
#         if self.downsample_factor == 1:
#             preds -= 0.5
#         elif self.downsample_factor == 2:
#             preds -= 1.5
#         elif self.downsample_factor == 3:
#             preds -= 2.5
#
#         # NOTE: we cannot use
#         # `preds.reshape(-1, self.num_targets)`
#         # This works fine for the non-multiview case
#         # This works fine for multiview training
#         # This fails during multiview inference when we might have an arbitrary number of views
#         # that we are processing (self.num_targets is tied to the labeled data)
#         return preds.reshape(-1, heatmaps.shape[1] * 2), confidences
#
#
# class UpsamplingCRNN(torch.nn.Module):
#     """Bidirectional Convolutional RNN network that handles heatmaps of context frames.
#
#     The input to the CRNN is a set of heatmaps at times t-k, ..., t, ...t+k, one heatmap for each
#     timepoint/keypoint
#
#     The output of the CRNN is a single heatmap for each keypoint
#
#     """
#
#     def __init__(
#         self,
#         num_filters_for_upsampling: int,
#         num_keypoints: int,
#         upsampling_factor: Literal[1, 2] = 2,
#         hkernel: int = 2,
#         hstride: int = 2,
#         hpad: int = 0,
#         nfilters_channel: int = 16,
#     ) -> None:
#         """Upsampling Convolutional RNN - initialize input and hidden weights."""
#
#         super().__init__()
#         self.upsampling_factor = upsampling_factor
#         self.pixel_shuffle = nn.PixelShuffle(2)
#         if self.upsampling_factor == 2:
#             self.W_pre = HeatmapTracker.create_double_upsampling_layer(
#                 in_channels=num_filters_for_upsampling // 4,
#                 out_channels=num_keypoints,
#             )
#             in_channels_rnn = num_keypoints
#         else:
#             in_channels_rnn = num_filters_for_upsampling // 4
#
#         self.W_f = HeatmapTracker.create_double_upsampling_layer(
#             in_channels=in_channels_rnn,
#             out_channels=num_keypoints,
#         )
#         H_f_layers = []
#         H_f_layers.append(
#             nn.Conv2d(
#                 in_channels=num_keypoints,
#                 out_channels=num_keypoints * nfilters_channel,
#                 kernel_size=(hkernel, hkernel),
#                 stride=(hstride, hstride),
#                 padding=(hpad, hpad),
#                 groups=num_keypoints,
#             )
#         )
#         H_f_layers.append(
#             nn.ConvTranspose2d(
#                 in_channels=num_keypoints * nfilters_channel,
#                 out_channels=num_keypoints,
#                 kernel_size=(hkernel, hkernel),
#                 stride=(hstride, hstride),
#                 padding=(hpad, hpad),
#                 output_padding=(hpad, hpad),
#                 groups=num_keypoints,
#             )
#         )
#         self.H_f = nn.Sequential(*H_f_layers)
#
#         self.W_b = HeatmapTracker.create_double_upsampling_layer(
#             in_channels=in_channels_rnn,
#             out_channels=num_keypoints,
#         )
#         H_b_layers = []
#         H_b_layers.append(
#             nn.Conv2d(
#                 in_channels=num_keypoints,
#                 out_channels=num_keypoints * nfilters_channel,
#                 kernel_size=(hkernel, hkernel),
#                 stride=(hstride, hstride),
#                 padding=(hpad, hpad),
#                 groups=num_keypoints,
#             )
#         )
#         H_b_layers.append(
#             nn.ConvTranspose2d(
#                 in_channels=num_keypoints * nfilters_channel,
#                 out_channels=num_keypoints,
#                 kernel_size=(hkernel, hkernel),
#                 stride=(hstride, hstride),
#                 padding=(hpad, hpad),
#                 output_padding=(hpad, hpad),
#                 groups=num_keypoints,
#             )
#         )
#         self.H_b = nn.Sequential(*H_b_layers)
#         self.initialize_layers()
#         if self.upsampling_factor == 2:
#             self.layers = torch.nn.ModuleList([self.W_pre, self.W_f, self.H_f, self.W_b, self.H_b])
#         else:
#             self.layers = torch.nn.ModuleList([self.W_f, self.H_f, self.W_b, self.H_b])
#
#     def initialize_layers(self):
#         if self.upsampling_factor == 2:
#             torch.nn.init.xavier_uniform_(self.W_pre.weight, gain=1.0)
#             torch.nn.init.zeros_(self.W_pre.bias)
#
#         torch.nn.init.xavier_uniform_(self.W_f.weight, gain=1.0)
#         torch.nn.init.zeros_(self.W_f.bias)
#         for index, layer in enumerate(self.H_f):
#             torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
#             torch.nn.init.zeros_(layer.bias)
#
#         torch.nn.init.xavier_uniform_(self.W_b.weight, gain=1.0)
#         torch.nn.init.zeros_(self.W_b.bias)
#         for index, layer in enumerate(self.H_b):
#             torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
#             torch.nn.init.zeros_(layer.bias)
#
#     def forward(
#         self,
#         representations: TensorType["frames", "batch", "features", "rep_height", "rep_width"]
#     ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
#
#         # expand representations in spatial domain using pixel shuffle to create heatmaps
#         if self.upsampling_factor == 2:
#             # upsample once more before passing through RNN
#             # need to reshape to push through convolution layers
#             frames, batch, features, rep_height, rep_width = representations.shape
#             frames_batch_shape = batch * frames
#             representations_batch_frames: TensorType[
#                 "batch*frames", "features", "rep_height", "rep_width"
#             ] = representations.reshape(frames_batch_shape, features, rep_height, rep_width)
#             x_tensor = self.W_pre(self.pixel_shuffle(representations_batch_frames))
#             x_tensor = x_tensor.reshape(
#                 frames,
#                 batch,
#                 x_tensor.shape[1],
#                 x_tensor.shape[2],
#                 x_tensor.shape[3],
#             )
#         else:
#             x_tensor = self.pixel_shuffle(representations)
#
#         # push heatmaps through CRNN
#         x_f = self.W_f(x_tensor[0])
#         for frame_batch in x_tensor[1:]:  # forward pass
#             x_f = self.W_f(frame_batch) + self.H_f(x_f)
#         x_tensor_b = torch.flip(x_tensor, dims=[0])
#         x_b = self.W_b(x_tensor_b[0])
#         for frame_batch in x_tensor_b[1:]:  # backwards pass
#             x_b = self.W_b(frame_batch) + self.H_b(x_b)
#
#         # average forward/backward heatmaps
#         heatmaps = (x_f + x_b) / 2
#
#         return heatmaps
