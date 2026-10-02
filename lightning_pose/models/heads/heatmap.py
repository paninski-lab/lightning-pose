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


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension at every pixel of a (B, C, H, W) map (ConvNeXt style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class BlockLayerNorm2d(LayerNorm2d):
    """LayerNorm2d whose statistics are computed separately in contiguous channel blocks.

    The affine parameters stay in ``self.norm`` (same state-dict keys and shapes as
    :class:`LayerNorm2d`); only the mean and variance are per block, so no block's output depends
    on another block's channels. With a single block this is exactly :class:`LayerNorm2d`.
    """

    def __init__(self, num_channels: int, block_sizes: list[int], eps: float = 1e-6) -> None:
        """Initialize the block LayerNorm.

        Args:
            num_channels: total number of channels C.
            block_sizes: sizes of the contiguous channel blocks, in channel order; must sum to C.
            eps: value added to the variance for numerical stability.

        Raises:
            ValueError: if ``block_sizes`` does not partition ``num_channels``.
        """
        super().__init__(num_channels, eps=eps)
        if not block_sizes or min(block_sizes) < 1 or sum(block_sizes) != num_channels:
            raise ValueError(
                f'block_sizes {block_sizes} must be positive and sum to {num_channels}'
            )
        self.block_sizes = list(block_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each channel block at every pixel, then apply the shared affine."""
        if len(self.block_sizes) == 1:
            # one block is plain LayerNorm2d; take its fused path so outputs match bit for bit
            return super().forward(x)
        x = x.permute(0, 2, 3, 1)
        blocks = torch.split(x, self.block_sizes, dim=-1)
        x = torch.cat(
            [nn.functional.layer_norm(block, (block.shape[-1],), eps=self.norm.eps)
             for block in blocks],
            dim=-1,
        )
        x = x * self.norm.weight + self.norm.bias
        return x.permute(0, 3, 1, 2)


class MaskedConv2d(nn.Conv2d):
    """Conv2d whose weight is multiplied by a fixed 0/1 connectivity mask in every forward.

    The mask is a non-persistent buffer (rebuilt from the constructor arguments, absent from
    checkpoints), so the state dict has the same keys and shapes as a plain ``nn.Conv2d``. Masked
    weights receive zero gradient; call :meth:`zero_masked_weights` after initialization so
    checkpoints hold exact zeros there.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask: torch.Tensor,
        kernel_size: int = 1,
    ) -> None:
        """Initialize the masked convolution.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            mask: (out_channels, in_channels) 0/1 tensor; ``mask[o, i] = 1`` lets output ``o``
                read input ``i``.
            kernel_size: convolution kernel size (the mask is broadcast over it).

        Raises:
            ValueError: if ``mask`` does not have shape (out_channels, in_channels).
        """
        super().__init__(in_channels, out_channels, kernel_size=kernel_size)
        if tuple(mask.shape) != (out_channels, in_channels):
            raise ValueError(
                f'mask shape {tuple(mask.shape)} != (out_channels, in_channels) = '
                f'{(out_channels, in_channels)}'
            )
        self.register_buffer(
            'mask', mask.to(self.weight.dtype)[:, :, None, None], persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve with the masked weight."""
        return self._conv_forward(x, self.weight * self.mask, self.bias)

    @torch.no_grad()
    def zero_masked_weights(self) -> None:
        """Set the masked-out weight entries to exactly zero (in place)."""
        self.weight.mul_(self.mask)


def make_group_mask(
    groups: list[int],
    hidden_channels: int,
    shared_channels: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """Build the keypoint -> hidden-channel connectivity of a grouped nonlinear head.

    Hidden channels are laid out as ``[shared | group 0 | group 1 | ...]``: the first
    ``shared_channels`` are read by every keypoint, and the remaining channels split equally into
    one contiguous block per group.

    Args:
        groups: group index (0..G-1) of every output keypoint; every index must occur.
        hidden_channels: total number of hidden channels H.
        shared_channels: number S of hidden channels read by every keypoint.

    Returns:
        tuple
            - (num_keypoints, H) 0/1 float mask
            - channel block sizes in channel order: ``[S] + [(H - S) / G] * G`` (no shared
              block when S is 0)

    Raises:
        ValueError: if the group indices are not 0..G-1, S is outside [0, H), or H - S does not
            split equally into G blocks.
    """
    num_groups = max(groups) + 1 if groups else 0
    if not groups or min(groups) < 0 or set(groups) != set(range(num_groups)):
        raise ValueError(f'groups must use every index 0..G-1 and only those, got {groups}')
    if not 0 <= shared_channels < hidden_channels:
        raise ValueError(
            f'shared_channels must be in [0, {hidden_channels}), got {shared_channels}'
        )
    if (hidden_channels - shared_channels) % num_groups:
        raise ValueError(
            f'hidden_channels - shared_channels = {hidden_channels - shared_channels} does not '
            f'split equally into {num_groups} groups'
        )
    block = (hidden_channels - shared_channels) // num_groups
    mask = torch.zeros(len(groups), hidden_channels)
    mask[:, :shared_channels] = 1.0
    for idx_keypoint, idx_group in enumerate(groups):
        start = shared_channels + idx_group * block
        mask[idx_keypoint, start:start + block] = 1.0
    block_sizes = ([shared_channels] if shared_channels else []) + [block] * num_groups
    return mask, block_sizes


def initialize_upsampling_layers(layers: nn.Sequential) -> None:
    """Intialize the Conv2DTranspose upsampling layers (and the optional nonlinear readout)."""
    for index, layer in enumerate(layers):
        if index > 0:  # we ignore the PixelShuffle
            if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                torch.nn.init.zeros_(layer.bias)  # type: ignore[arg-type]
            elif isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1.0)
                torch.nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, LayerNorm2d):
                torch.nn.init.constant_(layer.norm.weight, 1.0)
                torch.nn.init.constant_(layer.norm.bias, 0.0)


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
        hidden_channels: int | None = None,
        groups: list[int] | None = None,
        shared_channels: int = 0,
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
            hidden_channels: None = stock linear head. An int H makes the head nonlinear: the
                last ConvTranspose2d outputs H channels, then LayerNorm over channels -> ReLU ->
                Conv2d 1x1 (H -> out_channels), i.e. a per-pixel MLP readout before the softmax.
            groups: None = every keypoint reads all H hidden channels. A list gives the group
                index (0..G-1) of every output keypoint and routes keypoints to hidden channels
                by group (requires ``hidden_channels``): the first ``shared_channels`` hidden
                channels are read by every keypoint, the rest split into one contiguous block
                per group that only that group's keypoints read. The LayerNorm statistics are
                computed per block (:class:`BlockLayerNorm2d`) and the 1x1 readout is masked
                (:class:`MaskedConv2d`), so a keypoint's loss reaches only its own block and the
                shared block of the last ConvTranspose2d. Layer layout, state-dict keys and
                shapes equal the ungrouped nonlinear head's.
            shared_channels: number of hidden channels read by every keypoint (with ``groups``).

        Raises:
            ValueError: if ``groups`` is given without ``hidden_channels`` or has the wrong
                length, ``shared_channels`` is given without ``groups``, or the channels do not
                split (see :func:`make_group_mask`).

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

        if groups is not None and hidden_channels is None:
            raise ValueError('groups require hidden_channels (a nonlinear head)')
        if groups is None and shared_channels:
            raise ValueError('shared_channels requires groups')
        if groups is not None and len(groups) != out_channels:
            raise ValueError(
                f'groups has {len(groups)} entries, expected one per keypoint ({out_channels})'
            )
        self.hidden_channels = hidden_channels
        self.groups = None if groups is None else [int(g) for g in groups]
        self.shared_channels = int(shared_channels)
        if hidden_channels is None:
            self.upsampling_layers = make_upsampling_layers(
                in_channels=in_channels,
                out_channels=out_channels,
                int_channels=deconv_out_channels or out_channels,
                n_layers=n_layers,
            )
        else:
            layers = make_upsampling_layers(
                in_channels=in_channels,
                out_channels=int(hidden_channels),
                int_channels=deconv_out_channels or int(hidden_channels),
                n_layers=n_layers,
            )
            # norm and readout are built in the same order with or without groups (only the
            # conv draws random numbers), so a given seed yields the same initial tensors
            if self.groups is None:
                norm = LayerNorm2d(int(hidden_channels))
                readout = nn.Conv2d(int(hidden_channels), out_channels, kernel_size=1)
            else:
                mask, block_sizes = make_group_mask(
                    self.groups, int(hidden_channels), self.shared_channels,
                )
                norm = BlockLayerNorm2d(int(hidden_channels), block_sizes)
                readout = MaskedConv2d(int(hidden_channels), out_channels, mask=mask)
            self.upsampling_layers = nn.Sequential(
                *layers,
                norm,
                nn.ReLU(inplace=True),
                readout,
            )
        initialize_upsampling_layers(self.upsampling_layers)
        if self.groups is not None:
            # masked weights get zero gradient (and stay zero under Adam); zero them now so
            # checkpoints hold exact zeros outside each keypoint's channels
            self.upsampling_layers[-1].zero_masked_weights()

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
