"""Tests for the nonlinear heatmap head."""

import torch

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_nonlinear import HeatmapNonlinearHead, LayerNorm2d


class TestHeatmapNonlinearHead:
    """Test the class HeatmapNonlinearHead."""

    def _head(self, **kwargs) -> HeatmapNonlinearHead:
        return HeatmapNonlinearHead(
            backbone_arch='vitb_imagenet',
            in_channels=128,
            out_channels=17,
            downsample_factor=2,
            **kwargs,
        )

    def test_nonlinear_head_readout_layers(self):
        head = self._head(hidden_channels=32)
        layers = list(head.upsampling_layers)
        assert isinstance(layers[-3], LayerNorm2d)
        assert isinstance(layers[-2], torch.nn.ReLU)
        assert isinstance(layers[-1], torch.nn.Conv2d)
        assert layers[-1].in_channels == 32 and layers[-1].out_channels == 17
        # the last transposed convolution feeds the readout
        assert layers[-4].out_channels == 32
        assert head.out_channels == 17
        assert head.hidden_channels == 32

    def test_nonlinear_head_default_hidden_channels(self):
        head = self._head()
        assert list(head.upsampling_layers)[-1].in_channels == 256

    def test_nonlinear_head_output_matches_linear_head_shape(self):
        head = self._head(hidden_channels=32)
        linear = HeatmapHead(
            backbone_arch='vitb_imagenet', in_channels=128, out_channels=17, downsample_factor=2,
        )
        features = torch.randn(2, 128, 8, 8)
        heatmaps = head(features)
        assert heatmaps.shape == linear(features).shape
        # spatial softmax: every heatmap sums to one
        assert torch.allclose(heatmaps.sum(dim=(2, 3)), torch.ones(2, 17), atol=1e-5)

    def test_nonlinear_head_state_dict_keys(self):
        # parameter names are part of the checkpoint format: the readout follows the
        # transposed convolutions inside ``upsampling_layers``
        head = self._head(hidden_channels=32)
        n_deconv = len(head.upsampling_layers) - 3
        keys = set(head.state_dict())
        assert f'upsampling_layers.{n_deconv}.norm.weight' in keys
        assert f'upsampling_layers.{n_deconv}.norm.bias' in keys
        assert f'upsampling_layers.{n_deconv + 2}.weight' in keys
        assert f'upsampling_layers.{n_deconv + 2}.bias' in keys

    def test_nonlinear_head_init(self):
        head = self._head(hidden_channels=32)
        readout = list(head.upsampling_layers)[-1]
        assert torch.count_nonzero(readout.bias) == 0
        # small init, like the transposed convolutions of the linear head
        assert readout.weight.abs().max() < 0.1


class TestLayerNorm2d:
    """Test the class LayerNorm2d."""

    def test_layer_norm_2d_normalizes_channels_per_pixel(self):
        norm = LayerNorm2d(8)
        out = norm(torch.randn(2, 8, 5, 5) * 3 + 1)
        assert out.shape == (2, 8, 5, 5)
        assert torch.allclose(out.mean(dim=1), torch.zeros(2, 5, 5), atol=1e-5)
