import pytest
import torch
import torch.nn as nn

from lightning_pose.models.heads.heatmap import (
    make_upsampling_layers,
    upsample,
)


class TestMakeUpsamplingLayers:

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_basic_parameters(self, n_layers):
        """Test with different numbers of layers."""
        model = make_upsampling_layers(
            in_channels=256,
            out_channels=64,
            int_channels=128,
            n_layers=n_layers,
        )

        assert isinstance(model, nn.Sequential)
        # should have 1 PixelShuffle + n_layers ConvTranspose2d
        assert len(model) == n_layers + 1
        assert isinstance(model[0], nn.PixelShuffle)
        for i in range(1, len(model)):
            assert isinstance(model[i], nn.ConvTranspose2d)

        # first ConvTranspose2d should take in_channels // 4
        assert model[1].in_channels == 256 // 4
        if n_layers > 1:
            assert model[1].out_channels == 128

        # last ConvTranspose2d should output out_channels
        assert model[-1].out_channels == 64
        if n_layers > 1:
            assert model[-1].in_channels == 128

    def test_conv_parameters(self):
        """Test that ConvTranspose2d layers have correct parameters."""
        model = make_upsampling_layers(
            in_channels=256,
            out_channels=64,
            int_channels=128,
            n_layers=2,
        )

        for i in range(1, len(model)):
            conv = model[i]
            assert conv.kernel_size == (3, 3)
            assert conv.stride == (2, 2)
            assert conv.padding == (1, 1)
            assert conv.output_padding == (1, 1)

    def test_forward_pass(self):
        """Test that the model can process input tensors correctly."""
        batch_size = 2
        in_channels = 256
        h, w = 8, 8

        model = make_upsampling_layers(
            in_channels=in_channels,
            out_channels=64,
            int_channels=128,
            n_layers=3
        )

        x = torch.randn(batch_size, in_channels, h, w)
        output = model(x)

        # PixelShuffle reduces channels by 4 and doubles spatial dims
        # Then each ConvTranspose2d doubles spatial dims
        # Total upsampling: 2 * 2^3 = 16x
        expected_h = h * 2 * (2 ** 3)  # 128
        expected_w = w * 2 * (2 ** 3)  # 128

        assert output.shape == (batch_size, 64, expected_h, expected_w)


class TestUpsample:

    def test_shapes(self):

        batch = 4
        num_keypoints = 2
        height = 8
        width = 16

        data = torch.rand((batch, num_keypoints, height, width))

        data_upsampled = upsample(data)

        assert data_upsampled.shape == (batch, num_keypoints, 2 * height, 2 * width)

    def test_align_corners(self):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 2
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        inputs[0, 0, 2, 2] = 1.0  # peak at (2, 2)
        inputs[0, 1, 6, 6] = 1.0  # peak at (6, 6)

        result = upsample(inputs)

        # with align_corners=False and 2x upsampling:
        # input pixel at (i, j) maps to output pixel at (2*i, 2*j)
        # verify peaks are near expected locations (with tolerance for interpolation)
        peak_0_y, peak_0_x = torch.where(result[0, 0] == result[0, 0].max())
        peak_1_y, peak_1_x = torch.where(result[0, 1] == result[0, 1].max())

        assert abs(peak_0_y[0].item() - 4) <= 1  # 2 * 2 = 4
        assert abs(peak_0_x[0].item() - 4) <= 1
        assert abs(peak_1_y[0].item() - 12) <= 1  # 6 * 2 = 12
        assert abs(peak_1_x[0].item() - 12) <= 1
