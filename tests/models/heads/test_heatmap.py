import numpy as np
import pytest
import torch
import torch.nn as nn

from lightning_pose.models.heads.heatmap import (
    HeatmapHead,
    make_upsampling_layers,
    run_subpixelmaxima,
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


class TestRunSubpixelMaxima:

    @pytest.mark.parametrize("downsample_factor", [1, 2])
    def test_interior_points(self, downsample_factor):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 2
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        val_0 = 2
        inputs[0, 0, val_0, val_0] = 1.0  # peak at (2, 2)
        val_1 = 4
        inputs[0, 1, val_1, val_1] = 1.0  # peak at (4, 4)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=downsample_factor, temperature=torch.tensor(1000.0),
        )

        xy = val_0 * (2 ** downsample_factor)
        assert torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([float(xy), float(xy)]), dtype=preds.dtype),
        )
        assert confidences[0, 0] == torch.tensor(1.0)

        xy = val_1 * (2 ** downsample_factor)
        assert torch.allclose(
            preds[0, 2:4],
            torch.tensor(np.array([float(xy), float(xy)]), dtype=preds.dtype),
        )
        assert confidences[0, 1] == torch.tensor(1.0)

    @pytest.mark.parametrize("downsample_factor", [1, 2])
    def test_boundary_points(self, downsample_factor):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 2
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        inputs[0, 0, 0, 0] = 1.0  # peak at (0, 0) (corner)
        inputs[0, 1, 0, 1] = 1.0  # peak at (0, 1) (edge)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=downsample_factor, temperature=torch.tensor(1000.0),
        )

        assert torch.isclose(preds[0, 0], torch.tensor(0.), atol=0.5)
        assert torch.isclose(preds[0, 1], torch.tensor(0.), atol=0.5)
        assert torch.isclose(confidences[0, 0], torch.tensor(1.0), rtol=1e-3)

        assert torch.isclose(preds[0, 2], torch.tensor(float(2 ** downsample_factor)), rtol=1e-1)
        assert torch.isclose(preds[0, 3], torch.tensor(0.), atol=0.5)
        assert torch.isclose(confidences[0, 1], torch.tensor(1.0), rtol=1e-3)

    def test_temperature(self):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 1
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        inputs[0, 0, 4, 4] = 1.0  # peak at (0, 0)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=2, temperature=torch.tensor(1000.0),
        )
        assert torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([16.0, 16.0]), dtype=preds.dtype),
        )
        assert confidences[0, 0] == torch.tensor(1.0)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=2, temperature=torch.tensor(100.0),
        )
        assert torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([16.0, 16.0]), dtype=preds.dtype),
        )
        assert confidences[0, 0] != torch.tensor(1.0)
        assert torch.isclose(confidences[0, 0], torch.tensor(1.0), 1e-3)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=2, temperature=torch.tensor(10.0),
        )
        assert not torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([16.0, 16.0]), dtype=preds.dtype),
        )
        assert not torch.isclose(confidences[0, 0], torch.tensor(1.0), 1e-2)
        assert confidences[0, 0] < torch.tensor(0.5)


class TestHeatmapHead:

    @pytest.fixture
    def basic_head(self):
        """Create a basic HeatmapHead instance for testing."""
        return HeatmapHead(
            backbone_arch="resnet50",
            in_channels=256,
            out_channels=17,
            deconv_out_channels=32,
            downsample_factor=2,
            final_softmax=True,
        )

    def test_initialization(self, basic_head):
        """Test that HeatmapHead initializes with correct attributes."""
        assert basic_head.backbone_arch == "resnet50"
        assert basic_head.in_channels == 256
        assert basic_head.out_channels == 17
        assert basic_head.deconv_out_channels == 32
        assert basic_head.downsample_factor == 2
        assert basic_head.final_softmax is True
        assert isinstance(basic_head.temperature, torch.Tensor)
        assert basic_head.temperature.item() == 1000.0

    def test_forward_shape(self, basic_head):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        features_h, features_w = 16, 16
        features = torch.randn(batch_size, 256, features_h, features_w)

        output = basic_head(features)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 17  # out_channels
        assert output.shape[2] == features_h * (2 ** (basic_head.downsample_factor + 1))
        assert output.shape[3] == features_w * (2 ** (basic_head.downsample_factor + 1))

    def test_forward_with_softmax(self, basic_head):
        """Test that output is normalized when final_softmax=True."""

        features = torch.randn(1, 256, 8, 8)
        output = basic_head(features)

        # check that values are in valid probability range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

        # check that each heatmap approximately sums to 1 (spatial softmax)
        for i in range(output.shape[1]):
            heatmap_sum = output[0, i].sum()
            assert torch.isclose(heatmap_sum, torch.tensor(1.0), atol=1e-5)

    def test_forward_without_softmax(self):
        """Test that output is not normalized when final_softmax=False."""
        head = HeatmapHead(
            backbone_arch="resnet50",
            in_channels=128,
            out_channels=10,
            downsample_factor=2,
            final_softmax=False,
        )

        features = torch.randn(1, 128, 8, 8)
        output = head(features)

        # check that each heatmap does not approximately sum to 1
        for i in range(output.shape[1]):
            heatmap_sum = output[0, i].sum()
            assert not torch.isclose(heatmap_sum, torch.tensor(1.0), atol=1e-5)

    def test_different_downsample_factors(self):
        """Test initialization with different downsample factors."""
        for downsample_factor in [1, 2, 3]:
            head = HeatmapHead(
                backbone_arch="resnet50",
                in_channels=256,
                out_channels=17,
                downsample_factor=downsample_factor,
                final_softmax=True,
            )
            assert head.downsample_factor == downsample_factor
            assert len(head.upsampling_layers) == 4 - downsample_factor + 1  # +1 for PixelShuffle

    def test_vit_backbone_architecture(self):
        """Test that ViT backbone reduces number of layers by 1."""
        head_vit = HeatmapHead(
            backbone_arch="vitb_imagenet",
            in_channels=128,
            out_channels=17,
            downsample_factor=2,
            final_softmax=True,
        )

        head_resnet = HeatmapHead(
            backbone_arch="resnet50",
            in_channels=128,
            out_channels=17,
            downsample_factor=2,
            final_softmax=True,
        )

        # ViT should have one fewer layer
        assert len(head_vit.upsampling_layers) == len(head_resnet.upsampling_layers) - 1
