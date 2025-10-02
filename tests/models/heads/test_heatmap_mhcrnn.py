import pytest
import torch

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_mhcrnn import (
    HeatmapMHCRNNHead,
    UpsamplingCRNN,
)


class TestHeatmapMHCRNNHead:

    @pytest.fixture
    def basic_mhcrnn_head(self):
        """Create a basic HeatmapMHCRNNHead instance for testing."""
        return HeatmapMHCRNNHead(
            backbone_arch="resnet50",
            in_channels=256,
            out_channels=17,
            deconv_out_channels=32,
            downsample_factor=2,
            upsampling_factor=2,
        )

    def test_initialization(self, basic_mhcrnn_head):
        """Test that HeatmapMHCRNNHead initializes with correct attributes."""
        assert basic_mhcrnn_head.backbone_arch == "resnet50"
        assert basic_mhcrnn_head.in_channels == 256
        assert basic_mhcrnn_head.out_channels == 17
        assert basic_mhcrnn_head.deconv_out_channels == 32
        assert basic_mhcrnn_head.downsample_factor == 2
        assert basic_mhcrnn_head.upsampling_factor == 2
        assert isinstance(basic_mhcrnn_head.temperature, torch.Tensor)
        assert basic_mhcrnn_head.temperature.item() == 1000.0

        # test that single frame head is created
        assert hasattr(basic_mhcrnn_head, 'head_sf')
        assert isinstance(basic_mhcrnn_head.head_sf, HeatmapHead)
        assert basic_mhcrnn_head.head_sf.in_channels == 256
        assert basic_mhcrnn_head.head_sf.out_channels == 17

        # test that multi-frame head is created
        assert hasattr(basic_mhcrnn_head, 'head_mf')
        assert isinstance(basic_mhcrnn_head.head_mf, UpsamplingCRNN)

    def test_forward_shape_labeled(self, basic_mhcrnn_head):
        """Test forward pass with labeled batch."""
        batch_size = 4
        num_frames = 5
        features_h, features_w = 16, 16
        image_h, image_w = 32, 32

        features = torch.randn(batch_size, 256, features_h, features_w, num_frames)
        batch_shape = torch.tensor([batch_size, num_frames, 3, image_h, image_w])

        heatmaps_sf, heatmaps_mf = basic_mhcrnn_head(features, batch_shape, is_multiview=False)

        assert len(heatmaps_sf.shape) == 4
        assert heatmaps_sf.shape[0] == batch_size
        assert heatmaps_sf.shape[1] == basic_mhcrnn_head.out_channels
        assert len(heatmaps_mf.shape) == 4
        assert heatmaps_mf.shape[0] == batch_size
        assert heatmaps_mf.shape[1] == basic_mhcrnn_head.out_channels

    def test_forward_shape_unlabeled(self, basic_mhcrnn_head):
        """Test forward pass with unlabeled batch."""
        seq_length = 6
        num_frames = 5
        features_h, features_w = 16, 16
        image_h, image_w = 32, 32

        features = torch.randn(seq_length, 256, features_h, features_w, num_frames)
        batch_shape = torch.tensor([seq_length, 3, image_h, image_w])

        heatmaps_sf, heatmaps_mf = basic_mhcrnn_head(features, batch_shape, is_multiview=False)

        assert len(heatmaps_sf.shape) == 4
        assert heatmaps_sf.shape[0] == seq_length
        assert heatmaps_sf.shape[1] == basic_mhcrnn_head.out_channels
        assert len(heatmaps_mf.shape) == 4
        assert heatmaps_mf.shape[0] == seq_length
        assert heatmaps_mf.shape[1] == basic_mhcrnn_head.out_channels

    def test_forward_shape_multiview_labeled(self, basic_mhcrnn_head):
        """Test forward pass with labeled multiview batch."""
        batch_size = 4
        num_frames = 5
        num_views = 2
        features_h, features_w = 16, 16
        image_h, image_w = 32, 32

        features = torch.randn(batch_size * num_views, 256, features_h, features_w, num_frames)
        batch_shape = torch.tensor([batch_size, num_views, 5, 3, image_h, image_w])

        heatmaps_sf, heatmaps_mf = basic_mhcrnn_head(features, batch_shape, is_multiview=True)

        assert len(heatmaps_sf.shape) == 4
        assert heatmaps_sf.shape[0] == batch_size
        assert heatmaps_sf.shape[1] == basic_mhcrnn_head.out_channels * num_views
        assert len(heatmaps_mf.shape) == 4
        assert heatmaps_mf.shape[0] == batch_size
        assert heatmaps_mf.shape[1] == basic_mhcrnn_head.out_channels * num_views

    def test_forward_shape_multiview_unlabeled(self, basic_mhcrnn_head):
        """Test forward pass with unlabeled multiview data."""
        seq_length = 6
        num_frames = 5
        num_views = 2
        features_h, features_w = 16, 16
        image_h, image_w = 32, 32

        features = torch.randn(seq_length * num_views, 256, features_h, features_w, num_frames)
        batch_shape = torch.tensor([seq_length + 4, num_views, 3, image_h, image_w])

        heatmaps_sf, heatmaps_mf = basic_mhcrnn_head(features, batch_shape, is_multiview=True)

        assert len(heatmaps_sf.shape) == 4
        assert heatmaps_sf.shape[0] == seq_length
        assert heatmaps_sf.shape[1] == basic_mhcrnn_head.out_channels * num_views
        assert len(heatmaps_mf.shape) == 4
        assert heatmaps_mf.shape[0] == seq_length
        assert heatmaps_mf.shape[1] == basic_mhcrnn_head.out_channels * num_views

    def test_vit_backbone(self):
        """Test with ViT backbone architecture."""
        head = HeatmapMHCRNNHead(
            backbone_arch="vitb_imagenet",
            in_channels=768,
            out_channels=17,
            downsample_factor=2,
            upsampling_factor=2,
        )
        assert head.backbone_arch == "vitb_imagenet"
        assert head.head_sf.backbone_arch == "vitb_imagenet"


class TestUpsamplingCRNN:

    @pytest.fixture
    def crnn_up2(self):
        return UpsamplingCRNN(
            num_filters_for_upsampling=256,
            num_keypoints=17,
            upsampling_factor=2,
        )

    @pytest.fixture
    def crnn_up1(self):
        return UpsamplingCRNN(
            num_filters_for_upsampling=256,
            num_keypoints=17,
            upsampling_factor=1,
        )

    def test_initialization(self, crnn_up2):
        """Test that UpsamplingCRNN initializes with correct attributes."""
        assert crnn_up2.upsampling_factor == 2
        assert isinstance(crnn_up2.pixel_shuffle, torch.nn.PixelShuffle)
        assert hasattr(crnn_up2, 'W_pre')
        assert hasattr(crnn_up2, 'W_f')
        assert hasattr(crnn_up2, 'H_f')
        assert hasattr(crnn_up2, 'W_b')
        assert hasattr(crnn_up2, 'H_b')

    def test_upsampling_factor_2_layers(self, crnn_up2):
        """Test that upsampling_factor=2 creates correct layers."""
        assert len(crnn_up2.layers) == 5
        assert crnn_up2.W_pre in crnn_up2.layers
        assert crnn_up2.W_f in crnn_up2.layers
        assert crnn_up2.H_f in crnn_up2.layers
        assert crnn_up2.W_b in crnn_up2.layers
        assert crnn_up2.H_b in crnn_up2.layers

    def test_upsampling_factor_1_layers(self, crnn_up1):
        """Test that upsampling_factor=1 creates correct layers."""
        assert len(crnn_up1.layers) == 4
        assert not hasattr(crnn_up1, 'W_pre') or crnn_up1.W_pre not in crnn_up1.layers

    def test_forward_shape_upsampling_2(self, crnn_up2):
        """Test forward pass output shape with upsampling_factor=2."""
        num_frames = 5
        batch_size = 2
        num_features = 256
        rep_h, rep_w = 8, 8

        features = torch.randn(num_frames, batch_size, num_features, rep_h, rep_w)
        output = crnn_up2(features)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 17
        assert output.shape[2] == rep_h * (2 ** (2 + 1))  # upsampling=2 plus pixelshuffle
        assert output.shape[3] == rep_w * (2 ** (2 + 1))

    def test_forward_shape_upsampling_1(self, crnn_up1):
        """Test forward pass output shape with upsampling_factor=1."""
        num_frames = 5
        batch_size = 2
        num_features = 256
        rep_h, rep_w = 8, 8

        features = torch.randn(num_frames, batch_size, num_features, rep_h, rep_w)
        output = crnn_up1(features)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 17
        assert output.shape[2] == rep_h * (2 ** (1 + 1))  # upsampling=1 plus pixelshuffle
        assert output.shape[3] == rep_w * (2 ** (1 + 1))

    def test_output_normalized(self, crnn_up2):
        """Test that output heatmaps are normalized via spatial softmax."""
        num_frames = 3
        batch_size = 1
        features = torch.randn(num_frames, batch_size, 256, 8, 8)

        output = crnn_up2(features)

        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

        for i in range(output.shape[1]):
            heatmap_sum = output[0, i].sum()
            assert torch.isclose(heatmap_sum, torch.tensor(1.0), atol=1e-5)
