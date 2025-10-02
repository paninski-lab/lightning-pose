import pytest
import torch

from lightning_pose.models.heads.regression import LinearRegressionHead


class TestLinearRegressionHead:

    @pytest.fixture
    def basic_head(self):
        """Create a basic LinearRegressionHead instance for testing."""
        return LinearRegressionHead(
            in_channels=512,
            num_targets=34,
        )

    def test_initialization(self, basic_head):
        """Test that LinearRegressionHead initializes with correct attributes."""
        assert hasattr(basic_head, 'linear_layer')
        assert isinstance(basic_head.linear_layer, torch.nn.Linear)
        assert basic_head.linear_layer.in_features == 512
        assert basic_head.linear_layer.out_features == 34

    def test_forward_shape(self, basic_head):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        in_channels = 512
        height, width = 1, 1

        features = torch.randn(batch_size, in_channels, height, width)
        output = basic_head(features)

        assert output.shape == (batch_size, basic_head.linear_layer.out_features)
