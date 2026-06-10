"""Test backbone factory."""

import pytest

from lightning_pose.models.backbones.factory import (
    _build_convnet_backbone,
    _build_transformer_backbone,
    build_backbone,
)


class TestBuildBackbone:
    """Test the build_backbone function."""

    def test_raises_for_unknown_backbone(self):
        """Raises ValueError for an unrecognized backbone name."""
        with pytest.raises(ValueError, match='is not a valid backbone'):
            build_backbone('unknown_arch')


class TestBuildTransformerBackbone:
    """Test the _build_transformer_backbone function."""

    def test_raises_for_unknown_backbone(self):
        """Raises NotImplementedError for an unrecognized ViT backbone name."""
        with pytest.raises(NotImplementedError, match='is not a valid transformer backbone'):
            _build_transformer_backbone('vit_unknown')


class TestBuildConvnetBackbone:
    """Test the _build_convnet_backbone function."""

    def test_raises_for_unknown_backbone(self):
        """Raises NotImplementedError for an unrecognized convnet backbone name."""
        with pytest.raises(NotImplementedError, match='is not a valid convnet backbone'):
            _build_convnet_backbone('unknown_arch')
