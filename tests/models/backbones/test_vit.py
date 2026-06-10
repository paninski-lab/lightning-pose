"""Test ViT backbone utilities."""

from unittest.mock import MagicMock, patch

import safetensors.torch
import torch

from lightning_pose.models.backbones.vit import load_vit_backbone_checkpoint


class TestLoadVitBackboneCheckpoint:
    """Test the load_vit_backbone_checkpoint function."""

    def _make_base(self, state_dict: dict) -> MagicMock:
        """Return a mock base whose vision_encoder matches the given state dict."""
        mock_ve = MagicMock()
        mock_ve.state_dict.return_value = state_dict
        base = MagicMock()
        base.vision_encoder = mock_ve
        return base

    def test_loads_pt_checkpoint(self, tmp_path):
        """Loads and applies vit_mae-prefixed weights from a .pt file."""
        weight = torch.randn(4, 4)
        base = self._make_base({'encoder.layer.weight': weight})
        ckpt_path = str(tmp_path / 'ckpt.pt')
        torch.save({'vit_mae.vit.encoder.layer.weight': weight}, ckpt_path)

        load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert 'encoder.layer.weight' in loaded
        assert torch.allclose(loaded['encoder.layer.weight'], weight)

    def test_loads_safetensors_checkpoint(self, tmp_path):
        """Loads and applies vit_mae-prefixed weights from a .safetensors file."""
        weight = torch.randn(4, 4)
        base = self._make_base({'encoder.layer.weight': weight})
        ckpt_path = str(tmp_path / 'ckpt.safetensors')
        safetensors.torch.save_file({'vit_mae.vit.encoder.layer.weight': weight}, ckpt_path)

        load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert 'encoder.layer.weight' in loaded
        assert torch.allclose(loaded['encoder.layer.weight'], weight)

    def test_unwraps_state_dict_key(self, tmp_path):
        """Extracts weights from a checkpoint wrapped under a 'state_dict' key."""
        weight = torch.randn(4, 4)
        base = self._make_base({'encoder.layer.weight': weight})
        ckpt_path = str(tmp_path / 'ckpt.pt')
        torch.save({'state_dict': {'vit_mae.vit.encoder.layer.weight': weight}}, ckpt_path)

        load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert 'encoder.layer.weight' in loaded

    def test_skips_non_vit_mae_keys(self, tmp_path):
        """Keys not prefixed with 'vit_mae.' are excluded from the loaded weights."""
        weight = torch.randn(4, 4)
        base = self._make_base({'encoder.layer.weight': weight, 'other.weight': weight})
        ckpt_path = str(tmp_path / 'ckpt.pt')
        torch.save(
            {
                'vit_mae.vit.encoder.layer.weight': weight,
                'backbone.other.weight': weight,
            },
            ckpt_path,
        )

        load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert 'encoder.layer.weight' in loaded
        assert 'other.weight' not in loaded

    def test_skips_shape_mismatched_keys(self, tmp_path):
        """Keys whose checkpoint shapes differ from the model are excluded."""
        base = self._make_base({'encoder.layer.weight': torch.randn(4, 4)})
        ckpt_path = str(tmp_path / 'ckpt.pt')
        torch.save({'vit_mae.vit.encoder.layer.weight': torch.randn(8, 8)}, ckpt_path)

        load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert loaded == {}

    def test_skips_keys_absent_from_model(self, tmp_path):
        """Keys present in the checkpoint but absent from the model are excluded."""
        base = self._make_base({})
        ckpt_path = str(tmp_path / 'ckpt.pt')
        torch.save({'vit_mae.vit.encoder.layer.weight': torch.randn(4, 4)}, ckpt_path)

        load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert loaded == {}

    def test_fallback_on_torch_load_failure(self, tmp_path):
        """Falls back to weights_only=False when the initial torch.load call raises."""
        weight = torch.randn(4, 4)
        base = self._make_base({'encoder.layer.weight': weight})
        ckpt = {'vit_mae.vit.encoder.layer.weight': weight}
        ckpt_path = str(tmp_path / 'ckpt.pt')

        with patch(
            'lightning_pose.models.backbones.vit.torch.load',
            side_effect=[Exception('cannot load'), ckpt],
        ):
            load_vit_backbone_checkpoint(base, ckpt_path)

        loaded = base.vision_encoder.load_state_dict.call_args[0][0]
        assert 'encoder.layer.weight' in loaded
