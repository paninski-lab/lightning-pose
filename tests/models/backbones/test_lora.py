"""Tests for models/backbones/lora.py."""

import torch
from torch import nn

from lightning_pose.models.backbones.lora import (
    LORA_TARGETS_DEFAULT,
    LoRALinear,
    apply_lora_from_config,
    lora_parameters,
)


class _Block(nn.Module):
    """Two of the default targets and one non-target linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.other = nn.Linear(4, 4)


class TestApplyLoraFromConfig:
    """Test the function apply_lora_from_config."""

    def test_apply_lora_from_config_defaults(self):
        block = _Block()

        n = apply_lora_from_config(block, {})

        assert n == 2
        assert isinstance(block.q_proj, LoRALinear) and isinstance(block.v_proj, LoRALinear)
        assert not isinstance(block.other, LoRALinear)
        assert block.q_proj.rank == 16
        assert block.q_proj.scaling == 2.0      # alpha defaults to 2 * rank
        assert 'q_proj' in LORA_TARGETS_DEFAULT

    def test_apply_lora_from_config_explicit(self):
        block = _Block()

        n = apply_lora_from_config(block, {'targets': ['other'], 'rank': 4, 'alpha': 4})

        assert n == 1
        assert block.other.rank == 4 and block.other.scaling == 1.0

    def test_apply_lora_from_config_only_adapters_train(self):
        block = _Block()

        apply_lora_from_config(block, {'rank': 2})

        trainable = {id(p) for p in block.parameters() if p.requires_grad}
        assert trainable == {id(p) for p in lora_parameters(block)}
        x = torch.randn(3, 4)
        assert torch.equal(block.q_proj(x), nn.functional.linear(x, block.q_proj.weight,
                                                                 block.q_proj.bias))
