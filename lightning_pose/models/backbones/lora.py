"""Minimal LoRA (Hu et al. 2021) for the linear layers of a transformer backbone.

Each targeted ``nn.Linear`` is replaced in place by :class:`LoRALinear`, a subclass that
keeps the original ``weight``/``bias`` (frozen, same state-dict keys — so checkpoints of the
unadapted model load unchanged) and adds a trainable low-rank update ``B @ A`` scaled by
``alpha / rank``. ``B`` is zero-initialised, so the wrapped model is exactly the original
until training moves ``B``. No external dependency.
"""

import logging
import math

import torch
from torch import nn

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


class LoRALinear(nn.Linear):
    """``nn.Linear`` with a frozen base weight and a trainable rank-``r`` update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None:
                self.bias.copy_(base.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, lora_rank={self.rank}, scaling={self.scaling:g}'


def apply_lora(
    module: nn.Module,
    targets: list[str],
    rank: int,
    alpha: float,
    freeze_rest: bool = True,
) -> int:
    """Replace every ``nn.Linear`` child whose attribute name is in ``targets`` with LoRA.

    Args:
        module: root module to walk (e.g. the backbone)
        targets: attribute names of linear layers to adapt, e.g.
            ``['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']``
        rank: LoRA rank
        alpha: LoRA scaling numerator (update is scaled by ``alpha / rank``)
        freeze_rest: set ``requires_grad=False`` on every non-LoRA parameter of ``module``

    Returns:
        number of layers wrapped
    """
    if rank <= 0:
        raise ValueError(f'lora rank must be positive, got {rank}')
    if freeze_rest:
        for p in module.parameters():
            p.requires_grad_(False)
    n = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha))
                n += 1
    if n == 0:
        raise ValueError(f'no nn.Linear named {targets} found under {type(module).__name__}')
    n_lora = sum(p.numel() for p in lora_parameters(module))
    logger.info(f'LoRA: wrapped {n} linear layers (rank {rank}, alpha {alpha}); '
                f'{n_lora / 1e6:.2f} M trainable adapter params')
    return n


def lora_parameters(module: nn.Module):
    """Iterate over the LoRA parameters (``lora_A``/``lora_B``) under ``module``."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B
