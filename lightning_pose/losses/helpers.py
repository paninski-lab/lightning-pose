"""Helper functions for losses."""

from typing import Literal

import numpy as np
import torch
from torchtyping import TensorType

# to ignore imports for sphix-autoapidoc
__all__ = []


class EmpiricalEpsilon:
    """Find percentile value of a given loss tensor."""

    def __init__(self, percentile: float) -> None:
        self.percentile = percentile

    def __call__(self, loss: TensorType | np.ndarray) -> float:
        """Compute the percentile of some loss, to use an for epsilon-insensitive loss.

        Args:
            loss: tensor with scalar loss per term (e.g., loss per image, or loss per
                keypoint, etc.)

        Returns:
            the percentile of the loss which we use as epsilon

        """
        flattened_loss = loss.flatten()  # applies for both np arrays and torch tensors.
        if type(loss) is torch.Tensor:
            flattened_loss = flattened_loss.clone().detach().cpu().numpy()
        return np.nanpercentile(flattened_loss, self.percentile, axis=0)


# @typechecked
def convert_dict_values_to_tensors(
    param_dict: dict[str, np.ndarray | float],
    device: Literal["cpu", "cuda"] | torch.device,
) -> dict[str, torch.Tensor]:
    # TODO: currently supporting just floats
    for key, val in param_dict.items():
        param_dict[key] = torch.tensor(val, dtype=torch.float, device=device)
    return param_dict
