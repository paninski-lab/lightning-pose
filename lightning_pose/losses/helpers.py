"""Helper functions for losses."""

import numpy as np
import torch

# to ignore imports for sphix-autoapidoc
__all__ = []


class EmpiricalEpsilon:
    """Find percentile value of a given loss tensor."""

    def __init__(self, percentile: float) -> None:
        """Initialize EmpiricalEpsilon.

        Args:
            percentile: the percentile (0–100) of the loss distribution to use as epsilon.
        """
        self.percentile = percentile

    def __call__(self, loss: torch.Tensor | np.ndarray) -> float:
        """Compute the percentile of some loss, to use an for epsilon-insensitive loss.

        Args:
            loss: tensor with scalar loss per term (e.g., loss per image, or loss per
                keypoint, etc.)

        Returns:
            the percentile of the loss which we use as epsilon

        """
        if isinstance(loss, torch.Tensor):
            flattened_loss = loss.flatten().clone().detach().cpu().numpy()
        else:
            flattened_loss = loss.flatten()
        return float(np.nanpercentile(flattened_loss, self.percentile, axis=0))


# @typechecked
def convert_dict_values_to_tensors(
    param_dict: dict[str, np.ndarray | float],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Convert all values in a parameter dictionary to float tensors on the given device.

    Args:
        param_dict: mapping from parameter name to scalar or array value.
        device: target device for the output tensors.

    Returns:
        Dictionary with the same keys and values converted to ``torch.float`` tensors.
    """
    # TODO: currently supporting just floats
    return {
        key: torch.tensor(val, dtype=torch.float, device=device)
        for key, val in param_dict.items()
    }
