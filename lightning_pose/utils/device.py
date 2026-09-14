"""GPU-presence detection for selecting a ``pytorch_lightning.Trainer`` accelerator."""

import logging

import torch

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__ = ["get_accelerator"]


def get_accelerator() -> str:
    """Return the ``pl.Trainer`` accelerator string for the current machine.

    Uses a GPU whenever one is available, falling back to CPU otherwise. Centralizing this
    check keeps the training and inference entry points from each guessing independently, and
    means a machine with no CUDA device (Windows without WSL, macOS, CPU-only Linux) runs
    without raising instead of failing to find a GPU.

    Returns:
        ``"gpu"`` if a CUDA device is available, else ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "gpu"
    logger.info("no CUDA device found, using CPU accelerator")
    return "cpu"
