"""GPU-presence checks: selecting a ``pl.Trainer`` accelerator, and guarding DALI-only
semi-supervised training with a clear error instead of a confusing low-level one.
"""

import logging

import torch
from omegaconf import ListConfig

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__ = ["get_accelerator", "require_cuda_for_semi_supervised"]


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


def require_cuda_for_semi_supervised(losses_to_use: ListConfig | list | None) -> None:
    """Raise a clear error if semi-supervised training is requested without CUDA + DALI.

    Semi-supervised losses stream unlabeled video frames through a DALI GPU pipeline, which
    needs an NVIDIA GPU and, since ``nvidia-dali-cuda110`` only ships wheels for linux
    x86_64, often isn't even installed on Windows/macOS. Without this guard, that same
    request fails deep inside data-module or DALI-pipeline construction with a confusing
    low-level error -- an ``ImportError`` on a machine with no DALI wheel, or a raw DALI
    pipeline assertion on a GPU-less linux x86_64 machine that happens to have the wheel
    installed. Call this immediately after computing ``cfg.model.losses_to_use`` in every
    factory entry point that might build a semi-supervised model or data module, so the
    failure surfaces before any other setup work runs.

    Args:
        losses_to_use: the ``cfg.model.losses_to_use`` entry (same argument accepted by
            ``check_if_semi_supervised``).

    Raises:
        RuntimeError: if ``losses_to_use`` requests semi-supervised training and either no
            CUDA device is available or the ``nvidia.dali`` package isn't importable.
    """
    # lazy: avoids a circular import (lightning_pose.models.base imports lightning_pose.data,
    # and this module is imported from lightning_pose.data.factory)
    from lightning_pose.models.base import check_if_semi_supervised

    if not check_if_semi_supervised(losses_to_use):
        return

    losses = list(losses_to_use) if losses_to_use is not None else []

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"semi-supervised training requires an NVIDIA GPU with CUDA -- "
            f"model.losses_to_use={losses} need DALI to stream unlabeled video frames, and "
            "no CUDA device was detected on this machine. Either train a supervised-only "
            "model by removing these from model.losses_to_use, or run this on a Linux "
            "machine with an NVIDIA GPU."
        )

    try:
        import lightning_pose.data.video.dali  # noqa: F401  probe importability
    except ImportError as e:
        raise RuntimeError(
            f"semi-supervised training requires the nvidia-dali package -- "
            f"model.losses_to_use={losses} need DALI to stream unlabeled video frames, and "
            "DALI isn't installed on this machine (nvidia-dali-cuda110 only ships prebuilt "
            "wheels for linux x86_64). Either train a supervised-only model by removing "
            "these from model.losses_to_use, or install it with "
            "`pip install nvidia-dali-cuda110` on a linux x86_64 machine."
        ) from e
