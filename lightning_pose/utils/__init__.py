"""Utility helpers used across training, inference, and evaluation.

**Submodules**:

- ``utils.io`` — path handling and file I/O utilities.
- ``utils.predictions`` — inference on labeled datasets and unlabeled videos; video annotation.
- ``utils.pca`` — :class:`~lightning_pose.utils.pca.KeypointPCA` for PCA-based
  unsupervised losses.
- ``utils.cropzoom`` — crop/zoom pipeline: labeled-frame and video cropping to bounding-box ROIs.

This module itself exports :func:`pretty_print_str` and :func:`pretty_print_cfg`.
"""

import logging

from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "pretty_print_str",
    "pretty_print_cfg",
]


def pretty_print_str(string: str, symbol: str = "-") -> None:
    """Print a string surrounded by a horizontal rule made of the given symbol.

    Args:
        string: the text to print.
        symbol: single character used to draw the horizontal rule.
    """
    rule = symbol * len(string)
    logger.info(f'{rule}\n{string}\n{rule}')


def pretty_print_cfg(cfg: DictConfig | ListConfig) -> None:
    """Print a human-readable summary of the config, skipping the ``eval`` section.

    Args:
        cfg: hydra config to display.
    """
    lines = []
    for key, val in cfg.items():
        if key == "eval":
            continue
        lines.append("--------------------")
        lines.append(f"{key} parameters")
        lines.append("--------------------")
        if hasattr(val, "items"):
            for k, v in val.items():
                lines.append(f"{k}: {v}")
        if isinstance(val, str):
            lines.append(val)
        lines.append("")
    logger.info("\n".join(lines))
