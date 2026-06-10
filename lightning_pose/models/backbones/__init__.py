"""Public façade for the backbones package.

All backbone type constants and the ``build_backbone`` factory live in
``factory.py``; this module re-exports them so callers can write::

    from lightning_pose.models.backbones import build_backbone, ALLOWED_BACKBONES

Nothing should be defined here — add new symbols to ``factory.py`` and extend
the import list below.
"""

from lightning_pose.models.backbones.factory import (
    ALLOWED_BACKBONES,
    ALLOWED_CONVNET_BACKBONES,
    ALLOWED_TRANSFORMER_BACKBONES,
    ALLOWED_TRANSFORMER_BACKBONES_MULTIVIEW,
    BACKBONE_STRIDES,
    build_backbone,
)

__all__ = [
    'ALLOWED_BACKBONES',
    'ALLOWED_CONVNET_BACKBONES',
    'ALLOWED_TRANSFORMER_BACKBONES',
    'ALLOWED_TRANSFORMER_BACKBONES_MULTIVIEW',
    'BACKBONE_STRIDES',
    'build_backbone',
]
