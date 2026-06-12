"""Re-export façade for the backbones package.

All backbone type constants and the ``build_backbone`` factory are defined in
``factory.py``; this module re-exports them. Nothing should be defined here —
add new symbols to ``factory.py`` and extend the import list below.
"""

from lightning_pose.models.backbones.factory import (
    ALLOWED_BACKBONES,
    ALLOWED_CONVNET_BACKBONES,
    ALLOWED_TRANSFORMER_BACKBONES,
    ALLOWED_TRANSFORMER_BACKBONES_MULTIVIEW,
    BACKBONE_STRIDES,
    build_backbone,
)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []
