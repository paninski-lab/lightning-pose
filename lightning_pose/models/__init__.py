"""Pose estimation model classes, re-exported at the package level.

**Four model types**, each available in a supervised and a semi-supervised variant
(8 concrete classes total):

- ``regression`` — direct (x, y) coordinate regression.
  ``RegressionTracker`` / ``SemiSupervisedRegressionTracker`` → ``regression_tracker.py``
- ``heatmap`` — per-keypoint 2-D Gaussian heatmaps.
  ``HeatmapTracker`` / ``SemiSupervisedHeatmapTracker`` → ``heatmap_tracker.py``
- ``heatmap_mhcrnn`` — heatmaps with temporal context via a recurrent head (MHCRNN).
  ``HeatmapTrackerMHCRNN`` / ``SemiSupervisedHeatmapTrackerMHCRNN``
  → ``heatmap_tracker_mhcrnn.py``
- ``heatmap_multiview_transformer`` — multi-camera heatmaps with cross-view attention.
  ``HeatmapTrackerMultiviewTransformer`` /
  ``SemiSupervisedHeatmapTrackerMultiviewTransformer``
  → ``heatmap_tracker_multiview.py``

**Supervised / semi-supervised split**: every supervised class has a semi-supervised
counterpart produced by mixing in
:class:`~lightning_pose.models.base.SemiSupervisedTrackerMixin`.  The mixin adds a
second ``loss_factory_unsupervised`` argument and extends ``training_step`` to compute
unsupervised losses on unlabeled video frames.

**Other files in this package**:

- ``base.py`` — abstract bases and shared logic:
  :class:`~lightning_pose.models.base.BaseFeatureExtractor`,
  :class:`~lightning_pose.models.base.BaseSupervisedTracker`,
  :class:`~lightning_pose.models.base.SemiSupervisedTrackerMixin`.
- ``factory.py`` — :func:`get_model` (full construction from config) and
  :func:`get_model_class` (pure ``(model_type, semi_supervised) → class`` dispatch);
  :data:`ALLOWED_MODEL_TYPES` Literal defined here.
- ``backbones/`` — backbone wrappers and :func:`~lightning_pose.models.backbones.build_backbone`;
  see ``backbones/__init__.py`` for the type hierarchy and how to add a new backbone.
- ``heads/`` — output head classes
  (:class:`~lightning_pose.models.heads.HeatmapHead`,
  :class:`~lightning_pose.models.heads.HeatmapMHCRNNHead`,
  :class:`~lightning_pose.models.heads.LinearRegressionHead`).
"""

from lightning_pose.models.base import check_if_semi_supervised
from lightning_pose.models.factory import ALLOWED_MODEL_TYPES, get_model, get_model_class
from lightning_pose.models.heatmap_tracker import (
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
)
from lightning_pose.models.heatmap_tracker_mhcrnn import (
    HeatmapTrackerMHCRNN,
    SemiSupervisedHeatmapTrackerMHCRNN,
)
from lightning_pose.models.heatmap_tracker_multiview import (
    HeatmapTrackerMultiviewTransformer,
    SemiSupervisedHeatmapTrackerMultiviewTransformer,
)
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)

ALLOWED_MODELS = (
    HeatmapTracker
    | SemiSupervisedHeatmapTracker
    | HeatmapTrackerMHCRNN
    | SemiSupervisedHeatmapTrackerMHCRNN
    | HeatmapTrackerMultiviewTransformer
    | SemiSupervisedHeatmapTrackerMultiviewTransformer
    | RegressionTracker
    | SemiSupervisedRegressionTracker
)

# to ignore imports for sphinx-autoapidoc
__all__ = [
    'check_if_semi_supervised',
    'get_model',
    'get_model_class',
    'HeatmapTracker',
    'SemiSupervisedHeatmapTracker',
    'HeatmapTrackerMHCRNN',
    'SemiSupervisedHeatmapTrackerMHCRNN',
    'HeatmapTrackerMultiviewTransformer',
    'SemiSupervisedHeatmapTrackerMultiviewTransformer',
    'RegressionTracker',
    'SemiSupervisedRegressionTracker',
]
