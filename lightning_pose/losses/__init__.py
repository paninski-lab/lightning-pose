"""Supervised and semi-supervised loss functions for pose estimation.

Loss names follow a ``{domain}_{variant}`` convention.  The full registry is returned
by :func:`get_loss_classes`; the names below are the strings used in config files under
``cfg.model.losses_to_use`` and ``cfg.model.heatmap_loss_type``.

*Supervised losses* (always active; selected automatically by model type):

- ``heatmap_mse``, ``heatmap_kl``, ``heatmap_js`` — pixel-wise heatmap losses
  (:class:`~lightning_pose.losses.losses.HeatmapMSELoss`,
  :class:`~lightning_pose.losses.losses.HeatmapKLLoss`,
  :class:`~lightning_pose.losses.losses.HeatmapJSLoss`).
- ``regression`` — MSE on (x, y) coordinates
  (:class:`~lightning_pose.losses.losses.RegressionMSELoss`).
- ``rmse`` — RMSE on (x, y) coordinates
  (:class:`~lightning_pose.losses.losses.RegressionRMSELoss`).

*Supervised losses* (active when log_weight is not null; needs updating to mirror semi-supervised)
- ``supervised_pairwise_projections`` — multiview epipolar consistency
  (:class:`~lightning_pose.losses.losses.PairwiseProjectionsLoss`).
- ``supervised_reprojection_heatmap_mse`` — multiview reprojection heatmap loss
  (:class:`~lightning_pose.losses.losses.ReprojectionHeatmapLoss`).

*Semi-supervised losses* (opt-in via ``cfg.model.losses_to_use``):

- ``pca_multiview``, ``pca_singleview`` — both served by
  :class:`~lightning_pose.losses.losses.PCALoss`; enforce pose-space structure via PCA.
- ``temporal`` — penalises abrupt frame-to-frame keypoint movement
  (:class:`~lightning_pose.losses.losses.TemporalLoss`).
- ``temporal_heatmap_mse``, ``temporal_heatmap_kl`` — both served by
  :class:`~lightning_pose.losses.losses.TemporalHeatmapLoss`; heatmap-level temporal
  smoothness.
- ``unimodal_mse``, ``unimodal_kl``, ``unimodal_js`` — all served by
  :class:`~lightning_pose.losses.losses.UnimodalLoss`; encourage single-peaked heatmaps.
"""

from lightning_pose.losses.factory import get_loss_classes, get_loss_factories

# to ignore imports for sphinx-autoapidoc
__all__ = [
    "get_loss_classes",
    "get_loss_factories",
]
