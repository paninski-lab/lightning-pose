"""Prediction head classes, re-exported at the package level.

Unlike ``models/backbones/``, this package has no ``factory.py`` and no dispatch function:
each tracker hardcodes its own head class directly in ``__init__`` --
``RegressionTracker`` -> ``LinearRegressionHead`` (``regression_tracker.py``),
``HeatmapTrackerMultiviewTransformer`` -> ``HeatmapHead`` (``heatmap_tracker_multiview.py``),
``HeatmapTrackerMHCRNN`` -> ``HeatmapMHCRNNHead`` (``heatmap_tracker_mhcrnn.py``). The
exceptions are ``HeatmapTracker``, which takes a ``head: Literal["heatmap_cnn",
"heatmap_cnn_nonlinear"]`` constructor arg (``HeatmapHead`` or ``HeatmapNonlinearHead``), and
``HeatmapTrackerMultiviewTransformer``, which takes ``head: Literal["heatmap_cnn"]``. Both
dispatch on it with an ``if/else`` -- a live extension point for additional head types, with
no recipe of its own yet.

**Adding a new head**: define the class here, then wire it into the one (or more) tracker
``__init__`` methods that should use it -- there is no shared registry to update.
"""

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_mhcrnn import HeatmapMHCRNNHead
from lightning_pose.models.heads.heatmap_nonlinear import HeatmapNonlinearHead
from lightning_pose.models.heads.regression import LinearRegressionHead

__all__: list[str] = []
