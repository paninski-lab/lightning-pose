"""Prediction head classes, re-exported at the package level.

Unlike ``models/backbones/``, this package has no ``factory.py`` and no dispatch function:
each tracker hardcodes its own head class directly in ``__init__`` --
``RegressionTracker`` -> ``LinearRegressionHead`` (``regression_tracker.py``),
``HeatmapTracker``/``HeatmapTrackerMultiviewTransformer`` -> ``HeatmapHead``
(``heatmap_tracker.py``, ``heatmap_tracker_multiview.py``), ``HeatmapTrackerMHCRNN`` ->
``HeatmapMHCRNNHead`` (``heatmap_tracker_mhcrnn.py``). The one exception is
``HeatmapTrackerMultiviewTransformer``, which takes a ``head: Literal["heatmap_cnn"]``
constructor arg and dispatches on it with an ``if/else`` -- a live extension point for
additional multiview head types, with no recipe of its own yet.

**Adding a new head**: define the class here, then wire it into the one (or more) tracker
``__init__`` methods that should use it -- there is no shared registry to update.
"""

from lightning_pose.models.heads.heatmap import HeatmapHead
from lightning_pose.models.heads.heatmap_mhcrnn import HeatmapMHCRNNHead
from lightning_pose.models.heads.regression import LinearRegressionHead

__all__: list[str] = []
