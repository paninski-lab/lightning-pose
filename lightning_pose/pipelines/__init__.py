"""Multi-model inference pipelines.

A pipeline composes several trained models (see :class:`lightning_pose.api.Model`) into a
single inference run: it owns the data flow between the models, and nothing else. Pipelines
never train models and never modify the models they wrap.

Exports:

- :class:`Pipeline` — abstract base class; defines ``predict_on_video_file``.
- :class:`PipelineResult` — what a pipeline returns: keypoint predictions plus the
  intermediate per-frame bounding boxes.
- :class:`CropZoomPipeline` — detector -> per-frame crop -> pose estimator, in memory.
- :class:`KeypointBboxStage` — the detector-keypoints -> bounding-box stage used by
  ``CropZoomPipeline``; reusable by future multi-view / multi-animal pipelines.
"""

from lightning_pose.pipelines.base import Pipeline, PipelineResult
from lightning_pose.pipelines.cropzoom import CropZoomPipeline
from lightning_pose.pipelines.stages import KeypointBboxStage

__all__ = [
    'CropZoomPipeline',
    'KeypointBboxStage',
    'Pipeline',
    'PipelineResult',
]
