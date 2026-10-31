"""Detector -> crop -> pose estimator inference in a single in-memory pass."""

from __future__ import annotations

import logging
from pathlib import Path

from lightning_pose.api.model import Model
from lightning_pose.pipelines.base import Pipeline, PipelineResult
from lightning_pose.pipelines.stages import KeypointBboxStage
from lightning_pose.utils.inference_types import _Precision, _Reader

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []

_CONTEXT_MODEL_TYPES = ('heatmap_mhcrnn',)


class CropZoomPipeline(Pipeline):
    """Runs a detector and a pose estimator back to back on full-resolution video frames.

    Per batch of frames: the detector sees a resized copy; its keypoints become per-frame
    bounding boxes in original-frame coordinates (:class:`KeypointBboxStage`); the full-res
    frames are cropped to those boxes and resized for the pose estimator; the pose
    estimator's keypoints are mapped back to original-frame coordinates. The full-resolution
    batch is kept alive until the crop is taken. The two models are independent: neither
    reads the other's resize dims or keypoint set.

    Outputs are a standard LP prediction CSV (original-frame coordinates) and a standard LP
    bbox CSV. Only single-view, single-frame (non-context) models are supported for now.

    Attributes:
        detector: model whose keypoints define the boxes.
        pose_estimator: model run on the cropped frames.
        bbox_stage: converts detector keypoints into boxes.
    """

    def __init__(
        self,
        detector: Model,
        pose_estimator: Model,
        bbox_stage: KeypointBboxStage,
    ) -> None:
        """Initialize from already-built models.

        Args:
            detector: detector model.
            pose_estimator: pose estimator model.
            bbox_stage: stage that turns detector keypoints into boxes.

        Raises:
            NotImplementedError: if either model is multi-view or uses temporal context.
        """
        for role, model in (('detector', detector), ('pose estimator', pose_estimator)):
            self._check_supported(role, model)
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.bbox_stage = bbox_stage

    @classmethod
    def from_dirs(
        cls,
        detector_dir: str | Path,
        pose_estimator_dir: str | Path,
        anchor_keypoints: list[str] | None = None,
        crop_ratio: float | None = None,
        crop_height: int | None = None,
        crop_width: int | None = None,
        confidence_thresh: float | None = 0.5,
        precision: _Precision = 'fp32',
    ) -> CropZoomPipeline:
        """Build a pipeline from two model directories.

        Args:
            detector_dir: model directory of the detector.
            pose_estimator_dir: model directory of the pose estimator.
            anchor_keypoints: detector keypoints that define the box; None uses all.
            crop_ratio: scale factor applied to the anchor-keypoint span.
            crop_height: fixed box height; given together with ``crop_width``.
            crop_width: fixed box width; given together with ``crop_height``.
            confidence_thresh: detector confidence below which a frame reuses the previous box;
                None disables the check. Defaults to 0.5.
            precision: inference precision applied to both models.

        Returns:
            CropZoomPipeline with both models loaded lazily.
        """
        detector = Model.from_dir(detector_dir, precision=precision)
        pose_estimator = Model.from_dir(pose_estimator_dir, precision=precision)
        bbox_stage = KeypointBboxStage(
            keypoint_names=list(detector.cfg.data.keypoint_names),
            anchor_keypoints=anchor_keypoints,
            crop_ratio=crop_ratio,
            crop_height=crop_height,
            crop_width=crop_width,
            confidence_thresh=confidence_thresh,
        )
        return cls(detector, pose_estimator, bbox_stage)

    @staticmethod
    def _check_supported(role: str, model: Model) -> None:
        """Raise if ``model`` is a kind this pipeline cannot run yet."""
        if model.config.is_multi_view():
            raise NotImplementedError(
                f'CropZoomPipeline does not support multi-view models yet (got a multi-view '
                f'{role} at {model.model_dir}).'
            )
        if model.cfg.model.model_type in _CONTEXT_MODEL_TYPES:
            raise NotImplementedError(
                f'CropZoomPipeline does not support context models yet (got model_type '
                f'{model.cfg.model.model_type!r} for the {role} at {model.model_dir}).'
            )

    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: str | Path,
        reader: _Reader | None = None,
        progress_file: Path | None = None,
    ) -> PipelineResult:
        """Run detector and pose estimator on a video (not implemented yet).

        Args:
            video_file: path to the video file.
            output_dir: directory for the prediction and bbox CSVs.
            reader: video-reading backend.
            progress_file: path to a file used to report progress to the App.

        Raises:
            NotImplementedError: always; the inference loop is the next checkpoint.
        """
        raise NotImplementedError('CropZoomPipeline inference loop is not implemented yet.')
