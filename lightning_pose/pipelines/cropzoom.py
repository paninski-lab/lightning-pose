"""Detector -> crop -> pose estimator inference in a single in-memory pass."""

from __future__ import annotations

import contextlib
import gc
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from lightning_pose.api.model import _PRECISION_TO_AUTOCAST_DTYPE, Model
from lightning_pose.callbacks import JSONInferenceProgressTracker
from lightning_pose.data.bboxes import crop_and_resize_frames
from lightning_pose.data.datatypes import UnlabeledBatchDict
from lightning_pose.data.video.factory import build_video_reader
from lightning_pose.pipelines.base import Pipeline, PipelineResult
from lightning_pose.pipelines.stages import KeypointBboxStage
from lightning_pose.utils.device import get_accelerator
from lightning_pose.utils.inference_types import _Precision, _Reader
from lightning_pose.utils.predictions import PredictionHandler
from lightning_pose.utils.predictions import generate_labeled_video as generate_labeled_video_fn

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []

_CONTEXT_MODEL_TYPES = ('heatmap_mhcrnn',)
_BBOX_COLUMNS = ['x', 'y', 'h', 'w']

# frames per batch; small because frames are held at full video resolution
DEFAULT_BATCH_SIZE = 16


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
        confidence_thresh: float = 0.0,
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
            confidence_thresh: anchor keypoints with detector confidence below this are dropped
                from the box computation; frames left with fewer than 2 anchors reuse the
                previous box. The default of 0 keeps every anchor, as ``create_bbox`` does.
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

    def _run_model(
        self,
        model: Model,
        frames: torch.Tensor,
        bbox: torch.Tensor,
        transforms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one model's ``predict_step`` on a batch, at the model's configured precision.

        Args:
            model: loaded model to run.
            frames: normalized input frames, shape (batch, 3, H, W), on the model's device.
            bbox: ``[x, y, h, w]`` region of the original frame that ``frames`` covers, shape
                (batch, 4); used by ``predict_step`` to map keypoints to original-frame
                coordinates.
            transforms: the reader's ``transforms`` field, passed through.

        Returns:
            (keypoints of shape (batch, 2 * num_keypoints), confidence of shape
            (batch, num_keypoints)), keypoints in original-frame pixel coordinates.
        """
        batch_dict = UnlabeledBatchDict(
            frames=frames, transforms=transforms, bbox=bbox, is_multiview=False,
        )
        autocast_dtype = _PRECISION_TO_AUTOCAST_DTYPE.get(model.precision)
        with torch.inference_mode(), (
            torch.autocast(device_type=frames.device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        ):
            kp, conf = model.model.predict_step(batch_dict, 0)[:2]  # type: ignore[union-attr]
        return kp, conf

    def _dali_config(self, batch_size: int | None) -> DictConfig:
        """Reader windowing config, with the prediction batch size overridden if requested."""
        dali_config = OmegaConf.create(OmegaConf.to_container(self.pose_estimator.cfg.dali))
        assert isinstance(dali_config, DictConfig)
        if batch_size is not None:
            dali_config.base.predict.sequence_length = batch_size
            dali_config.context.predict.sequence_length = batch_size
        return dali_config

    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: str | Path,
        reader: _Reader | None = None,
        progress_file: Path | None = None,
        batch_size: int | None = DEFAULT_BATCH_SIZE,
        generate_labeled_video: bool = False,
    ) -> PipelineResult:
        """Run detector and pose estimator on a video, saving predictions and bboxes.

        Writes ``{video_stem}.csv`` (keypoints, original-frame coordinates) and
        ``{video_stem}_bbox.csv`` (the per-frame boxes, unclamped, as ``litpose create_bbox``
        would write them) to ``output_dir``.

        Per batch, full-resolution frames are held on the compute device until the crop is
        taken. Peak memory is a few copies of ``batch_size`` normalized float32 frames at the
        video's native resolution (about ``batch_size * 3 * H * W * 4`` bytes each), so lower
        ``batch_size`` for high-resolution video.

        Args:
            video_file: path to the video file.
            output_dir: directory for the prediction and bbox CSVs (created if missing).
            reader: video-reading backend; None auto-selects, as in
                ``Model.predict_on_video_file``.
            progress_file: path to a file used to report progress to the App.
            batch_size: frames per batch. Defaults to ``DEFAULT_BATCH_SIZE``, deliberately
                smaller than the models' own ``cfg.dali`` prediction batch since frames are
                full resolution; None uses the pose estimator's ``cfg.dali`` value.
            generate_labeled_video: whether to also save a keypoint-annotated video to
                ``{output_dir}/labeled_videos/{video_stem}_labeled.mp4``.

        Returns:
            PipelineResult with the predictions and the bboxes that were saved.
        """
        video_file = Path(video_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.detector._load()
        self.pose_estimator._load()
        device = torch.device('cuda' if get_accelerator() == 'gpu' else 'cpu')
        detector_net = self.detector.model
        pose_net = self.pose_estimator.model
        assert detector_net is not None and pose_net is not None
        detector_net.to(device).eval()
        pose_net.to(device).eval()

        resize_dims_detector = [
            self.detector.cfg.data.image_resize_dims.height,
            self.detector.cfg.data.image_resize_dims.width,
        ]
        resize_dims_pose = [
            self.pose_estimator.cfg.data.image_resize_dims.height,
            self.pose_estimator.cfg.data.image_resize_dims.width,
        ]

        loader = build_video_reader(
            reader=reader,
            probe_video=str(video_file),
            model_type='base',
            dali_config=self._dali_config(batch_size),
            filenames=[str(video_file)],
            resize_dims=None,  # full-resolution passthrough; this pipeline resizes and crops
            bbox_df=None,
        )
        progress = JSONInferenceProgressTracker(progress_file) if progress_file else None

        self.bbox_stage.reset()
        preds: list[tuple[torch.Tensor, torch.Tensor]] = []
        bboxes: list[torch.Tensor] = []
        try:
            for idx_batch, batch_dict in enumerate(loader):
                frames = batch_dict['frames'].to(device)  # (batch, 3, H, W), full resolution
                bbox_full = batch_dict['bbox'].to(device)
                transforms = batch_dict['transforms']
                height, width = frames.shape[-2:]

                # detector sees a resized copy; the full-res frames stay alive for the crop
                frames_detector = F.interpolate(
                    frames, size=resize_dims_detector, mode='bilinear', align_corners=False,
                )
                kp_detector, conf_detector = self._run_model(
                    self.detector, frames_detector, bbox_full, transforms,
                )
                del frames_detector
                bbox = self.bbox_stage(
                    kp_detector.reshape(len(frames), -1, 2), conf_detector, height, width,
                )

                # same crop + resize as the reader's bbox mode, so results match the two-step
                # `litpose create_bbox` -> `litpose predict --bbox_dir` flow
                bbox_rows = pd.DataFrame(bbox.cpu().numpy(), columns=pd.Index(_BBOX_COLUMNS))
                frames_pose, bbox_used = crop_and_resize_frames(
                    frames, bbox_rows, resize_dims_pose,
                )
                del frames
                kp, conf = self._run_model(self.pose_estimator, frames_pose, bbox_used, transforms)
                del frames_pose

                preds.append((kp.cpu(), conf.cpu()))
                bboxes.append(bbox.cpu())
                if progress is not None:
                    progress._save_progress(idx_batch + 1, len(loader))
        finally:
            # leave the GPU as we found it
            detector_net.to('cpu')
            pose_net.to('cpu')
            del loader
            gc.collect()
            torch.cuda.empty_cache()

        pred_handler = PredictionHandler(cfg=self.pose_estimator.cfg, video_file=str(video_file))
        preds_df = pred_handler(preds=preds, is_multiview_video=False)
        assert isinstance(preds_df, pd.DataFrame)
        bbox_df = pd.DataFrame(
            torch.cat(bboxes)[: len(preds_df)].numpy().astype('int64'),
            columns=pd.Index(_BBOX_COLUMNS),
        )

        if self.bbox_stage.num_fallback > 0:
            logger.warning(
                f'{self.bbox_stage.num_fallback} of {len(bbox_df)} frames had no usable '
                f'detection and reused the previous frame\'s bbox'
            )
        preds_df.to_csv(output_dir / f'{video_file.stem}.csv')
        # lineterminator='\n' keeps output byte-identical across platforms, as in generate_bbox
        bbox_df.to_csv(output_dir / f'{video_file.stem}_bbox.csv', lineterminator='\n')
        if generate_labeled_video:
            labeled_dir = output_dir / 'labeled_videos'
            labeled_dir.mkdir(parents=True, exist_ok=True)
            generate_labeled_video_fn(
                video_file=str(video_file),
                preds_df=preds_df,
                output_mp4_file=str(labeled_dir / f'{video_file.stem}_labeled.mp4'),
                confidence_thresh_for_vid=self.pose_estimator.cfg.eval.confidence_thresh_for_vid,
                colormap=self.pose_estimator.cfg.eval.get('colormap', 'cool'),
                bbox_df=bbox_df,
            )
        return PipelineResult(predictions=preds_df, bboxes=bbox_df)
