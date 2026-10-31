"""Reusable pipeline stages.

A stage is one step of the data flow between models in a pipeline. Stages hold only the
state they need across batches and know nothing about how the models themselves are run.
"""

import logging

import numpy as np
import torch
from jaxtyping import Float

from lightning_pose.data.bboxes import full_frame_bbox
from lightning_pose.utils.cropzoom import compute_bbox_array

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


class KeypointBboxStage:
    """Turns detector keypoints into per-frame bounding boxes, in original-frame coordinates.

    The box is centred on the mean of the anchor keypoints and sized exactly as in the offline
    ``litpose create_bbox`` flow (``crop_ratio`` or fixed ``crop_height``/``crop_width``).

    Frames with no usable detection (any anchor keypoint non-finite or, if
    ``confidence_thresh`` is set, below the threshold) reuse the previous frame's box; if no
    box has been produced yet in the current video, they get the full-frame box. There is no
    temporal smoothing. Call :meth:`reset` between videos.

    Attributes:
        num_fallback: number of frames since the last :meth:`reset` that used a fallback box.
    """

    def __init__(
        self,
        keypoint_names: list[str],
        anchor_keypoints: list[str] | None = None,
        crop_ratio: float | None = None,
        crop_height: int | None = None,
        crop_width: int | None = None,
        confidence_thresh: float | None = None,
    ) -> None:
        """Initialize the stage.

        Args:
            keypoint_names: detector keypoint names, in the order the detector outputs them.
            anchor_keypoints: names of the keypoints that define the box; None or empty uses
                all keypoints.
            crop_ratio: scale factor applied to the anchor-keypoint span to get the box size.
            crop_height: fixed box height in pixels; must be given together with ``crop_width``.
            crop_width: fixed box width in pixels; must be given together with ``crop_height``.
            confidence_thresh: if set, a frame is treated as missing when any anchor keypoint
                has confidence below this value. If None, only non-finite coordinates count.

        Raises:
            ValueError: if an anchor keypoint is not in ``keypoint_names``, or the size
                arguments are not exactly one of ``crop_ratio`` / ``(crop_height, crop_width)``.
        """
        fixed_size = crop_height is not None and crop_width is not None
        if (crop_height is None) != (crop_width is None):
            raise ValueError('crop_height and crop_width must be provided together.')
        if fixed_size and crop_ratio is not None:
            raise ValueError('provide either crop_ratio or (crop_height, crop_width), not both.')
        if not fixed_size and crop_ratio is None:
            raise ValueError('one of crop_ratio or (crop_height, crop_width) must be provided.')

        anchor_keypoints = list(anchor_keypoints or [])
        invalid = set(anchor_keypoints) - set(keypoint_names)
        if invalid:
            raise ValueError(
                f'anchor keypoints not found in detector keypoints: {sorted(invalid)}'
            )

        self.keypoint_names = list(keypoint_names)
        self.idx_anchor = (
            [self.keypoint_names.index(name) for name in anchor_keypoints]
            if anchor_keypoints
            else list(range(len(self.keypoint_names)))
        )
        self.crop_ratio = crop_ratio
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.confidence_thresh = confidence_thresh
        self._bbox_prev: np.ndarray | None = None
        self.num_fallback = 0

    def reset(self) -> None:
        """Clear cross-batch state; call before starting a new video."""
        self._bbox_prev = None
        self.num_fallback = 0

    def __call__(
        self,
        keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
        confidence: Float[torch.Tensor, 'batch num_keypoints'],
        frame_height: int,
        frame_width: int,
    ) -> Float[torch.Tensor, 'batch xyhw']:
        """Compute one bbox per frame in the batch.

        Args:
            keypoints: detector keypoints in original-frame pixel coordinates.
            confidence: detector confidence per keypoint.
            frame_height: height of the full-resolution frames (for the full-frame fallback).
            frame_width: width of the full-resolution frames.

        Returns:
            float32 tensor of shape (batch, 4) in ``[x, y, h, w]`` order, on the same device
            as ``keypoints``. Boxes are not clamped to the frame; cropping does that.
        """
        kp = keypoints[:, self.idx_anchor].detach().float().cpu().numpy()
        is_valid = np.asarray(np.isfinite(kp).all(axis=(1, 2)))
        if self.confidence_thresh is not None:
            conf = confidence[:, self.idx_anchor].detach().float().cpu().numpy()
            is_valid &= (conf >= self.confidence_thresh).all(axis=1)

        bboxes = np.zeros((len(kp), 4), dtype=np.float32)
        if is_valid.any():
            bboxes[is_valid] = compute_bbox_array(
                kp[is_valid],
                crop_ratio=self.crop_ratio,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
            )

        bbox_full = full_frame_bbox(frame_height, frame_width, 1, 'cpu')[0].numpy()
        for idx in range(len(bboxes)):
            if is_valid[idx]:
                self._bbox_prev = bboxes[idx].copy()
            else:
                bboxes[idx] = self._bbox_prev if self._bbox_prev is not None else bbox_full
                self.num_fallback += 1

        return torch.as_tensor(bboxes, dtype=torch.float32, device=keypoints.device)
