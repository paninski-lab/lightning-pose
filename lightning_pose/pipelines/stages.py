"""Reusable pipeline stages.

A stage is one step of the data flow between models in a pipeline. Stages hold only the
state they need across batches and know nothing about how the models themselves are run.
"""

import logging

import torch
from jaxtyping import Float

from lightning_pose.data.bboxes import full_frame_bbox, keypoints_to_bboxes

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
        confidence_thresh: float | None = 0.5,
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
                Defaults to 0.5.

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
        self._bbox_prev: torch.Tensor | None = None
        self._num_fallback: torch.Tensor | None = None

    @property
    def num_fallback(self) -> int:
        """Number of frames since the last :meth:`reset` that used a fallback box.

        Read this once at the end of a video: it syncs with the device.
        """
        return 0 if self._num_fallback is None else int(self._num_fallback.item())

    def reset(self) -> None:
        """Clear cross-batch state; call before starting a new video."""
        self._bbox_prev = None
        self._num_fallback = None

    def __call__(
        self,
        keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
        confidence: Float[torch.Tensor, 'batch num_keypoints'],
        frame_height: int,
        frame_width: int,
    ) -> Float[torch.Tensor, 'batch xyhw']:
        """Compute one bbox per frame in the batch, entirely on the inputs' device.

        Args:
            keypoints: detector keypoints in original-frame pixel coordinates.
            confidence: detector confidence per keypoint.
            frame_height: height of the full-resolution frames (for the full-frame fallback).
            frame_width: width of the full-resolution frames.

        Returns:
            float32 tensor of shape (batch, 4) in ``[x, y, h, w]`` order, on the same device
            as ``keypoints``. Boxes are not clamped to the frame; cropping does that.
        """
        device = keypoints.device
        kp = keypoints[:, self.idx_anchor].float()
        is_valid = torch.isfinite(kp).all(dim=2).all(dim=1)  # (batch,)
        if self.confidence_thresh is not None:
            conf = confidence[:, self.idx_anchor].float()
            is_valid = is_valid & (conf >= self.confidence_thresh).all(dim=1)

        # compute a box for every frame (invalid ones from zeros, overwritten below) so no
        # boolean indexing, and hence no device sync, is needed
        bboxes = keypoints_to_bboxes(
            torch.where(is_valid[:, None, None], kp, torch.zeros_like(kp)),
            crop_ratio=self.crop_ratio,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
        ).float()

        # forward-fill invalid frames from the most recent valid frame in this batch, else
        # from the last box of the previous batch, else the full frame
        if self._bbox_prev is None:
            self._bbox_prev = full_frame_bbox(frame_height, frame_width, 1, device)[0]
        idx_frame = torch.arange(len(bboxes), device=device)
        idx_last_valid = torch.where(is_valid, idx_frame, -torch.ones_like(idx_frame))
        idx_last_valid = torch.cummax(idx_last_valid, dim=0).values
        bboxes = torch.where(
            (idx_last_valid >= 0)[:, None],
            bboxes[idx_last_valid.clamp(min=0)],
            self._bbox_prev[None],
        )
        self._bbox_prev = bboxes[-1]

        num_invalid = (~is_valid).sum()
        self._num_fallback = num_invalid if self._num_fallback is None else (
            self._num_fallback + num_invalid
        )
        return bboxes
