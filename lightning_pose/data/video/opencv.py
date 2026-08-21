"""Data pipeline for video prediction based on OpenCV (``cv2.VideoCapture``).

Import discipline
------------------
Unlike ``dali.py``/``pynvvc.py``, ``cv2`` (``opencv-python-headless``) is an unconditional,
cross-platform dependency already declared in ``pyproject.toml`` and already imported at module
top level elsewhere in this package (e.g. ``lightning_pose.data.cameras``,
``lightning_pose.utils.predictions``). So, unlike those two backends, this module imports ``cv2``
eagerly; no lazy-import discipline is needed here (see ``lightning_pose.data.video``'s package
docstring for when lazy imports are required).

Architecture overview
----------------------
Mirrors ``lightning_pose.data.video.pynvvc``'s two-phase construction: ``PrepareOpenCV.__init__``
validates inputs and precomputes windowing parameters; calling the instance (``__call__``) builds
and returns a ready-to-iterate ``LitOpenCVWrapper``.

Predict-only, like pynvvc: this backend never runs ``random_shuffle`` or the ``imgaug``
augmentation pipeline, since ``litpose predict`` already runs with both off. Training continues
to go through DALI exclusively (``lightning_pose.data.video.dali``).

Sequential decode, not seek-based
-----------------------------------
Context (MHCRNN) models read overlapping windows: consecutive ``sequence_length``-frame windows
advance by ``step = sequence_length - 4``, so the last 4 frames of window *i* are the same
physical frames as the first 4 of window *i+1*. ``cv2.VideoCapture``'s ``CAP_PROP_POS_FRAMES``
seeking is not reliably frame-exact on containers with B-frames or a variable/estimated frame
rate -- a seek-per-window implementation risks silently misaligning that overlap: no crash, no
shape change, just a temporally-shifted context stack feeding a plausible-but-wrong prediction.

To avoid that, ``LitOpenCVWrapper`` never seeks: it opens each view's ``cv2.VideoCapture`` once
and reads strictly forward via ``cap.read()``, caching the trailing 4 frames of each window in
``self._tail`` to prepend to the next one. Every physical frame is decoded exactly once (aside
from the cached tail).
"""

from __future__ import annotations

import os
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.bboxes import crop_and_resize_frames
from lightning_pose.data.datatypes import (
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import count_frames

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def is_opencv_available(video_path: str) -> bool:
    """Best-effort check whether OpenCV can open ``video_path`` for reading.

    Cheap relative to ``is_pynvvc_available`` -- no real decoder construction, no
    CUDA -- since ``opencv-python-headless`` is an unconditional dependency, this only
    needs to catch a bad/corrupt/unsupported file, not an availability question about
    the package itself. Used to give a clear error when a user explicitly requests
    ``--reader opencv`` against an unreadable file; not used to gate the auto-select
    fallback chain, where opencv is the unconditional final rung (see
    ``lightning_pose.utils.predictions.predict_video``).

    Args:
        video_path: path to a real, existing video file to probe.

    Returns:
        True if ``cv2.VideoCapture`` can open the file, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        return cap.isOpened()
    finally:
        cap.release()


class LitOpenCVWrapper:
    """OpenCV-backed iterator for Lightning Pose video prediction.

    Mirrors ``LitPynvvcWrapper``'s public shape (iterable, ``__len__``, yields typed
    batch dicts consumable by ``trainer.predict()``) but reads sequentially from
    ``cv2.VideoCapture`` instead of indexed random access -- see module docstring for
    why.

    Predict-only: no ``random_shuffle``, no ``imgaug``. See
    ``lightning_pose.data.video.dali.LitDaliWrapper`` for the train-time DALI path,
    which this class does not replicate.
    """

    def __init__(
        self,
        filenames: list[list[str]],
        resize_dims: list[int],
        decode_resize_dims: list[int] | None,
        sequence_length: int,
        step: int,
        do_context: bool,
        num_iters: int,
        multiview: bool,
        bbox_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Args:
            filenames: one single-element list per view, e.g. ``[["v0.mp4"]]`` for
                single-view or ``[["v0.mp4"], ["v1.mp4"]]`` for multiview -- always
                exactly one video per view (no multi-session batching in predict).
            resize_dims: ``[height, width]`` the model expects as input. Used as the
                post-crop resize target when ``bbox_df`` is set.
            decode_resize_dims: ``[height, width]`` to resize decoded frames to
                immediately after decode. ``None`` in bbox-crop mode (full-resolution
                frames are needed for cropping); equal to ``resize_dims`` otherwise.
            sequence_length: number of frames per batch/window.
            step: frames to advance the read cursor between windows. Equals
                ``sequence_length`` for non-overlapping ("base") windows, or
                ``sequence_length - 4`` for context models' overlapping windows.
            do_context: whether this is a 5-frame-context (MHCRNN-style) model.
            num_iters: total number of batches this video will produce; precomputed
                by ``PrepareOpenCV.num_iters`` so Lightning can report progress.
            multiview: whether this is a multiview prediction (one capture per view,
                read in lockstep since predict never shuffles).
            bbox_df: optional per-frame bbox DataFrame (columns x, y, h, w); single-
                view only (enforced upstream in ``predict_video``).
        """
        self.resize_dims = resize_dims
        self.decode_resize_dims = decode_resize_dims
        self.sequence_length = sequence_length
        self.step = step
        self.do_context = do_context
        self.num_iters = num_iters
        self.multiview = multiview
        self.bbox_df = bbox_df

        self._caps = [cv2.VideoCapture(view_list[0]) for view_list in filenames]
        # up to 4 frames carried over from the previous window's tail, per view --
        # see module docstring for why this replaces seeking
        self._tail: list[list[np.ndarray]] = [[] for _ in self._caps]

        self._iters_done = 0
        # cursor into bbox_df; advances by the context-adjusted step per batch, same
        # convention as LitDaliWrapper._frame_idx / LitPynvvcWrapper._frame_idx
        self._frame_idx = 0

    def __len__(self) -> int:
        """Return the number of iterations (batches) in this dataloader."""
        return self.num_iters

    def __iter__(self) -> LitOpenCVWrapper:
        return self

    def _read_new_frames(self, cap: cv2.VideoCapture, n: int) -> list[np.ndarray]:
        """Read up to ``n`` frames sequentially from ``cap``; returns fewer at EOF."""
        frames: list[np.ndarray] = []
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)  # HWC, BGR, uint8
        return frames

    def _next_window(self, cap_idx: int) -> torch.Tensor:
        """Read the next ``sequence_length``-frame window for one view.

        Prepends any tail carried over from the previous window (context models
        only), reads just the new frames needed to fill out ``sequence_length``, pads
        by repeating the last frame at end-of-video (matches DALI's
        ``pad_sequences=True`` / ``LastBatchPolicy.FILL`` and ``LitPynvvcWrapper``'s
        equivalent padding -- needed so every batch has a static shape, which matters
        for torch.compile), and -- for context models -- caches the window's last 4
        frames as the tail for next time.
        """
        tail = self._tail[cap_idx]
        n_new = self.sequence_length - len(tail)
        new_frames = self._read_new_frames(self._caps[cap_idx], n_new)
        window = tail + new_frames

        n_missing = self.sequence_length - len(window)
        if n_missing > 0:
            if not window:
                raise RuntimeError(
                    f"opencv reader exhausted view {cap_idx} with no frames decoded for this "
                    "window; this indicates a mismatch between count_frames() and the actual "
                    "decodable frame count."
                )
            window = window + [window[-1]] * n_missing

        self._tail[cap_idx] = window[-4:] if self.do_context else []

        # HWC BGR uint8 -> (seq_len, C, H, W) RGB, matching the other backends' output layout
        frames_np = np.stack(window)  # (seq_len, H, W, C)
        frames_bgr = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (seq_len, C, H, W)
        return frames_bgr[:, [2, 1, 0], :, :]  # BGR -> RGB

    def _resize_normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (seq_len, C, H, W), uint8, RGB.

        Returns float, optionally resized to ``decode_resize_dims``, [0,1]-scaled,
        then ImageNet-normalized -- same order of operations as DALI's
        ``fn.resize`` -> ``fn.crop_mirror_normalize`` in ``dali.py:video_pipe``.
        """
        frames = frames.float() / 255.0
        if self.decode_resize_dims is not None:
            frames = F.interpolate(
                frames, size=self.decode_resize_dims, mode='bilinear', align_corners=False,
            )
        mean = torch.tensor(_IMAGENET_MEAN, device=frames.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=frames.device).view(1, 3, 1, 1)
        return (frames - mean) / std

    def __next__(self) -> UnlabeledBatchDict | MultiviewUnlabeledBatchDict:
        """Fetch the next batch, applying per-frame bbox crop+resize when configured."""
        if self._iters_done >= self.num_iters:
            raise StopIteration
        self._iters_done += 1

        per_view_frames = [self._next_window(i) for i in range(len(self._caps))]
        processed = [self._resize_normalize(f) for f in per_view_frames]

        if not self.multiview:
            frames = processed[0]
            height, width = frames.shape[-2], frames.shape[-1]

            if self.bbox_df is not None:
                assert self.resize_dims is not None  # required whenever bbox_df is set
                step = self.sequence_length - 4 if self.do_context else self.sequence_length
                rows = self.bbox_df.iloc[self._frame_idx:self._frame_idx + self.sequence_length]
                if len(rows) < self.sequence_length:
                    last_row = self.bbox_df.iloc[[-1]]
                    rows = pd.concat(
                        [rows] + [last_row] * (self.sequence_length - len(rows)),
                        ignore_index=True,
                    )
                cropped_frames, bboxes = crop_and_resize_frames(frames, rows, self.resize_dims)
                self._frame_idx += step
                return UnlabeledBatchDict(
                    frames=cropped_frames,
                    transforms=torch.tensor([-1.0]),
                    bbox=bboxes,
                    is_multiview=False,
                )

            bbox = torch.tensor(
                [0, 0, height, width], device=frames.device, dtype=torch.float32,
            ).repeat(frames.shape[0], 1)
            return UnlabeledBatchDict(
                frames=frames,
                transforms=torch.tensor([-1.0]),
                bbox=bbox,
                is_multiview=False,
            )

        else:
            frames = torch.stack(processed, dim=1)  # (seq_len, num_views, C, H, W)
            height, width = frames.shape[-2], frames.shape[-1]
            num_views = len(self._caps)
            bbox_per_view = torch.tensor(
                [0, 0, height, width], device=frames.device, dtype=torch.float32,
            )
            bbox = bbox_per_view.repeat(num_views).unsqueeze(0).repeat(frames.shape[0], 1)
            transforms = torch.tensor([-1.0]).repeat(num_views, 1, 1)
            return MultiviewUnlabeledBatchDict(
                frames=frames,
                transforms=transforms,
                bbox=bbox,
                is_multiview=True,
            )


class PrepareOpenCV:
    """Factory for OpenCV-backed inference dataloaders.

    Predict-only counterpart to ``lightning_pose.data.video.dali.PrepareDALI`` -- only
    needs the "predict" x {"base", "context"} combinations, since this backend never
    shuffles or augments. Construction is split the same way as ``PrepareDALI``/
    ``PreparePynvvc``: ``__init__`` validates inputs and precomputes windowing
    parameters; ``__call__`` builds and returns a ready-to-iterate ``LitOpenCVWrapper``.

    Sequence-length source: reuses ``dali_config["base"]["predict"]["sequence_length"]``
    / ``dali_config["context"]["predict"]["sequence_length"]`` (i.e. the existing
    ``cfg.dali`` section), same precedent as ``PreparePynvvc`` -- see
    ``lightning_pose.data.video``'s package docstring for why this parameter name is
    kept even though it isn't DALI-specific.
    """

    def __init__(
        self,
        model_type: Literal["base", "context"],
        filenames: list[str] | list[list[str]],
        resize_dims: list[int],
        dali_config: dict | DictConfig | ListConfig,
        bbox_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Args:
            model_type: ``"base"`` for standard single-frame models, ``"context"``
                for MHCRNN models that consume a temporal window.
            filenames: for single-view, a single video path (as a length-1 list or
                bare string); for multi-view, one video path per view.
            resize_dims: ``[height, width]`` to resize frames to before feeding the
                model. Also used as the post-crop resize target when ``bbox_df`` is
                provided.
            dali_config: same ``cfg.dali`` dict ``PrepareDALI``/``PreparePynvvc`` read
                from -- only the ``predict.sequence_length`` values (base and context)
                are used.
            bbox_df: optional DataFrame with columns ``["x", "y", "h", "w"]``, one
                row per frame. When provided, frames are decoded at full resolution
                and ``LitOpenCVWrapper`` crops each to its bbox before resizing.

        Raises:
            FileNotFoundError: if any path in ``filenames`` does not exist or is not
                a file.
            NotImplementedError: if any view supplies more than one video (multi-
                session batching is a DALI-train-only concept, not needed here).
            ValueError: for multiview inputs, if views have differing frame counts
                (which would desynchronize the per-view sequential reads), or for an
                unknown ``model_type``.
        """
        if isinstance(filenames, list) and isinstance(filenames[0], list):
            self.multiview = True
        else:
            self.multiview = False

        filenames_2d: list[list[str]]
        if isinstance(filenames[0], str):
            filenames_2d = [filenames]  # type: ignore[list-item]
        else:
            filenames_2d = filenames  # type: ignore[assignment]

        for view_list in filenames_2d:
            if len(view_list) != 1:
                raise NotImplementedError(
                    "opencv predict backend supports exactly one video per view "
                    f"(no multi-session batching); got {len(view_list)} videos for one view."
                )
            vid = view_list[0]
            if not os.path.exists(vid) or not os.path.isfile(vid):
                raise FileNotFoundError(f"{vid} is not a video file!")

        view0_frame_count = count_frames(filenames_2d[0][0])
        if self.multiview:
            for view_idx, view_list in enumerate(filenames_2d[1:], start=1):
                frame_count = count_frames(view_list[0])
                if frame_count != view0_frame_count:
                    raise ValueError(
                        "Mismatched frame counts across views; multiview opencv reading "
                        f"would desynchronize. view 0={view0_frame_count}, "
                        f"view {view_idx}={frame_count}"
                    )

        self.model_type = model_type
        self.filenames = filenames_2d
        self.resize_dims = resize_dims
        self.bbox_df = bbox_df
        self.frame_count = view0_frame_count

        if model_type == "base":
            predict_cfg = dali_config["base"]["predict"]  # type: ignore[index]
            self.sequence_length = predict_cfg["sequence_length"]
            self.step = self.sequence_length
            self.do_context = False
        elif model_type == "context":
            predict_cfg = dali_config["context"]["predict"]  # type: ignore[index]
            self.sequence_length = predict_cfg["sequence_length"]
            self.step = self.sequence_length - 4
            self.do_context = True
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        # bbox-crop mode needs full-resolution decoded frames to crop from; the
        # post-crop resize to resize_dims happens in crop_and_resize_frames instead.
        self._decode_resize_dims: list[int] | None = (
            None if self.bbox_df is not None else resize_dims
        )

    @property
    def num_iters(self) -> int:
        """Number of dataloader iterations required to process all frames.

        Identical formula to ``PreparePynvvc.num_iters`` (single sequence at a time,
        batch_size=1): for context models this is the "step == sequence_length - 4"
        case.
        """
        if self.model_type == "base":
            return int(np.ceil(self.frame_count / self.sequence_length))
        else:  # context
            if self.step <= 0:
                raise ValueError(
                    "step cannot be 0, please modify "
                    "cfg.dali.context.predict.sequence_length to be > 4"
                )
            data_except_first_batch = self.frame_count - self.sequence_length
            return int(np.ceil(data_except_first_batch / self.step)) + 1

    def __call__(self) -> LitOpenCVWrapper:
        """Build and return a ready-to-iterate ``LitOpenCVWrapper``."""
        return LitOpenCVWrapper(
            filenames=self.filenames,
            resize_dims=self.resize_dims,
            decode_resize_dims=self._decode_resize_dims,
            sequence_length=self.sequence_length,
            step=self.step,
            do_context=self.do_context,
            num_iters=self.num_iters,
            multiview=self.multiview,
            bbox_df=self.bbox_df,
        )
