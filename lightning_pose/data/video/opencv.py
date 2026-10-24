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

import cv2
import numpy as np
import pandas as pd
import torch

from lightning_pose.data.video.utils import LitPredictWrapperBase, PreparePredictBase

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


class LitOpenCVWrapper(LitPredictWrapperBase):
    """OpenCV-backed iterator for Lightning Pose video prediction.

    Mirrors ``LitPynvvcWrapper``'s public shape (iterable, ``__len__``, yields typed
    batch dicts consumable by ``trainer.predict()``) but reads sequentially from
    ``cv2.VideoCapture`` instead of indexed random access -- see module docstring for
    why. Iteration, normalization, bbox cropping and batch assembly live in
    ``LitPredictWrapperBase``; this class only decodes.

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
        bbox_df: pd.DataFrame | list[pd.DataFrame] | None = None,
    ) -> None:
        """
        Args:
            filenames: one single-element list per view, e.g. ``[["v0.mp4"]]`` for
                single-view or ``[["v0.mp4"], ["v1.mp4"]]`` for multiview -- always
                exactly one video per view (no multi-session batching in predict).
            resize_dims: see ``LitPredictWrapperBase``.
            decode_resize_dims: see ``LitPredictWrapperBase``.
            sequence_length: see ``LitPredictWrapperBase``.
            step: see ``LitPredictWrapperBase``.
            do_context: see ``LitPredictWrapperBase``.
            num_iters: see ``LitPredictWrapperBase``.
            multiview: see ``LitPredictWrapperBase``.
            bbox_df: see ``LitPredictWrapperBase``.
        """
        super().__init__(
            resize_dims=resize_dims,
            decode_resize_dims=decode_resize_dims,
            sequence_length=sequence_length,
            step=step,
            do_context=do_context,
            num_iters=num_iters,
            multiview=multiview,
            bbox_df=bbox_df,
        )
        self._caps = [cv2.VideoCapture(view_list[0]) for view_list in filenames]
        # up to 4 frames carried over from the previous window's tail, per view --
        # see module docstring for why this replaces seeking
        self._tail: list[list[np.ndarray]] = [[] for _ in self._caps]

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

    def _read_views(self) -> list[torch.Tensor]:
        """Read the next window of every view (see ``_next_window``)."""
        return [self._next_window(i) for i in range(len(self._caps))]


class PrepareOpenCV(PreparePredictBase):
    """Factory for OpenCV-backed inference dataloaders.

    Predict-only counterpart to ``lightning_pose.data.video.dali.PrepareDALI`` -- only
    needs the "predict" x {"base", "context"} combinations, since this backend never
    shuffles or augments. Validation, windowing parameters and ``num_iters`` are inherited
    from ``PreparePredictBase``; ``__call__`` builds a ready-to-iterate ``LitOpenCVWrapper``.
    """

    backend_name = 'opencv'

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
