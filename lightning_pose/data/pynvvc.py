"""Data pipeline for video prediction based on PyNvVideoCodec (direct NVDEC access).

Import warning
--------------
Same discipline as ``lightning_pose.data.dali``: ``pynvvideocodec`` is an unconditional
but platform-gated dependency (Linux x86_64 only), so importing this module at the top
level of any other module could raise ``ImportError``/``ModuleNotFoundError`` on other
platforms. Always import from this module lazily, inside the function or method body
that uses it::

    def my_function(...):
        from lightning_pose.data.pynvvc import PreparePynvvc  # lazy
        ...

The ``PyNvVideoCodec`` package itself is additionally deferred to inside
``LitPynvvcWrapper.__init__`` and ``is_pynvvc_available`` specifically, rather than
this module's top level, since ``import PyNvVideoCodec`` succeeds regardless of GPU
generation or driver age (Turing/Ampere/Ada/Hopper/Blackwell only, driver >= 530.41.03
on Linux) -- the failure only surfaces when a decoder is actually constructed.

Architecture overview
----------------------
Mirrors ``lightning_pose.data.dali``'s two-phase construction: ``PreparePynvvc.__init__``
validates inputs and precomputes windowing parameters; calling the instance
(``__call__``) builds and returns a ready-to-iterate ``LitPynvvcWrapper``.

Predict-only, unlike ``PrepareDALI``: this backend never runs ``random_shuffle`` or the
``imgaug`` augmentation pipeline, since ``litpose predict`` already runs with both off.
Training continues to go through DALI exclusively (``lightning_pose.data.dali``).

CUDA stream synchronization
----------------------------
``PyNvVideoCodec.SimpleDecoder`` writes decoded frames asynchronously on its own
internal CUDA stream and hands them back as DLPack buffers. If a consumer (this
module's own resize/normalize step, or eventually the model's forward pass) reads
those buffers on a different stream before the decode is actually finished, the read
happens on stale/partial data -- silently. No crash, no NaN, just quietly wrong
pixels feeding a plausible-looking keypoint prediction. ``LitPynvvcWrapper.__next__``
guards against this with an explicit ``torch.cuda.current_stream().synchronize()``
after building each batch's frame tensors and before returning them. This is the
"safe but not necessarily optimal" fix flagged in issue #476 -- passing an explicit
stream handle into the decoder constructor (if supported) would avoid the sync cost
entirely, but that needs checking against the installed PyNvVideoCodec version and is
tracked as a follow-up, not done here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    # PyNvVideoCodec is NVIDIA's proprietary decoder package -- not on a
    # CI-reachable index, so pyright can never resolve it here (see module
    # docstring for the runtime lazy-import discipline this mirrors).
    import PyNvVideoCodec as nvc  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def is_pynvvc_available(video_path: str, gpu_id: int = 0) -> bool:
    """Best-effort check whether the pynvvc decoder backend can actually decode.

    Unlike a plain ``import PyNvVideoCodec`` (which succeeds regardless of GPU
    generation or driver age -- see module docstring), this attempts a real decoder
    construction against ``video_path``, since NVDEC hardware-support failures only
    surface there, not at import time. Intended to be called once per predict call
    (e.g. by the CLI/API's auto-selection logic), not per-batch -- constructing a
    ``SimpleDecoder`` has real setup cost.

    Args:
        video_path: path to a real, existing video file to probe decodability with.
        gpu_id: CUDA device index to attempt decoding on.

    Returns:
        True if a decoder was constructed successfully, False on any exception
        (unsupported GPU generation, driver too old, corrupt/unsupported container,
        pynvvideocodec not installed, etc.) -- deliberately broad, since every
        failure mode here should fall back to DALI rather than propagate.
    """
    try:
        # Same reason as the module-level TYPE_CHECKING import above -- not on a
        # CI-reachable index. Also lazy at runtime to avoid ImportError on cpu-only
        # installs.
        import PyNvVideoCodec as nvc  # pyright: ignore[reportMissingImports]

        decoder = nvc.SimpleDecoder(video_path, gpu_id=gpu_id, use_device_memory=True)
        del decoder
        return True
    except Exception:
        logger.debug("pynvvc decoder unavailable", exc_info=True)
        return False


class LitPynvvcWrapper:
    """PyNvVideoCodec-backed iterator for Lightning Pose video prediction.

    Mirrors ``LitDaliWrapper``'s public shape (iterable, ``__len__``, yields typed
    batch dicts consumable by ``trainer.predict()``) but is not a
    ``DALIGenericIterator`` subclass -- built directly on one
    ``PyNvVideoCodec.SimpleDecoder`` per view instead of a DALI pipeline.

    Predict-only: no ``random_shuffle``, no ``imgaug``. See
    ``lightning_pose.data.dali.LitDaliWrapper`` for the train-time DALI path, which
    this class does not replicate.
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
        gpu_id: int = 0,
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
                by ``PreparePynvvc.num_iters`` so Lightning can report progress.
            multiview: whether this is a multiview prediction (one decoder per view,
                read in lockstep since predict never shuffles).
            bbox_df: optional per-frame bbox DataFrame (columns x, y, h, w); single-
                view only (enforced upstream in ``predict_video``).
            gpu_id: CUDA device index to decode on.
        """
        # Same reason as the module-level TYPE_CHECKING import above -- not on a
        # CI-reachable index. Also lazy at runtime to avoid ImportError on cpu-only
        # installs.
        import PyNvVideoCodec as nvc  # pyright: ignore[reportMissingImports]

        self.resize_dims = resize_dims
        self.decode_resize_dims = decode_resize_dims
        self.sequence_length = sequence_length
        self.step = step
        self.do_context = do_context
        self.num_iters = num_iters
        self.multiview = multiview
        self.bbox_df = bbox_df
        self.gpu_id = gpu_id

        self._decoders = [
            nvc.SimpleDecoder(
                view_list[0],
                gpu_id=gpu_id,
                use_device_memory=True,
                # planar CHW per frame, so stacking sequence_length frames gives
                # (seq_len, C, H, W) directly -- matches DALI's FCHW output layout.
                output_color_type=nvc.OutputColorType.RGBP,  # type: ignore[attr-defined]
            )
            for view_list in filenames
        ]
        self._total_frames = len(self._decoders[0])

        self._cursor = 0
        self._iters_done = 0
        # cursor into bbox_df; advances by the context-adjusted step per batch, same
        # convention as LitDaliWrapper._frame_idx
        self._frame_idx = 0

    def __len__(self) -> int:
        """Return the number of iterations (batches) in this dataloader."""
        return self.num_iters

    def __iter__(self) -> LitPynvvcWrapper:
        return self

    def _read_window(self, decoder: nvc.SimpleDecoder) -> torch.Tensor:
        """Read ``sequence_length`` frames starting at ``self._cursor``.

        Pads the final window by repeating the last real frame so every batch has a
        static shape (matches DALI's ``pad_sequences=True`` / ``LastBatchPolicy.FILL``
        behavior) -- this matters for torch.compile, which would otherwise eat a
        recompilation stall on the last, oddly-shaped batch of every video.

        Uses ``get_batch_frames_by_index`` rather than slice indexing because
        PyNvVideoCodec 2.1.0's ``SimpleDecoder.__getitem__`` does not support
        ordinary Python slices whose step is omitted.
        """
        end = min(self._cursor + self.sequence_length, self._total_frames)
        indices = list(range(self._cursor, end))
        frames = list(decoder.get_batch_frames_by_index(indices)) if indices else []

        n_missing = self.sequence_length - len(frames)
        if len(frames) == 0:
            frames = [decoder[self._total_frames - 1]]
            n_missing -= 1
        if n_missing > 0:
            frames = frames + [frames[-1]] * n_missing
        return torch.stack([torch.from_dlpack(f) for f in frames])  # (seq_len, C, H, W)

    def _resize_normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (seq_len, C, H, W), decoder output (assumed uint8 -- verify on
        first real run; if the installed PyNvVideoCodec version returns a different
        dtype for RGBP output this silently produces a wrong [0,1] scale).

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

        per_view_frames = [self._read_window(dec) for dec in self._decoders]
        # Synchronize each decoder's internal CUDA stream against the current stream
        # before anything (including our own resize/normalize below) reads the
        # tensors. See module docstring -- this is the safe-but-costly guard against
        # silently reading a not-yet-finished decode.
        torch.cuda.current_stream().synchronize()

        self._cursor += self.step

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
            num_views = len(self._decoders)
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


class PreparePynvvc:
    """Factory for PyNvVideoCodec-backed inference dataloaders.

    Predict-only counterpart to ``lightning_pose.data.dali.PrepareDALI`` -- only
    needs the "predict" x {"base", "context"} combinations, since NVDEC inference
    never shuffles or augments. Construction is split the same way as ``PrepareDALI``:
    ``__init__`` validates inputs and precomputes windowing parameters; ``__call__``
    builds and returns a ready-to-iterate ``LitPynvvcWrapper``.

    Sequence-length source (open design question, flag for review): reuses
    ``dali_config["base"]["predict"]["sequence_length"]`` /
    ``dali_config["context"]["predict"]["sequence_length"]`` (i.e. the existing
    ``cfg.dali`` section) rather than introducing a parallel ``cfg.pynvvc`` config
    block, since the windowing semantics are decoder-agnostic. Worth confirming this
    is actually what's wanted rather than a separate config section.
    """

    def __init__(
        self,
        model_type: Literal["base", "context"],
        filenames: list[str] | list[list[str]],
        resize_dims: list[int],
        dali_config: dict | DictConfig | ListConfig,
        gpu_id: int = 0,
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
            dali_config: same ``cfg.dali`` dict ``PrepareDALI`` reads from -- only
                the ``predict.sequence_length`` values (base and context) are used.
                See class docstring for why this reuses the DALI config section.
            gpu_id: CUDA device index to decode on.
            bbox_df: optional DataFrame with columns ``["x", "y", "h", "w"]``, one
                row per frame. When provided, frames are decoded at full resolution
                and ``LitPynvvcWrapper`` crops each to its bbox before resizing.

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
                    "pynvvc predict backend supports exactly one video per view "
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
                        "Mismatched frame counts across views; multiview pynvvc reading "
                        f"would desynchronize. view 0={view0_frame_count}, "
                        f"view {view_idx}={frame_count}"
                    )

        self.model_type = model_type
        self.filenames = filenames_2d
        self.resize_dims = resize_dims
        self.gpu_id = gpu_id
        self.bbox_df = bbox_df
        self.frame_count = view0_frame_count

        if model_type == "base":
            self.sequence_length = dali_config["base"]["predict"]["sequence_length"]  # type: ignore[arg-type]
            self.step = self.sequence_length
            self.do_context = False
        elif model_type == "context":
            self.sequence_length = dali_config["context"]["predict"]["sequence_length"]  # type: ignore[arg-type]
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

        Mirrors the specific branch of ``PrepareDALI.num_iters`` that applies to
        prediction (single sequence at a time, batch_size=1): for context models
        this is the "step == sequence_length - 4" case.
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

    def __call__(self) -> LitPynvvcWrapper:
        """Build and return a ready-to-iterate ``LitPynvvcWrapper``."""
        return LitPynvvcWrapper(
            filenames=self.filenames,
            resize_dims=self.resize_dims,
            decode_resize_dims=self._decode_resize_dims,
            sequence_length=self.sequence_length,
            step=self.step,
            do_context=self.do_context,
            num_iters=self.num_iters,
            multiview=self.multiview,
            bbox_df=self.bbox_df,
            gpu_id=self.gpu_id,
        )
