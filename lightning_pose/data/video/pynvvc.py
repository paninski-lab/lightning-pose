"""Data pipeline for video prediction based on PyNvVideoCodec (direct NVDEC access).

Import warning
--------------
Same discipline as ``lightning_pose.data.video.dali``: ``pynvvideocodec`` is an unconditional
but platform-gated dependency (Linux x86_64 only), so importing this module at the top
level of any other module could raise ``ImportError``/``ModuleNotFoundError`` on other
platforms. Always import from this module lazily, inside the function or method body
that uses it::

    def my_function(...):
        from lightning_pose.data.video.pynvvc import PreparePynvvc  # lazy
        ...

The ``PyNvVideoCodec`` package itself is additionally deferred to inside
``LitPynvvcWrapper.__init__`` and ``is_pynvvc_available`` specifically, rather than
this module's top level, since ``import PyNvVideoCodec`` succeeds regardless of GPU
generation or driver age (Turing/Ampere/Ada/Hopper/Blackwell only, driver >= 530.41.03
on Linux) -- the failure only surfaces when a decoder is actually constructed.

Architecture overview
----------------------
Mirrors ``lightning_pose.data.video.dali``'s two-phase construction: ``PreparePynvvc.__init__``
validates inputs and precomputes windowing parameters; calling the instance
(``__call__``) builds and returns a ready-to-iterate ``LitPynvvcWrapper``. Both classes subclass
``PreparePredictBase``/``LitPredictWrapperBase`` in ``lightning_pose.data.video.utils``, which
hold everything that is not specific to NVDEC (validation, windowing, ``num_iters``,
normalization, per-view bbox cropping, batch assembly); this module implements only decoding
and the CUDA synchronization below.

Predict-only, unlike ``PrepareDALI``: this backend never runs ``random_shuffle`` or the
``imgaug`` augmentation pipeline, since ``litpose predict`` already runs with both off.
Training continues to go through DALI exclusively (``lightning_pose.data.video.dali``).

CUDA stream synchronization
----------------------------
``PyNvVideoCodec.SimpleDecoder`` writes decoded frames asynchronously on its own
internal CUDA stream and hands them back as DLPack buffers. If a consumer (this
module's own resize/normalize step, or eventually the model's forward pass) reads
those buffers on a different stream before the decode is actually finished, the read
happens on stale/partial data -- silently. No crash, no NaN, just quietly wrong
pixels feeding a plausible-looking keypoint prediction. ``LitPynvvcWrapper._read_views``
guards against this with an explicit ``torch.cuda.current_stream().synchronize()``
after building each batch's frame tensors and before they are handed to the shared
normalization/cropping code in ``LitPredictWrapperBase.__next__``. This is the
"safe but not necessarily optimal" fix flagged in issue #476 -- passing an explicit
stream handle into the decoder constructor (if supported) would avoid the sync cost
entirely, but that needs checking against the installed PyNvVideoCodec version and is
tracked as a follow-up, not done here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig

from lightning_pose.data.video.utils import LitPredictWrapperBase, PreparePredictBase

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
    construction *and* a single-frame decode against ``video_path``, since some
    NVDEC hardware-support failures (e.g. a frame's macroblock count exceeding this
    GPU's supported maximum) only surface on first decode, inside NVDEC's
    ``HandleVideoSequence`` parser callback -- constructing a ``SimpleDecoder`` alone
    does not trigger it. Intended to be called once per predict call (e.g. by the
    CLI/API's auto-selection logic), not per-batch -- constructing a ``SimpleDecoder``
    and decoding a frame has real setup cost.

    Args:
        video_path: path to a real, existing video file to probe decodability with.
        gpu_id: CUDA device index to attempt decoding on.

    Returns:
        True if a decoder was constructed and a frame decoded successfully, False on
        any exception (unsupported GPU generation, driver too old, corrupt/unsupported
        container, unsupported resolution/macroblock count, pynvvideocodec not
        installed, etc.) -- deliberately broad, since every failure mode here should
        fall back to DALI rather than propagate. The three failure stages below are
        logged at different levels: package-not-installed and decoder-construction
        failures are common/expected (e.g. on a machine with no NVDEC-capable GPU) and
        logged at ``debug``; a decode failure after successful construction means
        PyNvVideoCodec *is* usable on this machine in general but not for this
        specific video, which is a more actionable, less expected situation, so it's
        logged at ``warning`` with the underlying exception message included.
    """
    try:
        # Same reason as the module-level TYPE_CHECKING import above -- not on a
        # CI-reachable index. Also lazy at runtime to avoid ImportError on cpu-only
        # installs.
        import PyNvVideoCodec as nvc  # pyright: ignore[reportMissingImports]
    except Exception:
        logger.debug("PyNvVideoCodec is not installed", exc_info=True)
        return False

    try:
        decoder = nvc.SimpleDecoder(video_path, gpu_id=gpu_id, use_device_memory=True)
    except Exception:
        logger.debug("pynvvc decoder construction failed", exc_info=True)
        return False

    try:
        decoder.get_batch_frames_by_index([0])
    except Exception as e:
        logger.warning(
            f"PyNvVideoCodec is installed and its decoder constructed successfully for "
            f"{video_path!r}, but failed to decode a frame from it: {e}. This usually "
            "means this GPU's NVDEC hardware decoder doesn't support this video's "
            "resolution/macroblock count (see the 'MBCount not supported' error above, "
            "if present), not that PyNvVideoCodec is misconfigured. Falling back to a "
            "different video-reading backend for this video."
        )
        return False
    finally:
        del decoder
    return True


class LitPynvvcWrapper(LitPredictWrapperBase):
    """PyNvVideoCodec-backed iterator for Lightning Pose video prediction.

    Mirrors ``LitDaliWrapper``'s public shape (iterable, ``__len__``, yields typed
    batch dicts consumable by ``trainer.predict()``) but is not a
    ``DALIGenericIterator`` subclass -- built directly on one
    ``PyNvVideoCodec.SimpleDecoder`` per view instead of a DALI pipeline. Iteration,
    normalization, bbox cropping and batch assembly live in ``LitPredictWrapperBase``;
    this class only decodes.

    Predict-only: no ``random_shuffle``, no ``imgaug``. See
    ``lightning_pose.data.video.dali.LitDaliWrapper`` for the train-time DALI path, which
    this class does not replicate.
    """

    def __init__(
        self,
        filenames: list[list[str]],
        resize_dims: list[int] | None,
        decode_resize_dims: list[int] | None,
        sequence_length: int,
        step: int,
        do_context: bool,
        num_iters: int,
        multiview: bool,
        bbox_df: pd.DataFrame | list[pd.DataFrame] | None = None,
        gpu_id: int = 0,
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
            gpu_id: CUDA device index to decode on.
        """
        # Same reason as the module-level TYPE_CHECKING import above -- not on a
        # CI-reachable index. Also lazy at runtime to avoid ImportError on cpu-only
        # installs.
        import PyNvVideoCodec as nvc  # pyright: ignore[reportMissingImports]

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

    def _read_views(self) -> list[torch.Tensor]:
        """Read the next window of every view and advance the decode cursor."""
        per_view_frames = [self._read_window(dec) for dec in self._decoders]
        # Synchronize each decoder's internal CUDA stream against the current stream
        # before anything (including our own resize/normalize) reads the
        # tensors. See module docstring -- this is the safe-but-costly guard against
        # silently reading a not-yet-finished decode.
        torch.cuda.current_stream().synchronize()

        self._cursor += self.step
        return per_view_frames


class PreparePynvvc(PreparePredictBase):
    """Factory for PyNvVideoCodec-backed inference dataloaders.

    Predict-only counterpart to ``lightning_pose.data.video.dali.PrepareDALI`` -- only
    needs the "predict" x {"base", "context"} combinations, since NVDEC inference
    never shuffles or augments. Validation, windowing parameters and ``num_iters`` are
    inherited from ``PreparePredictBase``; ``__call__`` builds a ready-to-iterate
    ``LitPynvvcWrapper``.
    """

    backend_name = 'pynvvc'

    def __init__(
        self,
        model_type: Literal['base', 'context'],
        filenames: list[str] | list[list[str]],
        resize_dims: list[int] | None,
        dali_config: dict | DictConfig | ListConfig,
        gpu_id: int = 0,
        bbox_df: pd.DataFrame | list[pd.DataFrame] | None = None,
    ) -> None:
        """
        Args:
            model_type: see ``PreparePredictBase``.
            filenames: see ``PreparePredictBase``.
            resize_dims: see ``PreparePredictBase``.
            dali_config: see ``PreparePredictBase``.
            gpu_id: CUDA device index to decode on.
            bbox_df: see ``PreparePredictBase``.

        Raises:
            FileNotFoundError, NotImplementedError, ValueError: see ``PreparePredictBase``.
        """
        super().__init__(model_type, filenames, resize_dims, dali_config, bbox_df=bbox_df)
        self.gpu_id = gpu_id

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
