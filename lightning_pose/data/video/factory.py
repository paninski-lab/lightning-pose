"""Reader-selection dispatch: turns a backend name (or ``None``) into a built dataloader.

Single entry point: :func:`build_video_reader`. Resolves which backend to use --
either the caller's explicit choice, validated against this machine/video, or an
auto-select fallback chain (pynvvc -> dali -> opencv, richest/fastest first, most
portable last) -- then constructs and returns that backend's ready-to-iterate loader.

Adding a new video reader
-------------------------

Every backend module (``dali.py``, ``pynvvc.py``, ``opencv.py``, ...) implements the
same contract. To add one:

1. **Two-phase prepare class**: ``Prepare<Name>.__init__(model_type, filenames,
   resize_dims, dali_config, bbox_df=None, ...)`` validates inputs and precomputes
   windowing parameters (raise early, before any GPU/decoder allocation);
   ``__call__()`` builds and returns a ``Lit<Name>Wrapper``. (The ``dali_config``
   parameter name is a pre-existing wart -- ``PreparePynvvc`` already reuses
   ``cfg.dali``'s windowing section rather than inventing a per-backend config block,
   since the windowing semantics are decoder-agnostic; new backends should follow the
   same precedent rather than fixing it in isolation.)
2. **Iterator wrapper class**: ``Lit<Name>Wrapper`` is iterable, defines ``__len__``,
   and yields :class:`~lightning_pose.data.datatypes.UnlabeledBatchDict` /
   :class:`~lightning_pose.data.datatypes.MultiviewUnlabeledBatchDict` -- FCHW float
   frames, ImageNet-normalized, plus ``transforms`` and ``bbox``. Match
   ``LitDaliWrapper``/``LitPynvvcWrapper``'s shape exactly so downstream code
   (``PredictionHandler``, ``trainer.predict``) doesn't need to know which backend
   produced a batch.
3. **Availability probe (optional)**: ``is_<name>_available(video_path, ...) -> bool``,
   only needed if the backend can be installed but still unusable for a given
   machine/video (wrong hardware/driver generation, as with pynvvc). Skip it for a
   backend that's simply present or absent as a plain package dependency.
4. **Import discipline**: if the backend's package is platform-gated or proprietary
   (not a guaranteed cross-platform install), every import of it must be lazy --
   inside the function or method body that uses it, never at module top level (see
   the ``# lazy: avoids ImportError on cpu-only installs`` convention in
   ``dali.py``/``pynvvc.py``). A top-level import of such a package anywhere in the
   package or tests will break ``import lightning_pose`` on machines that don't have
   it. If the package is an unconditional, cross-platform dependency already declared
   in ``pyproject.toml`` (like ``opencv-python-headless``), top-level imports are
   fine.
5. **Wire it in**:

   - add the name to ``_Reader`` in ``lightning_pose/utils/inference_types.py`` (the
     single source for the CLI's ``--reader`` choices and API type hints);
   - add a branch -- and, if step 3 applies, a fallback rung -- in
     :func:`build_video_reader` (this file);
   - update the ``--reader`` help text in ``lightning_pose/cli/commands/predict.py``;
   - add this module to the package map in ``lightning_pose.data.video``'s docstring.
6. **Tests**: add ``tests/data/video/test_<name>.py`` mirroring a sibling reader's
   test module one-for-one, plus a case in ``tests/data/video/test_factory.py``
   covering its fallback/validation branch; mark any test that needs real hardware
   with ``@pytest.mark.gpu`` so the CPU CI workflow (``pytest -m "not gpu"``) still
   exercises everything else.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd
from omegaconf import DictConfig, ListConfig

from lightning_pose.utils.inference_types import _Reader

if TYPE_CHECKING:
    from lightning_pose.data.video.dali import LitDaliWrapper
    from lightning_pose.data.video.opencv import LitOpenCVWrapper
    from lightning_pose.data.video.pynvvc import LitPynvvcWrapper

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def build_video_reader(
    reader: _Reader | None,
    probe_video: str,
    model_type: Literal["base", "context"],
    dali_config: dict | DictConfig | ListConfig,
    filenames: list[str] | list[list[str]],
    resize_dims: list[int],
    bbox_df: pd.DataFrame | None,
) -> LitDaliWrapper | LitPynvvcWrapper | LitOpenCVWrapper:
    """Resolve which video-reading backend to use and build its dataloader.

    ``reader=None`` auto-selects the best backend usable on this machine for this
    video -- pynvvc if it's actually usable (installed + this GPU/driver + this video
    decode successfully), else dali if it's installed (``nvidia-dali-cuda110`` is
    platform-gated to Linux x86_64, so it's simply absent on macOS/Windows/other
    architectures), else opencv, an unconditional cross-platform dependency and thus
    the guaranteed final rung. An explicit ``reader`` is validated the same way and
    raises a clear error instead of a deep traceback if it isn't usable.

    Args:
        reader: explicit backend choice, or ``None`` to auto-select.
        probe_video: a real video file path used to probe hardware/file-specific
            availability (pynvvc's GPU/driver check, opencv's file-readability
            check). For multiview callers, pass the first view's video.
        model_type: ``"base"`` or ``"context"`` -- selects windowing parameters.
        dali_config: ``cfg.dali`` dict/section; only ``predict.sequence_length``
            (base and context) is used, by every backend (see
            ``lightning_pose.data.video``'s package docstring for why this
            parameter name is kept even for non-DALI backends).
        filenames: for single-view, a flat list containing one video path; for
            multiview, one single-element list per view (see each backend's
            ``Prepare<Name>`` class for the exact shape).
        resize_dims: ``[height, width]`` the model expects as input.
        bbox_df: optional per-frame bbox DataFrame; single-view only.

    Returns:
        A ready-to-iterate loader (``LitDaliWrapper``, ``LitPynvvcWrapper``, or
        ``LitOpenCVWrapper``).

    Raises:
        RuntimeError: if an explicit ``reader`` is requested but isn't usable on this
            machine/video.
    """
    if reader is None:
        from lightning_pose.data.video.pynvvc import is_pynvvc_available
        if is_pynvvc_available(probe_video):
            reader = "pynvvc"
        else:
            try:
                import lightning_pose.data.video.dali  # noqa: F401  probe importability
                reader = "dali"
            except ImportError:
                reader = "opencv"
    elif reader == "dali":
        try:
            import lightning_pose.data.video.dali  # noqa: F401  probe importability
        except ImportError as e:
            raise RuntimeError(
                "reader='dali' was requested but nvidia-dali isn't installed on this "
                "machine (it's platform-gated to Linux x86_64, so it's simply absent on "
                "macOS/Windows/other architectures). Pass reader='pynvvc'/'opencv' or "
                "omit reader to auto-select."
            ) from e
    elif reader == "pynvvc":
        from lightning_pose.data.video.pynvvc import is_pynvvc_available
        if not is_pynvvc_available(probe_video):
            raise RuntimeError(
                "reader='pynvvc' was requested but PyNvVideoCodec can't decode "
                f"{probe_video!r} on this machine (unsupported GPU generation, driver "
                "too old, pynvvideocodec not installed, an unsupported video format, "
                "or this GPU's NVDEC decoder doesn't support this video's resolution -- "
                "see the warning/debug log just above for the specific reason). Pass "
                "reader='dali'/'opencv' or omit reader to auto-select."
            )
    elif reader == "opencv":
        from lightning_pose.data.video.opencv import is_opencv_available
        if not is_opencv_available(probe_video):
            raise RuntimeError(
                f"reader='opencv' was requested but OpenCV can't open {probe_video!r} "
                "(corrupt file or unsupported codec/container). Pass a different "
                "reader or omit reader to auto-select."
            )
    logger.info(f"build_video_reader: using '{reader}' reader backend")

    if reader == "dali":
        from lightning_pose.data.video.dali import PrepareDALI  # avoids cpu-only ImportError
        vid_pred_class = PrepareDALI(
            train_stage="predict",
            model_type=model_type,
            dali_config=dali_config,
            # Important: This will be a list of lists for multiview.
            # This will trigger dali to return multiview batches to predict_step.
            filenames=filenames,
            resize_dims=resize_dims,
            bbox_df=bbox_df,
        )
    elif reader == "pynvvc":
        from lightning_pose.data.video.pynvvc import PreparePynvvc  # avoids cpu-only ImportError
        vid_pred_class = PreparePynvvc(
            model_type=model_type,
            dali_config=dali_config,
            filenames=filenames,
            resize_dims=resize_dims,
            bbox_df=bbox_df,
        )
    else:  # opencv
        from lightning_pose.data.video.opencv import PrepareOpenCV
        vid_pred_class = PrepareOpenCV(
            model_type=model_type,
            dali_config=dali_config,
            filenames=filenames,
            resize_dims=resize_dims,
            bbox_df=bbox_df,
        )
    return vid_pred_class()
