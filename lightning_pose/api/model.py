"""High-level Model class for loading trained checkpoints and running inference."""

from __future__ import annotations

import copy
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Literal, cast, get_args

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from lightning_pose.api.model_config import ModelConfig
from lightning_pose.data import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)
from lightning_pose.data.bboxes import model_to_frame_batch
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datatypes import MultiviewPredictionResult, PredictionResult
from lightning_pose.metrics import compute_metrics_single
from lightning_pose.models import ALLOWED_MODELS, get_model_class
from lightning_pose.utils import io as io_utils
from lightning_pose.utils.predictions import generate_labeled_video as generate_labeled_video_fn
from lightning_pose.utils.predictions import (
    predict_dataset,
    predict_video,
)

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []

# User-facing precision strings for both the CLI (--precision) and this API
# (Model.from_dir(precision=...)). Kept identical across CLI and API so a value
# can be passed straight through without a separate lookup table at the CLI layer.
_Precision = Literal["fp32", "fp16", "bf16"]

# The subset of PyTorch Lightning's own _PRECISION_INPUT that we actually use.
# Narrower than plain str so a value returned from here type-checks directly
# against pl.Trainer(precision=...).
_PLPrecision = Literal["32-true", "16-mixed", "bf16-mixed"]

# Internal-only: maps our user-facing precision strings to the strings
# PyTorch Lightning's Trainer(precision=...) actually expects.
_PRECISION_TO_PL: dict[_Precision, _PLPrecision] = {
    "fp32": "32-true",
    "fp16": "16-mixed",
    "bf16": "bf16-mixed",
}

# Maps our precision strings to the torch dtype used for ``torch.autocast`` in
# code paths that don't go through a ``pl.Trainer`` (e.g. ``Model.predict_frame``).
# "fp32" needs no entry -- no autocast.
_PRECISION_TO_AUTOCAST_DTYPE: dict[_Precision, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Temporal context length used when tracing a context (MHCRNN) model for export.
# Matches the (T, H, W, 3) convention documented in Model.predict_frame.
_CONTEXT_SEQUENCE_LENGTH = 5

# Weight precisions supported for ONNX export. Deliberately narrower than
# _Precision -- this is the precision baked into the .onnx file itself, not the
# autocast precision used for eager inference.
_OnnxPrecision = Literal["fp32", "fp16"]

# Inference runtimes accepted by Model.from_dir(runtime=...).
_Runtime = Literal["eager", "onnx", "tensorrt"]
# Video-decoding backends accepted by predict_on_video_file(_multiview)'s decoder=
# kwarg. Independent of _Runtime -- decoder controls video ingestion, _Runtime
# controls model execution.
_Decoder = Literal["dali", "pynvvc"]

# Maps onnxruntime's tensor type strings to numpy dtypes. Used to bind input
# buffers at whatever precision the exported file actually expects, rather than
# assuming fp32 -- an fp16 export takes fp16 input.
_ORT_TYPE_TO_NUMPY: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}


# torch.compile's inductor backend emits Triton kernels, which need Volta or newer.
_TORCH_COMPILE_MIN_CUDA_CAPABILITY = (7, 0)


def _check_torch_compile_supported() -> None:
    """Raise if ``torch.compile``'s inductor backend cannot run on the available GPU.

    Without this check the failure surfaces as a ``BackendCompilerFailed`` traceback
    partway through the first prediction, long after ``compile()`` returned. CPU
    inference is unaffected -- inductor has a separate C++ backend that does not
    require Triton.

    Raises:
        RuntimeError: if a CUDA device is available but its compute capability is
            below 7.0.
    """
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    if capability >= _TORCH_COMPILE_MIN_CUDA_CAPABILITY:
        return
    raise RuntimeError(
        f"torch.compile() requires a GPU of CUDA compute capability "
        f"{_TORCH_COMPILE_MIN_CUDA_CAPABILITY[0]}.{_TORCH_COMPILE_MIN_CUDA_CAPABILITY[1]} "
        f"or higher, but {torch.cuda.get_device_name()} has capability "
        f"{capability[0]}.{capability[1]}. Run without compilation (omit --compile, or "
        f"do not call Model.compile()) to use this GPU."
    )


def load_model_from_checkpoint(
    cfg: DictConfig | ListConfig,
    ckpt_file: str | None,
    eval: bool = False,
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
    skip_data_module: bool = False,
) -> ALLOWED_MODELS:
    """Load a Lightning Pose model from a checkpoint file.

    Args:
        cfg: model config
        ckpt_file: absolute path to model checkpoint
        eval: True for eval mode, False for train mode
        data_module: used to initialise unsupervised losses
        skip_data_module: if ``data_module`` is not None this is ignored.
            If False and ``data_module=None``, a data module is created from the config file and
            unsupervised losses are accessible in the model.
            If True and ``data_module=None``, the unsupervised losses are not accessible in the
            model; recommended for running inference on new videos.

    Returns:
        model as a Lightning Module

    Raises:
        ValueError: if ``ckpt_file`` is None

    """
    if ckpt_file is None:
        raise ValueError('ckpt_file must be provided to load a model from checkpoint')
    from lightning_pose.data import (
        get_data_module,
        get_dataset,
        get_imgaug_transform,
    )
    from lightning_pose.losses import get_loss_factories
    from lightning_pose.models import check_if_semi_supervised
    from lightning_pose.utils.io import return_absolute_data_paths

    delete_extras = False
    if not data_module and not skip_data_module:
        delete_extras = True
        data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
        imgaug_transform = get_imgaug_transform(cfg=cfg)
        dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
        data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
    if not data_module:
        loss_factories = {'supervised': None, 'unsupervised': None}
    else:
        loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    ModelClass = get_model_class(
        model_type=cfg.model.model_type,
        semi_supervised=semi_supervised,
    )

    try:
        checkpoint = torch.load(ckpt_file)
    except Exception as e:
        logger.warning(f'failed to load checkpoint with default settings: {e}')
        logger.warning('attempting to load with weights_only=False...')
        checkpoint = torch.load(ckpt_file, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # fix state dict key mismatch for upsampling layers in old checkpoints
    keys_remapped = False
    for key in list(state_dict.keys()):
        if key.startswith('upsampling_layers.'):
            state_dict['head.' + key] = state_dict.pop(key)
            keys_remapped = True

    if keys_remapped:
        checkpoint['state_dict'] = state_dict
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            fixed_ckpt_file = tmp_file.name
    else:
        fixed_ckpt_file = ckpt_file

    if semi_supervised:
        model = ModelClass.load_from_checkpoint(
            fixed_ckpt_file,
            loss_factory=loss_factories['supervised'],
            loss_factory_unsupervised=loss_factories['unsupervised'],
            strict=False,
        )
    else:
        model = ModelClass.load_from_checkpoint(
            fixed_ckpt_file,
            loss_factory=loss_factories['supervised'],
            strict=False,
        )

    if keys_remapped:
        import os
        os.unlink(fixed_ckpt_file)

    if eval:
        model.eval()

    if delete_extras:
        del imgaug_transform
        del dataset
        del data_module
    del loss_factories
    torch.cuda.empty_cache()

    return model


def _tensorrt_version() -> str | None:
    """Return the installed ``tensorrt`` package version, or None if absent."""
    try:
        import tensorrt  # type: ignore[import-not-found]

        return tensorrt.__version__
    except ImportError:
        return None


def _onnxruntime_version() -> str | None:
    """Return the installed ``onnxruntime`` package version, or None if absent."""
    try:
        import onnxruntime  # type: ignore[import-not-found]

        return onnxruntime.__version__
    except ImportError:
        return None


def _build_onnx_forward(session: Any) -> Any:
    """Build a forward-replacement closure that runs inference through an ONNX
    Runtime session, regardless of which execution provider backs it.

    Shared by ``Model._attach_onnx_runtime()`` and
    ``Model._attach_tensorrt_runtime()`` -- the session-driven forward pass
    (io_binding setup, dtype detection, CPU/GPU branching) is identical for
    both; only the ``providers`` passed to ``InferenceSession`` differs. The
    returned callable is assigned directly to ``self.model.forward`` (rather
    than wrapping the module) so every existing prediction path keeps working,
    including those that call ``get_loss_inputs_labeled`` directly instead of
    going through ``__call__``. Because ``Model.model``'s type is a union of
    several model classes with genuinely different ``forward`` signatures,
    callers assign this with a ``# type: ignore[method-assign]``.
    """
    providers = session.get_providers()
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    output_name = session.get_outputs()[0].name
    input_np_dtype = _ORT_TYPE_TO_NUMPY[input_meta.type]
    input_torch_dtype = torch.from_numpy(np.empty(0, dtype=input_np_dtype)).dtype
    cuda_available = (
        "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers
    )

    def onnx_forward(images: torch.Tensor) -> torch.Tensor:
        images = images.to(input_torch_dtype).contiguous()
        if cuda_available and images.is_cuda:
            io_binding = session.io_binding()
            device_id = images.device.index or 0
            io_binding.bind_input(
                name=input_name,
                device_type="cuda",
                device_id=device_id,
                element_type=input_np_dtype,
                shape=tuple(images.shape),
                buffer_ptr=images.data_ptr(),
            )
            io_binding.bind_output(
                name=output_name, device_type="cuda", device_id=device_id
            )
            session.run_with_iobinding(io_binding)
            output = io_binding.copy_outputs_to_cpu()[0]
        else:
            output = session.run([output_name], {input_name: images.cpu().numpy()})[0]
        return torch.from_numpy(output).to(images.device)

    return onnx_forward


class Model:
    """High-level interface for inference with a trained lightning-pose model.

    Load a saved model with `Model.from_dir`, then call prediction methods directly.
    Model weights are loaded lazily on the first prediction call.

    Attributes:
        model_dir: absolute path to the directory the model is stored in.
        config: the model configuration as a `ModelConfig` object.
        model: the underlying PyTorch model; None until the first prediction call.

    Examples:
        >>> from lightning_pose.api import Model
        >>> model = Model.from_dir("outputs/2024-01-01/12-00-00")

        Single-frame inference (no file I/O):
        >>> import numpy as np
        >>> frame = np.zeros((256, 256, 3), dtype=np.uint8)
        >>> result = model.predict_frame(frame)
        >>> result["keypoints"].shape   # (num_keypoints, 2)
        >>> result["confidence"].shape  # (num_keypoints,)

        Predict on a video file:
        >>> pred_result = model.predict_on_video_file("path/to/video.mp4")
        >>> pred_result.predictions     # pd.DataFrame with MultiIndex columns
        >>> pred_result.metrics         # ComputeMetricsSingleResult or None

        Predict on a labeled CSV (also computes pixel error):
        >>> pred_result = model.predict_on_label_csv("path/to/CollectedData.csv")
    """

    model_dir: Path
    """Directory the model is stored in."""

    config: ModelConfig
    """The model configuration stored as a `ModelConfig` object.
    `ModelConfig` wraps the `omegaconf.DictConfig` and provides util functions
    over it.
    """

    model: ALLOWED_MODELS | None = None

    precision: _Precision = "fp32"
    """Precision used for inference: ``"fp32"``, ``"fp16"``, or ``"bf16"``
    (same strings as the ``litpose predict --precision`` CLI flag). Does not
    affect the checkpoint on disk."""

    _compiled: bool = False
    """Whether ``compile()`` has been called. Guards against double-wrapping
    ``forward`` on repeat calls."""

    _runtime: _Runtime = "eager"
    """Which inference runtime backs ``forward``: ``"eager"`` (the loaded
    PyTorch checkpoint) or ``"onnx"`` (an ONNX Runtime session). Set by
    ``from_dir(runtime=...)``; not user-assignable after construction."""

    # Just a constant we can use as a default value for kwargs,
    # to differentiate between user omitting a kwarg, vs explicitly passing None.
    UNSPECIFIED = "unspecified"

    @staticmethod
    def from_dir(
        model_dir: str | Path,
        precision: _Precision = "fp32",
        runtime: _Runtime = "eager",
        onnx_precision: _OnnxPrecision | None = None,
    ) -> Model:
        """Create a `Model` instance for a model stored at `model_dir`.

        Args:
            model_dir: path to a model output directory containing ``config.yaml``
                and a ``.ckpt`` checkpoint file.
            precision: precision to run inference at. One of ``"fp32"``
                (default), ``"fp16"``, or ``"bf16"`` -- same strings as the
                ``litpose predict --precision`` CLI flag. Does not affect
                the checkpoint itself -- weights stay fp32 on disk; this only
                controls the precision used during the forward pass.
            runtime: inference backend. ``"eager"`` (default) loads the trained
                checkpoint as usual. ``"onnx"`` loads an ONNX Runtime session
                from ``exports_onnx_dir()``, which must have been built with
                ``model.export("onnx", ...)`` beforehand. ``"tensorrt"`` loads
                a TensorRT engine from ``exports_trt_dir()``, built with
                ``model.export("tensorrt", ...)`` beforehand (which itself
                requires an existing ``"onnx"`` export for the same
                ``onnx_precision``). ``precision`` is ignored in both
                non-eager modes, since the exported file's own precision is
                what runs.
            onnx_precision: only used when ``runtime="onnx"`` or
                ``runtime="tensorrt"``. Selects which exported file (or, for
                tensorrt, which engine cache) to load. If omitted and exactly
                one export exists for this checkpoint, it is used
                automatically; if more than one exists, raises.

        Returns:
            Model ready for inference. Weights are loaded lazily on the first
            prediction call.

        Examples:
            >>> from lightning_pose.api import Model
            >>> model = Model.from_dir("outputs/2024-01-01/12-00-00")
            >>> model.config.is_multi_view()
            False

            Run inference in FP16:
            >>> model = Model.from_dir("outputs/2024-01-01/12-00-00", precision="fp16")
        """
        return Model.from_dir2(
            model_dir,
            precision=precision,
            runtime=runtime,
            onnx_precision=onnx_precision,
        )

    @staticmethod
    def from_dir2(
        model_dir: str | Path,
        hydra_overrides: list[str] | None = None,
        precision: _Precision = "fp32",
        runtime: _Runtime = "eager",
        onnx_precision: _OnnxPrecision | None = None,
    ) -> Model:
        """Internal version of from_dir that supports hydra_overrides. Not sure whether to
        promote this to public API yet."""

        model_dir = Path(model_dir).absolute()

        if hydra_overrides is not None:
            import hydra

            with hydra.initialize_config_dir(
                version_base="1.1", config_dir=str(model_dir)
            ):
                cfg = hydra.compose(config_name="config", overrides=hydra_overrides)
                config = ModelConfig(cfg)
        else:
            config = ModelConfig.from_yaml_file(model_dir / "config.yaml")

        model = Model(model_dir, config, precision=precision)

        if runtime == "eager":
            return model
        elif runtime == "onnx":
            model._attach_onnx_runtime(onnx_precision)
            return model
        elif runtime == "tensorrt":
            model._attach_tensorrt_runtime(onnx_precision)
            return model
        else:
            supported = ", ".join(repr(r) for r in get_args(_Runtime))
            raise ValueError(
                f"Unsupported runtime: '{runtime}'. Use one of {supported}."
            )

    def __init__(
        self, model_dir: str | Path, config: ModelConfig, precision: _Precision = "fp32"
    ) -> None:
        """Initialize a Model from a directory and a pre-loaded config.

        Prefer `Model.from_dir` for typical usage. Use this constructor when you
        have already constructed a `ModelConfig` (e.g. after applying Hydra overrides).

        Args:
            model_dir: path to the model output directory.
            config: the model configuration.
            precision: precision to run inference at. One of ``"fp32"``
                (default), ``"fp16"``, or ``"bf16"``.
        """
        self.model_dir = Path(model_dir).absolute()
        self.config = config
        self.precision = precision

    @property
    def cfg(self) -> DictConfig | ListConfig:
        """The model configuration as an `omegaconf.DictConfig`."""
        return self.config.cfg

    @property
    def pl_precision(self) -> _PLPrecision:
        """PyTorch Lightning ``Trainer`` precision string for ``self.precision``.

        Internal plumbing for the two ``pl.Trainer`` construction sites in
        ``lightning_pose.utils.predictions``. User-facing code should read/set
        ``self.precision`` (``"fp32"``/``"fp16"``/``"bf16"``) instead.
        """
        return _PRECISION_TO_PL[self.precision]

    def compile(self) -> None:
        """Compile the model's forward pass with ``torch.compile()`` for faster inference.

        Loads the checkpoint if it hasn't been loaded yet, so this can be called
        immediately after ``Model.from_dir``. Compilation itself is lazy: it is
        triggered on the first inference call and can take tens of seconds.
        Subsequent calls with the same input shape reuse the compiled graph;
        changing batch size or input resolution triggers a new compilation
        automatically.

        Calling this more than once is a no-op.

        Raises:
            RuntimeError: if a CUDA device is available but too old for the inductor
                backend (compute capability below 7.0).

        Examples:
            >>> model = Model.from_dir("outputs/2024-01-01/12-00-00")
            >>> model.compile()
            >>> result = model.predict_on_video_file("path/to/video.mp4")
        """
        if self._runtime != "eager":
            raise RuntimeError(
                f"compile() is only supported for runtime='eager', but this model was "
                f"loaded with runtime='{self._runtime}'. torch.compile() has no effect "
                f"on an ONNX Runtime session."
            )
        if self._compiled:
            return
        _check_torch_compile_supported()
        self._load()
        if self.model is None:
            raise RuntimeError('model failed to load; self.model is None after _load()')
        # `self.model` is a union of model classes whose `forward` signatures differ, so
        # the compiled callable's inferred union type isn't assignable to any single
        # class's `forward`. The rebind itself is intentional and behaviorally safe.
        self.model.forward = torch.compile(self.model.forward)  # type: ignore[method-assign]
        self._compiled = True

    def _load(self) -> None:
        """Load model weights from the checkpoint file on first call; no-op thereafter.

        Raises:
            FileNotFoundError: if no checkpoint file is found in `model_dir`.
        """
        if self.model is None:
            ckpt_file = io_utils.ckpt_path_from_base_path(
                base_path=str(self.model_dir), model_name=self.cfg.model.model_name
            )
            if ckpt_file is None:
                raise FileNotFoundError(
                    "Checkpoint file not found, have you trained for enough epochs?"
                )
            self.model = load_model_from_checkpoint(
                cfg=self.cfg,
                ckpt_file=ckpt_file,
                eval=True,
                skip_data_module=True,
            )

    def _find_onnx_export(self, onnx_precision: _OnnxPrecision | None) -> Path:
        """Locate the ONNX export for this checkpoint, disambiguating by precision.

        Shared by ``_attach_onnx_runtime()`` and ``_export_tensorrt()``, which
        builds its engine from an existing ONNX export rather than re-tracing
        the model.

        Raises:
            FileNotFoundError: if no matching export exists.
            ValueError: if multiple exports exist and ``onnx_precision`` is None.
        """
        stem = self._ckpt_stem()
        export_dir = self.exports_onnx_dir()
        pattern = (
            f"{stem}_{onnx_precision}.onnx"
            if onnx_precision is not None
            else f"{stem}_*.onnx"
        )
        candidates = sorted(export_dir.glob(pattern))
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No ONNX export found for checkpoint '{stem}'"
                + (f" with onnx_precision='{onnx_precision}'" if onnx_precision else "")
                + f" in {export_dir}. Run model.export('onnx') first."
            )
        if len(candidates) > 1:
            found = [c.name for c in candidates]
            raise ValueError(
                f"Multiple ONNX exports found: {found}. "
                "Pass onnx_precision= to disambiguate."
            )
        return candidates[0]

    def _attach_onnx_runtime(self, onnx_precision: _OnnxPrecision | None) -> None:
        """Replace the model's forward pass with an ONNX Runtime session.

        Locates the export for this model's checkpoint inside
        ``exports_onnx_dir()`` and rebinds ``self.model.forward`` to run through
        it. Rebinding ``forward`` rather than wrapping the module keeps every
        existing prediction path working, including those that call
        ``get_loss_inputs_labeled`` directly instead of going through
        ``__call__``.

        Args:
            onnx_precision: which export to load, or None to auto-select when
                exactly one export exists for this checkpoint.

        Raises:
            FileNotFoundError: if no matching export exists.
            ValueError: if multiple exports exist and ``onnx_precision`` is None.
            RuntimeError: if the model fails to load, or if ONNX Runtime silently
                fell back to CPU on a CUDA machine.
        """
        # Export discovery happens before importing onnxruntime or loading
        # weights: a user who forgot to run export() should get that message
        # regardless of whether the optional dependency is installed, and
        # there's no reason to pay for _load() on a path that can't succeed.
        onnx_path = self._find_onnx_export(onnx_precision)

        # Imported lazily: onnxruntime is an optional dependency, and a
        # module-level import would make it mandatory for every eager user.
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for runtime='onnx' but is not installed. "
                "See https://lightning-pose.readthedocs.io/en/latest/source/"
                "user_guide_advanced/increasing_inference_speed.html#onnx-installation "
                "for installation instructions."
            ) from e

        self._load()
        if self.model is None:
            raise RuntimeError('model failed to load; self.model is None after _load()')

        session = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Guard against a silent fallback to CPU. onnxruntime does not raise if
        # CUDAExecutionProvider fails to load (e.g. onnxruntime-gpu's CUDA/cuDNN
        # major version doesn't match the installed PyTorch/CUDA) -- it just runs
        # on CPU instead, which "works" but is drastically slower with no
        # indication why. Fail loudly here instead.
        providers = session.get_providers()
        if torch.cuda.is_available() and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "ONNX Runtime failed to load CUDAExecutionProvider even though a CUDA "
                "device is available -- inference would silently run on CPU instead, "
                "which works but is drastically slower with no indication why. This is "
                "usually a CUDA major version mismatch between onnxruntime-gpu and your "
                "installed PyTorch: onnxruntime-gpu wheels are built against a specific "
                "CUDA version and do not adapt to the one you have. Check yours with "
                "`python -c \"import torch; print(torch.version.cuda)\"` and reinstall "
                "the matching build -- see "
                "https://lightning-pose.readthedocs.io/en/latest/source/"
                "user_guide_advanced/increasing_inference_speed.html#onnx-installation"
            )

        self.model.forward = _build_onnx_forward(session)  # type: ignore[method-assign]
        self._runtime = "onnx"
        logger.info(f'loaded ONNX runtime session from {onnx_path}')

    def _find_trt_cache(self, onnx_precision: _OnnxPrecision | None) -> Path:
        """Locate the TensorRT engine cache directory for this checkpoint.

        Analogous to ``_find_onnx_export()``.

        Raises:
            FileNotFoundError: if no matching cache exists.
            ValueError: if multiple caches exist and ``onnx_precision`` is None.
        """
        stem = self._ckpt_stem()
        trt_dir = self.exports_trt_dir()
        pattern = f"{stem}_{onnx_precision}" if onnx_precision is not None else f"{stem}_*"
        candidates = sorted(p for p in trt_dir.glob(pattern) if p.is_dir())
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No TensorRT export found for checkpoint '{stem}'"
                + (f" with onnx_precision='{onnx_precision}'" if onnx_precision else "")
                + f" in {trt_dir}. Run model.export('tensorrt', ...) first."
            )
        if len(candidates) > 1:
            found = [c.name for c in candidates]
            raise ValueError(
                f"Multiple TensorRT exports found: {found}. "
                "Pass onnx_precision= to disambiguate."
            )
        return candidates[0]

    def _attach_tensorrt_runtime(self, onnx_precision: _OnnxPrecision | None) -> None:
        """Replace the model's forward pass with a TensorRT-backed ONNX Runtime session.

        Stricter than ``_attach_onnx_runtime()``: TensorRT has no CPU execution
        path, so this raises immediately if no CUDA device is available. It
        also raises (rather than warns, unlike ONNX's CPU fallback) if the
        session falls back to CUDAExecutionProvider instead of engaging
        TensorRT -- landing on plain CUDA here silently defeats the entire
        reason the caller asked for ``runtime="tensorrt"``.

        Raises:
            FileNotFoundError: if no matching engine cache exists.
            ValueError: if multiple caches exist and ``onnx_precision`` is None.
            RuntimeError: if no CUDA device is available, if the session fails
                to load TensorrtExecutionProvider, or if the model fails to load.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "runtime='tensorrt' requires a CUDA-capable GPU; none is "
                "available. TensorRT has no CPU execution path -- use "
                "runtime='onnx' or runtime='eager' instead."
            )

        cache_dir = self._find_trt_cache(onnx_precision)
        metadata_path = cache_dir / "trt_metadata.json"
        resolved_onnx_precision = onnx_precision
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            current_gpu = torch.cuda.get_device_name(0)
            recorded_gpu = metadata.get("gpu_name")
            if recorded_gpu is not None and recorded_gpu != current_gpu:
                warnings.warn(
                    f"TensorRT engine at {cache_dir} was built on "
                    f"'{recorded_gpu}' but is running on '{current_gpu}'. "
                    "onnxruntime will rebuild the engine on first inference, "
                    "which can take several minutes.",
                    stacklevel=2,
                )
            if resolved_onnx_precision is None:
                resolved_onnx_precision = metadata.get("onnx_precision")

        onnx_path = self._find_onnx_export(resolved_onnx_precision)

        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for runtime='tensorrt' but is not "
                "installed. See https://lightning-pose.readthedocs.io/en/latest/"
                "source/user_guide_advanced/increasing_inference_speed.html"
                "#onnx-installation for installation instructions."
            ) from e

        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_fp16_enable": resolved_onnx_precision == "fp16",
        }
        session = ort.InferenceSession(
            str(onnx_path),
            providers=[("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider"],
        )
        if "TensorrtExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "ONNX Runtime failed to load TensorrtExecutionProvider -- "
                "inference would silently run on CUDAExecutionProvider instead, "
                "without the TensorRT speedup you asked for. This is usually a "
                "missing/mismatched TensorRT install or a missing "
                "LD_LIBRARY_PATH entry. See "
                "https://lightning-pose.readthedocs.io/en/latest/source/"
                "user_guide_advanced/increasing_inference_speed.html"
                "#tensorrt-installation"
            )

        self._load()
        if self.model is None:
            raise RuntimeError('model failed to load; self.model is None after _load()')
        self.model.forward = _build_onnx_forward(session)  # type: ignore[method-assign]
        self._runtime = "tensorrt"
        logger.info(f'loaded TensorRT engine session from {cache_dir}')

    def _ckpt_stem(self) -> str:
        """Return the filename stem of this model's checkpoint.

        Used to name and locate ONNX exports. Shared by ``export()`` and
        ``_attach_onnx_runtime()`` so the two always agree on which checkpoint
        an export corresponds to.

        Raises:
            FileNotFoundError: if no checkpoint file is found in ``model_dir``.
        """
        ckpt_file = io_utils.ckpt_path_from_base_path(
            base_path=str(self.model_dir), model_name=self.cfg.model.model_name
        )
        if ckpt_file is None:
            raise FileNotFoundError(
                "Checkpoint file not found, have you trained for enough epochs?"
            )
        return Path(ckpt_file).stem

    def export(
        self,
        runtime: str,
        onnx_precision: _OnnxPrecision = "fp16",
        max_batch_size: int = 8,
        opt_batch_size: int | None = None,
    ) -> Path:
        """Export the model to an optimized inference format.

        Supports ``runtime="onnx"`` and ``runtime="tensorrt"``. Writes to a fixed location inside
        the model directory (``exports_onnx_dir() / "{ckpt_stem}_{onnx_precision}.onnx"``)
        and returns the path to the exported file. Export paths are not
        user-configurable by design -- ``from_dir(runtime="onnx")`` reads from
        this same location.

        Args:
            runtime: export target, ``"onnx"`` or ``"tensorrt"``. TensorRT
                builds its engine from an existing ONNX export rather than
                re-tracing the model -- run ``export("onnx", ...)`` first for
                the same ``onnx_precision``.
            max_batch_size: only used for ``runtime="tensorrt"``. Upper bound
                of the dynamic-shape batch profile the engine is built for;
                inference at a batch size above this fails. Default 8.
            opt_batch_size: only used for ``runtime="tensorrt"``. Batch size
                the engine is optimized for; defaults to ``max_batch_size``.
                Correctness holds for any batch size in
                ``[1, max_batch_size]``, but performance is best near this
                value.
            onnx_precision: weight precision baked into the exported file, either
                ``"fp32"`` or ``"fp16"``. Distinct from ``self.precision``, which
                controls autocast precision for eager inference only.

        Returns:
            Path to the exported ``.onnx`` file.

        Raises:
            ValueError: if ``runtime`` or ``onnx_precision`` is unsupported.
            FileNotFoundError: if no checkpoint file is found in
                ``model_dir`` (``runtime="onnx"``), or if the required ONNX
                export doesn't exist yet (``runtime="tensorrt"``).
            RuntimeError: if the model fails to load.

        Examples:
            >>> model = Model.from_dir("outputs/2024-01-01/12-00-00")
            >>> model.export("onnx", onnx_precision="fp16")
            >>> model.export("tensorrt", onnx_precision="fp16", max_batch_size=16)
        """
        if runtime not in ("onnx", "tensorrt"):
            raise ValueError(
                f"Unsupported export runtime: '{runtime}'. Use 'onnx' or 'tensorrt'."
            )
        if onnx_precision not in ("fp32", "fp16"):
            raise ValueError(
                f"Unsupported onnx_precision: '{onnx_precision}'. Use 'fp32' or 'fp16'."
            )
        if runtime == "tensorrt":
            return self._export_tensorrt(onnx_precision, max_batch_size, opt_batch_size)
        # Weights are lazy-loaded; tracing needs a real forward pass.
        self._load()
        if self.model is None:
            raise RuntimeError('model failed to load; self.model is None after _load()')

        onnx_path = self.exports_onnx_dir() / f"{self._ckpt_stem()}_{onnx_precision}.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        resize_h = self.cfg.data.image_resize_dims.height
        resize_w = self.cfg.data.image_resize_dims.width

        # Dummy input shape depends on model type. Context models take a temporal
        # stack and multiview models take a per-view stack; see predict_frame and
        # predict_on_video_file_multiview for the corresponding runtime shapes.
        # The two cases are mutually exclusive rather than composable:
        # HeatmapTrackerMultiviewTransformer raises on a do_context kwarg and
        # hardcodes do_context=False, so a true-multiview model never reports
        # do_context=True.
        if self.config.is_multi_view():
            num_views = len(self.cfg.data.view_names)
            shape = (1, num_views, 3, resize_h, resize_w)
        elif self.model.do_context:
            shape = (1, _CONTEXT_SEQUENCE_LENGTH, 3, resize_h, resize_w)
        else:
            shape = (1, 3, resize_h, resize_w)

        # .half() mutates in place and returns self, so exporting fp16 from a
        # live Model would silently leave self.model in half precision for any
        # subsequent eager call. Trace a copy instead.
        module_to_export = self.model
        dummy = torch.randn(*shape, device=self.model.device)
        if onnx_precision == "fp16":
            module_to_export = copy.deepcopy(self.model).half()
            dummy = dummy.half()

        module_to_export.eval()
        torch.onnx.export(
            module_to_export,
            (dummy,),
            str(onnx_path),
            input_names=["images"],
            output_names=["heatmaps"],
            dynamic_axes={"images": {0: "batch"}, "heatmaps": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

        if onnx_precision == "fp16":
            del module_to_export
            torch.cuda.empty_cache()

        logger.info(f'exported {onnx_precision} ONNX model to {onnx_path}')
        return onnx_path

    def _export_tensorrt(
        self,
        onnx_precision: _OnnxPrecision,
        max_batch_size: int,
        opt_batch_size: int | None,
    ) -> Path:
        """Build a TensorRT engine cache from an existing ONNX export.

        Unlike ONNX export, this does not call ``self._load()`` or touch the
        trained torch weights at all -- verified empirically (T4, 2026-08-06)
        that onnxruntime's TensorrtExecutionProvider builds the engine purely
        from the ``.onnx`` file, which already has the weights baked in. The
        input shape needed for the dynamic-shape batch profile is read
        directly off the ONNX graph's own input metadata, rather than
        re-deriving it from ``self.cfg``/``self.model.do_context`` the way the
        ONNX export branch above does -- this avoids reimplementing the
        singleview/multiview/context shape branching a second time.

        Raises:
            FileNotFoundError: if the matching ONNX export doesn't exist yet.
            RuntimeError: if the build doesn't engage TensorrtExecutionProvider.
        """
        try:
            onnx_path = self._find_onnx_export(onnx_precision)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No ONNX export found for onnx_precision='{onnx_precision}'. "
                f"Run model.export('onnx', onnx_precision='{onnx_precision}') "
                "first."
            ) from e

        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required to build a TensorRT engine but is not "
                "installed. See https://lightning-pose.readthedocs.io/en/latest/"
                "source/user_guide_advanced/increasing_inference_speed.html"
                "#onnx-installation for installation instructions."
            ) from e

        opt_batch_size = opt_batch_size or max_batch_size
        stem = self._ckpt_stem()
        cache_dir = self.exports_trt_dir() / f"{stem}_{onnx_precision}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Read the input shape straight off the ONNX graph rather than
        # recomputing it from self.cfg -- see docstring above.
        probe_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_meta = probe_session.get_inputs()[0]
        input_name = input_meta.name
        shape_suffix = "x".join(str(d) for d in input_meta.shape[1:])
        del probe_session

        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_fp16_enable": onnx_precision == "fp16",
            "trt_profile_min_shapes": f"{input_name}:1x{shape_suffix}",
            "trt_profile_opt_shapes": f"{input_name}:{opt_batch_size}x{shape_suffix}",
            "trt_profile_max_shapes": f"{input_name}:{max_batch_size}x{shape_suffix}",
        }
        session = ort.InferenceSession(
            str(onnx_path),
            providers=[("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider"],
        )
        if "TensorrtExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "TensorRT engine build failed to use TensorrtExecutionProvider. "
                "This usually means a missing/mismatched TensorRT install. See "
                "https://lightning-pose.readthedocs.io/en/latest/source/"
                "user_guide_advanced/increasing_inference_speed.html"
                "#tensorrt-installation"
            )
        del session

        self._write_trt_metadata(cache_dir, onnx_precision, max_batch_size, opt_batch_size)
        logger.info(f'built TensorRT engine cache at {cache_dir}')
        return cache_dir

    def _write_trt_metadata(
        self,
        cache_dir: Path,
        onnx_precision: _OnnxPrecision,
        max_batch_size: int,
        opt_batch_size: int,
    ) -> None:
        """Write a sidecar recording the exact environment a TensorRT engine was built in.

        Unlike the ``.onnx`` file, a TensorRT engine cache is tied to the exact
        GPU architecture, TensorRT version, and batch profile it was built
        with -- it is not portable across machines. ``_attach_tensorrt_runtime()``
        reads this back and warns on a GPU-name mismatch, so a mismatch
        produces a clear message instead of a confusing onnxruntime failure or
        an unexpected silent rebuild.
        """
        import datetime

        metadata = {
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "tensorrt_version": _tensorrt_version(),
            "onnxruntime_version": _onnxruntime_version(),
            "onnx_precision": onnx_precision,
            "max_batch_size": max_batch_size,
            "opt_batch_size": opt_batch_size,
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (cache_dir / "trt_metadata.json").write_text(json.dumps(metadata, indent=2))

    def image_preds_dir(self) -> Path:
        """Return the directory where image/CSV predictions are saved."""
        return self.model_dir / "image_preds"

    def video_preds_dir(self) -> Path:
        """Return the directory where video predictions are saved."""
        return self.model_dir / "video_preds"

    def labeled_videos_dir(self) -> Path:
        """Return the directory where prediction-annotated videos are saved."""
        return self.model_dir / "video_preds" / "labeled_videos"

    def cropped_data_dir(self) -> Path:
        """Return the directory where cropzoom-cropped images are saved."""
        return self.model_dir / "cropped_images"

    def cropped_videos_dir(self) -> Path:
        """Return the directory where cropzoom-cropped videos are saved."""
        return self.model_dir / "cropped_videos"

    def exports_onnx_dir(self) -> Path:
        """Return the directory where ONNX exports are saved."""
        return self.model_dir / "exports_onnx"

    def exports_trt_dir(self) -> Path:
        """Return the directory where TensorRT engine caches are saved."""
        return self.model_dir / "exports_trt"

    def cropped_csv_file_path(self, csv_file_path: str | Path) -> Path:
        """Return the path where a cropzoom-adjusted CSV file will be saved.

        Args:
            csv_file_path: path to the original labeled CSV file.

        Returns:
            path of the form ``{model_dir}/image_preds/{csv_name}/cropped_{csv_name}``.
        """
        csv_file_path = Path(csv_file_path)
        return (
            self.model_dir
            / "image_preds"
            / csv_file_path.name
            / ("cropped_" + csv_file_path.name)
        )

    def predict_frame(
        self,
        frame_rgb: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Single-frame inference. No file I/O, no DALI.

        Preprocessing uses cv2 (not DALI). Results will differ numerically
        from ``predict_on_video_file`` due to interpolation and normalization
        differences. Do not mix results from the two paths in quantitative
        analysis.

        For MHCRNN (context) models, pass a ``(T, H, W, 3)`` array where T
        is the temporal context length (typically 5). Passing a single frame
        to a context model raises ``ValueError`` — use
        ``predict_on_video_file`` for proper temporal inference.

        The first call triggers model loading and CUDA initialization, which
        may take several seconds. Subsequent calls are fast (~5-50ms depending
        on backbone). For latency-sensitive loops, call once on a dummy frame
        before entering the loop.

        Args:
            frame_rgb: ``(H, W, 3)`` uint8 RGB array for standard models, or
                ``(T, H, W, 3)`` uint8 RGB array for context (MHCRNN) models.
            bbox: Optional ``(x, y, w, h)`` crop region. Note: this is
                ``(x, y, width, height)``, NOT ``(x1, y1, x2, y2)``.
                If provided, crops first, then remaps keypoints back to
                original coordinates.

        Returns:
            {"keypoints": (num_kp, 2) float32 array (x, y) in original frame coords,
             "confidence": (num_kp,) float32 in [0, 1] -- likelihood/confidence
              per keypoint. For regression models, confidence is always 1.0.}

        Raises:
            ValueError: If frame_rgb has wrong shape/dtype, bbox has non-positive
                dimensions, bbox produces an empty crop, or a context model
                receives single-frame input.

        Examples:
            >>> import numpy as np
            >>> frame = np.zeros((256, 256, 3), dtype=np.uint8)
            >>> result = model.predict_frame(frame)
            >>> result["keypoints"].shape    # (num_keypoints, 2)
            >>> result["confidence"].shape   # (num_keypoints,)

            With a bounding-box crop (x, y, width, height):
            >>> result = model.predict_frame(frame, bbox=(100, 50, 128, 128))

        """
        self._load()
        if self.model is None:
            raise RuntimeError('model failed to load; self.model is None after _load()')

        # --- Input validation ---
        if frame_rgb.dtype != np.uint8:
            raise ValueError(
                f"frame_rgb must be uint8, got {frame_rgb.dtype}. "
                "Convert with frame.astype(np.uint8) if values are in [0, 255]."
            )

        is_context_input = frame_rgb.ndim == 4
        if is_context_input:
            if frame_rgb.shape[3] != 3:
                raise ValueError(
                    f"frame_rgb must be (T, H, W, 3), got shape {frame_rgb.shape}"
                )
        elif frame_rgb.ndim == 3:
            if frame_rgb.shape[2] != 3:
                raise ValueError(
                    f"frame_rgb must be (H, W, 3), got shape {frame_rgb.shape}"
                )
        else:
            raise ValueError(
                f"frame_rgb must be (H, W, 3) or (T, H, W, 3), "
                f"got {frame_rgb.ndim}D array with shape {frame_rgb.shape}"
            )

        if frame_rgb.size == 0:
            raise ValueError("frame_rgb is empty")

        is_context_model = self.model.do_context
        if is_context_model and not is_context_input:
            raise ValueError(
                "Context model requires frame_rgb of shape (T, H, W, 3) "
                "where T is the temporal context length (typically 5). "
                "Use predict_on_video_file for single-frame input."
            )

        # --- Crop ---
        if bbox is not None:
            bx, by, bw, bh = bbox
            if bx < 0 or by < 0:
                raise ValueError(
                    f"bbox origin must be non-negative, got x={bx}, y={by}"
                )
            if bw <= 0 or bh <= 0:
                raise ValueError(
                    f"bbox width and height must be positive, got w={bw}, h={bh}"
                )
            if is_context_input:
                crop = frame_rgb[:, by:by + bh, bx:bx + bw]
            else:
                crop = frame_rgb[by:by + bh, bx:bx + bw]
            if crop.size == 0:
                raise ValueError(
                    f"bbox (x={bx}, y={by}, w={bw}, h={bh}) produces an empty "
                    f"crop on frame of shape {frame_rgb.shape}"
                )
            # Use actual crop dims for remap -- numpy clips silently when
            # bbox extends beyond frame boundaries.
            if is_context_input:
                actual_h, actual_w = crop.shape[1], crop.shape[2]
            else:
                actual_h, actual_w = crop.shape[0], crop.shape[1]
        else:
            crop = frame_rgb

        # --- Preprocess ---
        resize_h = self.cfg.data.image_resize_dims.height
        resize_w = self.cfg.data.image_resize_dims.width
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)

        def _preprocess_single(img: np.ndarray) -> np.ndarray:
            """Resize, normalize, and transpose a single HWC uint8 frame to CHW float32."""
            resized = cv2.resize(
                img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR,
            )
            t = resized.astype(np.float32) / 255.0
            t = (t - mean) / std
            return np.transpose(t, (2, 0, 1))  # (3, H, W)

        if is_context_input:
            frames = [_preprocess_single(crop[i]) for i in range(crop.shape[0])]
            tensor = np.stack(frames)  # (T, 3, H, W)
            tensor_t = torch.from_numpy(tensor).unsqueeze(0)  # (1, T, 3, H, W)
        else:
            tensor = _preprocess_single(crop)
            tensor_t = torch.from_numpy(tensor).unsqueeze(0)  # (1, 3, H, W)

        device = self.model.device
        tensor_t = tensor_t.to(device)

        # --- Build batch dict ---
        # Bbox in LP format: [x, y, height, width]
        if bbox is not None:
            bbox_lp = torch.tensor(
                [[bx, by, actual_h, actual_w]], dtype=torch.float32, device=device,
            )
        else:
            if is_context_input:
                fh, fw = frame_rgb.shape[1], frame_rgb.shape[2]
            else:
                fh, fw = frame_rgb.shape[0], frame_rgb.shape[1]
            bbox_lp = torch.tensor(
                [[0, 0, fh, fw]], dtype=torch.float32, device=device,
            )

        num_kp = self.model.num_keypoints
        batch_dict = {
            "images": tensor_t,
            "keypoints": torch.zeros(1, num_kp * 2, dtype=torch.float32, device=device),
            "bbox": bbox_lp,
            "idxs": torch.zeros(1, dtype=torch.long, device=device),
            "heatmaps": torch.zeros(1, num_kp, 1, 1, dtype=torch.float32, device=device),
        }

        # --- Inference via get_loss_inputs_labeled ---
        self.model.eval()
        autocast_dtype = _PRECISION_TO_AUTOCAST_DTYPE.get(self.precision)
        with torch.inference_mode():
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    result = self.model.get_loss_inputs_labeled(batch_dict)  # type: ignore[arg-type]
            else:
                result = self.model.get_loss_inputs_labeled(batch_dict)  # type: ignore[arg-type]

        # --- Extract predictions ---
        kp_pred = result["keypoints_pred"]
        has_confidence = "confidences" in result

        if is_context_model:
            # Context model's get_loss_inputs_labeled concatenates [sf; mf] along batch dim
            n = kp_pred.shape[0] // 2
            kp_sf = kp_pred[:n].reshape(n, -1, 2)
            kp_mf = kp_pred[n:].reshape(n, -1, 2)
            conf_sf = result["confidences"][:n]
            conf_mf = result["confidences"][n:]
            # Merge: pick higher-confidence prediction per keypoint
            mf_better = conf_mf > conf_sf
            kp_sf[mf_better] = kp_mf[mf_better]
            conf_merged = conf_sf.clone()
            conf_merged[mf_better] = conf_mf[mf_better]
            kp = kp_sf[0].cpu().numpy().astype(np.float32)
            conf = conf_merged[0].cpu().numpy().astype(np.float32)
        elif has_confidence:
            # Heatmap model — keypoints already in original frame coords
            # (get_loss_inputs_labeled calls model_to_frame_batch internally)
            kp = kp_pred[0].cpu().numpy().reshape(-1, 2).astype(np.float32)
            conf = result["confidences"][0].cpu().numpy().astype(np.float32)
        else:
            # Regression model — get_loss_inputs_labeled does not call
            # model_to_frame_batch, so we apply the remap ourselves.
            kp_pred = model_to_frame_batch(batch_dict, kp_pred, in_place=False)  # type: ignore[arg-type]
            kp = kp_pred[0].cpu().numpy().reshape(-1, 2).astype(np.float32)
            conf = np.ones(num_kp, dtype=np.float32)

        return {"keypoints": kp, "confidence": conf}

    def predict_on_label_csv(
        self,
        csv_file: str | Path,
        data_dir: str | Path | None = None,
        compute_metrics: bool = True,
        add_train_val_test_set: bool = False,
        bbox_file: str | Path | None = None,
    ) -> PredictionResult:
        """Predicts on a labeled dataset and computes error/loss metrics if applicable.

        Args:
            csv_file: path to the CSV file of images and keypoint locations.
            data_dir: root path for relative image paths in the CSV file. Defaults to the
                data_dir used during training.
            compute_metrics: whether to compute pixel error and loss metrics on predictions.
            add_train_val_test_set: set to True when predicting on the training dataset to
                add a ``set`` column to the output.
            bbox_file: optional path to a bbox CSV produced by ``litpose create_bbox`` (or
                any compatible source). When provided, each frame is cropped to its bounding
                box before being passed to the model, and predictions are returned in the
                original (un-cropped) coordinate space.

        Returns:
            PredictionResult: A PredictionResult object containing the predictions and metrics.

        Examples:
            >>> result = model.predict_on_label_csv("path/to/CollectedData.csv")
            >>> result.predictions           # pd.DataFrame with MultiIndex columns
            >>> result.metrics.pixel_error   # mean pixel error per keypoint

            Skip metric computation for faster inference:
            >>> result = model.predict_on_label_csv(
            ...     "path/to/CollectedData.csv",
            ...     compute_metrics=False,
            ... )
        """
        self._load()
        # Convert this to absolute, because if relative, downstream will
        # assume its relative to the data_dir.
        csv_file = Path(csv_file).absolute()
        if data_dir is None:
            data_dir = self.config.cfg.data.data_dir

        output_dir = self.image_preds_dir() / csv_file.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Point predict_dataset to the csv_file and data_dir.
        # HACK: For true multi-view model, trick predict_dataset and compute_metrics
        # into thinking this is a single-view model.
        cfg_overrides: dict[str, Any] = {
            "data": {
                "data_dir": str(data_dir),
                "csv_file": str(csv_file),
                "bbox_file": str(bbox_file) if bbox_file is not None else None,
            }
        }

        # Avoid annotating set=train/val/test for CSV file other than the training CSV file.
        if not add_train_val_test_set:
            cfg_overrides.update({"train_prob": 1, "val_prob": 0, "train_frames": 1})

        # open_dict: cfg_overrides may introduce keys (e.g. data.bbox_file) that are
        # absent from configs saved by older LP versions -- merging those into a
        # struct-mode cfg (e.g. one composed via Model.from_dir2's hydra_overrides)
        # would otherwise raise ConfigAttributeError.
        with open_dict(self.cfg):
            cfg_pred = OmegaConf.merge(self.cfg, cfg_overrides)

        # HACK: For true multi-view model, trick predict_dataset and compute_metrics
        # into thinking this is a single-view model.
        if self.config.is_multi_view():
            del cfg_pred.data.view_names
            # HACK: If we don't delete mirrored_column_matches, downstream
            # interprets this as a mirrored multiview model, and compute_metrics fails.
            del cfg_pred.data.mirrored_column_matches

        data_module_pred = _build_datamodule_pred(cfg_pred)

        preds_file_path = output_dir / "predictions.csv"
        preds_file = str(preds_file_path)

        df = predict_dataset(
            model=self, data_module=data_module_pred, preds_file=preds_file, cfg=cfg_pred,
        )

        if compute_metrics:
            metrics = compute_metrics_single(
                cfg=cfg_pred,
                labels_file=str(csv_file),
                preds_file=preds_file,
                data_module=data_module_pred,
            )
        else:
            metrics = None

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError('expected a single-view DataFrame from predict_dataset')
        return PredictionResult(predictions=df, metrics=metrics)

    def predict_on_label_csv_multiview(
        self,
        csv_file_per_view: list[str] | list[Path],
        bbox_file_per_view: list[str] | list[Path] | None = None,
        camera_params_file: str | Path | None = None,
        data_dir: str | Path | None = None,
        compute_metrics: bool = True,
        add_train_val_test_set: bool = False,
    ) -> MultiviewPredictionResult:
        """Version of ``predict_on_label_csv`` that gives models access to all views of each frame.

        Args:
            csv_file_per_view: a list of csv files each from a different view of the same session;
                order must match ``view_names`` in the config file.

        See ``predict_on_label_csv`` docstring for other arguments.
        """
        if not self.config.is_multi_view():
            raise ValueError('predict_on_label_csv_multiview requires a multi-view model')
        self._load()

        view_names = self.config.cfg.data.view_names
        if len(csv_file_per_view) != len(view_names):
            raise ValueError(
                f'expected {len(view_names)} csv files (one per view), '
                f'got {len(csv_file_per_view)}'
            )

        # Convert this to absolute, because if relative, downstream will
        # assume its relative to the data_dir.
        csv_file_per_view = [Path(f).absolute() for f in csv_file_per_view]

        if data_dir is None:
            data_dir = self.config.cfg.data.data_dir

        # Point predict_dataset to the csv_file and data_dir.
        cfg_overrides: dict[str, Any] = {
            "data": {
                "data_dir": str(data_dir),
                "csv_file": [str(p) for p in csv_file_per_view],
            }
        }
        if camera_params_file:
            cfg_overrides["data"]["camera_params_file"] = camera_params_file
        if bbox_file_per_view:
            cfg_overrides["data"]["bbox_file"] = [str(p) for p in bbox_file_per_view]
        else:
            cfg_overrides["data"]["bbox_file"] = None

        # Avoid annotating set=train/val/test for CSV file other than the training CSV file.
        if not add_train_val_test_set:
            cfg_overrides.update({"train_prob": 1, "val_prob": 0, "train_frames": 1})

        # open_dict: see predict_on_label_csv for why this guards against struct-mode
        # ConfigAttributeError on configs saved by older LP versions.
        with open_dict(self.cfg):
            cfg_pred = OmegaConf.merge(self.cfg, cfg_overrides)

        data_module_pred = _build_datamodule_pred(cfg_pred)

        preds_files = []
        for i, _view_name in enumerate(view_names):
            output_dir = self.image_preds_dir() / csv_file_per_view[i].name
            output_dir.mkdir(parents=True, exist_ok=True)
            preds_files.append(str(output_dir / "predictions.csv"))

        # Outputs dict[str, pd.DataFrame] because inputs indicate multiview.
        view_to_df_dict = predict_dataset(
            model=self, data_module=data_module_pred, preds_file=preds_files, cfg=cfg_pred,
        )

        if compute_metrics:
            metrics = {}
            for view_name, labels_file, _preds_file in zip(
                view_names, csv_file_per_view, preds_files, strict=True
            ):
                metrics[view_name] = compute_metrics_single(
                    cfg=self.cfg,
                    labels_file=str(labels_file),
                    preds_file=_preds_file,
                    data_module=data_module_pred,
                )
        else:
            metrics = None

        return MultiviewPredictionResult(
            predictions=cast(dict[str, pd.DataFrame], view_to_df_dict),
            metrics=metrics,
        )

    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: str | Path | None = UNSPECIFIED,
        compute_metrics: bool = True,
        generate_labeled_video: bool = False,
        progress_file: Path | None = None,
        decoder: _Decoder | None = None,
        bbox_file: str | Path | None = None,
    ) -> PredictionResult:
        """Predicts on a video file and computes unsupervised loss metrics if applicable.

        Args:
            video_file (str | Path): Path to the video file.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`.
                If set to None, outputs are not saved.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_video (bool, optional): Whether to save a labeled video.
                Defaults to False.
            progress_file (Path, optional): Path to a file to save progress information for the
                App. Defaults to None.
            decoder (optional): which video-decoding backend to use, "dali" or "pynvvc".
                None (default) auto-selects pynvvc if it's usable on this machine for this
                video, else falls back to dali. Independent of the model's runtime
                (eager/onnx) and torch.compile -- this only controls video ingestion.
            bbox_file (str | Path, optional): Path to a per-frame bbox CSV (columns x, y, h, w;
                one row per frame). When provided, each frame is cropped to its bounding box
                before being passed to the model, and predictions are returned in the original
                coordinate space. Single-view only. Defaults to None.

        Returns:
            PredictionResult: A PredictionResult object containing the predictions and metrics.

        Examples:
            >>> result = model.predict_on_video_file("path/to/video.mp4")
            >>> result.predictions   # pd.DataFrame, one row per frame

            Save a keypoint-annotated video alongside the predictions CSV:
            >>> result = model.predict_on_video_file(
            ...     "path/to/video.mp4",
            ...     generate_labeled_video=True,
            ... )

        """
        self._load()
        video_file = Path(video_file)

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.video_preds_dir()

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_csv_file = output_dir / f"{video_file.stem}.csv"

        df = predict_video(
            video_file=str(video_file),
            model=self,
            output_pred_file=str(prediction_csv_file),
            progress_file=progress_file,
            decoder=decoder,
            bbox_file=bbox_file,
        )
        if generate_labeled_video:
            labeled_mp4_file = str(self.labeled_videos_dir() / f"{video_file.stem}_labeled.mp4")
            generate_labeled_video_fn(
                video_file=str(video_file),
                preds_df=df,
                output_mp4_file=labeled_mp4_file,
                confidence_thresh_for_vid=self.cfg.eval.confidence_thresh_for_vid,
                colormap=self.cfg.eval.get("colormap", "cool"),
            )

        if compute_metrics:
            # FIXME: Data module is only used for computing PCA metrics.
            data_module = _build_datamodule_pred(self.cfg)
            metrics = compute_metrics_single(
                cfg=self.cfg,
                labels_file=None,
                preds_file=str(prediction_csv_file),
                data_module=data_module,
            )
        else:
            metrics = None

        return PredictionResult(predictions=df, metrics=metrics)

    def predict_on_video_file_multiview(
        self,
        video_file_per_view: list[str] | list[Path],
        output_dir: str | Path | None = UNSPECIFIED,
        compute_metrics: bool = True,
        generate_labeled_video: bool = False,
        progress_file: Path | None = None,
        decoder: _Decoder | None = None,
    ) -> MultiviewPredictionResult:
        """Version of ``predict_on_video_file`` that accesses multiple camera views of each frame.

        Args:
            video_file_per_view: a list of video files each from a different view of the same
                session; number of files must match ``view_names`` in the config; order does not
                matter as files are matched to views by filename.
            output_dir: directory to save outputs to; defaults to
                ``{model_dir}/video_preds``; set to None to skip saving.
            compute_metrics: whether to compute pixel error and loss metrics on predictions.
            generate_labeled_video: whether to save a labeled video.
            progress_file: path to a file to save progress information for the App.
            decoder: which video-decoding backend to use, "dali" or "pynvvc". None (default)
                auto-selects pynvvc if it's usable on this machine for this video, else falls
                back to dali.

        Returns:
            object containing the predictions and metrics for each view.
        """
        if not self.config.is_multi_view():
            raise ValueError('predict_on_video_file_multiview requires a multi-view model')
        self._load()

        view_names = self.config.cfg.data.view_names
        if len(video_file_per_view) != len(view_names):
            raise ValueError(
                f'expected {len(view_names)} video files (one per view), '
                f'got {len(video_file_per_view)}'
            )

        video_file_per_view = [Path(f) for f in video_file_per_view]

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.video_preds_dir()

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Arranges video_file_per_view to be in the same order as cfg.data.view_names.
        _view_to_video_file: dict[str, Path] = io_utils.collect_video_files_by_view(
            video_file_per_view, view_names
        )
        video_file_per_view = [
            _view_to_video_file[view_name] for view_name in view_names
        ]

        prediction_csv_file_list = [
            str(output_dir / f"{video_file.stem}.csv")
            for video_file in video_file_per_view
        ]

        df_list = predict_video(
            video_file=list(map(str, video_file_per_view)),
            model=self,
            output_pred_file=prediction_csv_file_list,
            progress_file=progress_file,
            decoder=decoder,
        )
        if generate_labeled_video:
            for video_file, preds_df in zip(video_file_per_view, df_list, strict=True):
                labeled_mp4_file = str(
                    self.labeled_videos_dir() / f"{video_file.stem}_labeled.mp4"
                )
                generate_labeled_video_fn(
                    video_file=str(video_file),
                    preds_df=preds_df,
                    output_mp4_file=labeled_mp4_file,
                    confidence_thresh_for_vid=self.cfg.eval.confidence_thresh_for_vid,
                    colormap=self.cfg.eval.get("colormap", "cool"),
                )

        data_module = _build_datamodule_pred(self.cfg)
        if compute_metrics:
            metrics = {}
            for view_name, preds_file in zip(view_names, prediction_csv_file_list, strict=True):
                metrics[view_name] = compute_metrics_single(
                    cfg=self.cfg,
                    labels_file=None,
                    preds_file=preds_file,
                    data_module=data_module,
                )
        else:
            metrics = None

        df_dict = {view_name: df for view_name, df in zip(view_names, df_list, strict=True)}

        return MultiviewPredictionResult(predictions=df_dict, metrics=metrics)


def _build_datamodule_pred(cfg: DictConfig | ListConfig) -> BaseDataModule | UnlabeledDataModule:
    """Build a data module configured for prediction (no augmentation).

    Args:
        cfg: model config; augmentation is overridden to ``"default"`` (resize only).

    Returns:
        data module ready for use with `predict_dataset`.
    """
    cfg_pred = copy.deepcopy(cfg)
    # open_dict: imgaug_hflip (and any future prediction-only flag added here) may be
    # absent from configs saved by older LP versions; plain assignment would raise
    # ConfigAttributeError if cfg is struct-mode (e.g. composed via hydra_overrides).
    with open_dict(cfg_pred.training):
        cfg_pred.training.imgaug = "default"
        cfg_pred.training.imgaug_hflip = False
    imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
    dataset_pred = get_dataset(
        cfg=cfg_pred,
        data_dir=cfg_pred.data.data_dir,
        imgaug_transform=imgaug_transform_pred,
    )
    data_module_pred = get_data_module(
        cfg=cfg_pred, dataset=dataset_pred, video_dir=cfg_pred.data.video_dir
    )

    return data_module_pred
