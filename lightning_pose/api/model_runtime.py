"""Model execution-backend logic: eager checkpoint loading, ``torch.compile()``, and
ONNX/TensorRT export and runtime attachment.

`_RuntimeMixin` is mixed into `lightning_pose.api.model.Model`. It owns every method
that changes *how* a loaded model executes -- eager weight loading, compilation, and
export/attachment of the ONNX and TensorRT backends. `model.py` owns construction and
the `predict_*` methods that decide *what* to run the model on.
"""

from __future__ import annotations

import copy
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.models import ALLOWED_MODELS, get_model_class
from lightning_pose.utils import io as io_utils
from lightning_pose.utils.inference_types import _ExportRuntime, _OnnxPrecision

if TYPE_CHECKING:
    from lightning_pose.api.model import Model

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []

# Temporal context length used when tracing a context (MHCRNN) model for export.
# Matches the (T, H, W, 3) convention documented in Model.predict_frame.
_CONTEXT_SEQUENCE_LENGTH = 5

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
            torch.cuda.synchronize(images.device)
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


class _RuntimeMixin(Model if TYPE_CHECKING else object): # pyright: ignore[reportGeneralTypeIssues]
    """Mixin providing `Model`'s execution-backend methods.

    Always mixed into `lightning_pose.api.model.Model`, which supplies
    `model_dir`, `config`/`cfg`, `model`, `precision`, `_compiled`, `_runtime`, and
    the `exports_onnx_dir()`/`exports_trt_dir()` path helpers this mixin reads and
    writes. The conditional inheritance from `Model` at TYPE_CHECKING time gives the
    type checker visibility into those attributes without a real circular import at
    runtime -- `model.py` imports this mixin, so the reverse can't be a real base.
    """

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
            str(onnx_path),
            providers=[
                ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider",
            ],
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
        # Free the eager backbone's GPU memory -- the ONNX graph already has the
        # trained weights baked in, and self.backbone is never called again from
        # any inference path once forward() is replaced above (only from
        # get_parameters(), which is training-only). Without this, the eager
        # weights and the ONNX Runtime session are both resident on GPU at once.
        self.model.backbone.to("cpu")
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
            providers=[
                ("TensorrtExecutionProvider", trt_options),
                ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            ],
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
        # See the matching comment in _attach_onnx_runtime() -- same fix, same reason.
        self.model.backbone.to("cpu")
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
        runtime: _ExportRuntime,
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
        if runtime not in get_args(_ExportRuntime):
            supported = ", ".join(repr(r) for r in get_args(_ExportRuntime))
            raise ValueError(
                f"Unsupported export runtime: '{runtime}'. Use one of {supported}."
            )
        if onnx_precision not in get_args(_OnnxPrecision):
            supported = ", ".join(repr(p) for p in get_args(_OnnxPrecision))
            raise ValueError(
                f"Unsupported onnx_precision: '{onnx_precision}'. Use one of {supported}."
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
