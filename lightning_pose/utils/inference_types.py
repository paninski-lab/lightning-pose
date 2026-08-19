"""Shared type aliases for inference-time configuration: precision, execution
runtime, export target, and video decoder.

Kept in a standalone module with no heavy dependencies (no torch/cv2/pandas) so
`lightning_pose.cli` can import these values eagerly, at argument-parser
construction time, without paying the cost of importing `lightning_pose.api`
(which pulls in torch and friends) just to build ``--help`` output. The API layer
(``lightning_pose.api.model``, ``lightning_pose.api.model_runtime``) imports and
re-exports these same names rather than redefining them, so each choice set has
exactly one definition -- ``typing.get_args()`` on the alias is the single source
for both the CLI's ``choices=`` tuples and any runtime validation.
"""

from typing import Literal

# User-facing precision strings for both the CLI (--precision) and the Model API
# (Model.from_dir(precision=...)). Kept identical across CLI and API so a value
# can be passed straight through without a separate lookup table at the CLI layer.
_Precision = Literal["fp32", "fp16", "bf16"]

# Inference runtimes accepted by Model.from_dir(runtime=...) / `litpose predict
# --runtime`. "eager" runs the loaded PyTorch checkpoint; "onnx" and "tensorrt"
# run a session previously built with `litpose export --runtime onnx` / `--runtime
# tensorrt`.
_Runtime = Literal["eager", "onnx", "tensorrt"]

# Export targets accepted by Model.export(runtime=...) / `litpose export
# --runtime`. Narrower than _Runtime -- there is no "eager" export, since eager
# inference just runs the checkpoint directly with no export step.
_ExportRuntime = Literal["onnx", "tensorrt"]

# Video-decoding backends accepted by predict_on_video_file(_multiview)'s decoder=
# kwarg / `litpose predict --decoder`. Independent of _Runtime -- decoder controls
# video ingestion, _Runtime controls model execution.
_Decoder = Literal["dali", "pynvvc"]

# Weight precisions supported for ONNX export, accepted by Model.export(
# onnx_precision=...) / `litpose export --onnx-precision`. Deliberately narrower
# than _Precision -- this is the precision baked into the .onnx file itself, not
# the autocast precision used for eager inference.
_OnnxPrecision = Literal["fp32", "fp16"]
