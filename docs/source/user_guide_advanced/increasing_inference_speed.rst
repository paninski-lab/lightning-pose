.. _increasing_inference_speed:

##########################
Increasing Inference Speed
##########################

In addition to :ref:`running inference at reduced precision <mixed_precision>`, Lightning Pose
models can be accelerated further with three additional techniques:
`torch.compile() <https://pytorch.org/docs/stable/generated/torch.compile.html>`_,
`ONNX Runtime <https://onnxruntime.ai/>`_, and
`TensorRT <https://developer.nvidia.com/tensorrt>`_. This page benchmarks all four options
together (including eager FP16 for reference) and walks through how to use each one, assuming
you've already trained a model with ``litpose train``.

**TL;DR**

- **All three techniques help, and complexity roughly tracks with payoff.** ``torch.compile()``
  is a one-line change and gives solid gains. ONNX Runtime requires an export step but is the
  simplest option to deploy without a full Python/PyTorch runtime. TensorRT requires the most
  setup (matching CUDA/TensorRT library versions) but wins across every model and GPU tested,
  up to **8.32x** on the 6-view multiview transformer on an A100.
- **None of these techniques change the model's predictions.** We compared final keypoint
  predictions against the eager FP32 baseline on real video frames for all three methods --
  max deviation was under 0.08px in every case, consistent with ordinary floating-point kernel
  differences and far below any dataset's typical pixel error (see
  :ref:`Accuracy check <accuracy_check>` below).
- **These numbers are isolated forward-pass speedups** (no data loading), same methodology as
  the mixed-precision speed numbers. See the :ref:`caveat <caveats>` at the bottom of this page
  about the gap between forward-pass and end-to-end ``litpose predict`` speed.

Overview
========

- **Eager FP16 / BF16** -- no code changes beyond ``--precision``, see
  :ref:`Mixed Precision Training & Inference <mixed_precision>`. Included in the table below
  for reference.
- **torch.compile()** -- a single-line PyTorch feature that JIT-compiles the model's forward
  pass into fused GPU kernels. No export step, no new dependencies. See
  :ref:`Usage: torch.compile() <usage_torch_compile>`.
- **ONNX Runtime** -- export the model to the ONNX format, then run inference through
  ``onnxruntime``'s ``CUDAExecutionProvider``. Lets you deploy without a full PyTorch install.
  See :ref:`Usage: ONNX Runtime <usage_onnx_runtime>`.
- **TensorRT** -- same ONNX export, but run through onnxruntime's ``TensorrtExecutionProvider``,
  which builds an autotuned, hardware-specific inference engine. Most setup, biggest gains. See
  :ref:`Usage: TensorRT <usage_tensorrt>`.

Results
=======

Best-case speedup vs. eager FP32, FP16, isolated forward pass (10 warmup + 100 timed passes,
largest batch size tested per model/GPU: batch 64 for ResNet50/ViT-S, batch 16 -- effective
ViT batch 96 across 6 views -- for the multiview transformer):

.. list-table::
   :header-rows: 1

   * - Model / GPU
     - Eager FP16
     - torch.compile + FP16
     - ONNX Runtime FP16
     - TensorRT FP16
   * - ResNet50 -- L4
     - 1.96x
     - 2.82x
     - 1.98x
     - **4.73x**
   * - ResNet50 -- A100
     - 1.57x
     - 2.41x
     - 1.48x
     - **4.51x**
   * - ViT-S (single-view) -- L4
     - 3.17x
     - 4.10x
     - 2.36x
     - **5.43x**
   * - ViT-S (single-view) -- A100
     - 4.69x
     - 6.18x
     - 3.13x
     - **7.82x**
   * - Multiview (6-view) -- L4
     - 3.24x
     - 4.14x
     - 1.74x
     - **5.27x**
   * - Multiview (6-view) -- A100
     - 4.56x
     - 6.33x
     - 3.12x
     - **8.32x**

TensorRT wins in every case, sometimes by a wide margin -- particularly on the multiview model,
where more compute per forward pass gives it more to work with.

.. _accuracy_check:

Accuracy check
==============

Before recommending any of these, we checked whether they change the model's actual
predictions. Using a real trained checkpoint (ResNet50, ``mirror-mouse-fused``) and 5 sampled
frames from a real test video, we compared keypoint predictions from each accelerated method
against the eager FP32 reference, reusing the same real preprocessing and keypoint-extraction
code as ``Model.predict_frame`` -- the only thing swapped per method was the "images ->
heatmaps" compute step itself.

.. list-table::
   :header-rows: 1

   * - Method
     - Mean pixel deviation vs. eager FP32
     - Max pixel deviation vs. eager FP32
   * - Eager FP16
     - 0.014px
     - 0.049px
   * - torch.compile (FP32)
     - 0.004px
     - 0.071px
   * - ONNX Runtime (FP32)
     - 0.003px
     - 0.020px
   * - TensorRT (FP32)
     - 0.005px
     - 0.078px

All four methods land within a small fraction of a pixel of the eager FP32 reference --
consistent with expected floating-point kernel differences rather than any real change in the
computation.

Usage
=====

The tutorials below assume you've already trained a model with ``litpose train`` and have a
``model_dir`` containing ``config.yaml`` and a checkpoint.

.. _usage_torch_compile:

torch.compile
-------------

``Model.compile()`` compiles the model's forward pass in place. Call it after loading the
model and before running prediction:

.. code-block:: python

    from lightning_pose.api import Model

    model = Model.from_dir("path/to/model_dir")
    model.compile()

    # use as normal -- the compiled graph is now used internally
    result = model.predict_on_video_file("path/to/video.mp4")

The same thing is available from the CLI with the ``--compile`` flag:

.. code-block:: console

    litpose predict /path/to/model_dir /path/to/video.mp4 --compile

The first prediction after compiling triggers compilation and is therefore *slower* than an
uncompiled run -- expect tens of seconds of one-time overhead. This is expected. Subsequent
calls with the same input shape reuse the compiled graph. Changing batch size or input
resolution triggers a new compilation automatically. Calling ``compile()`` more than once is
a no-op.

.. note::
   ``compile()`` compiles the model's ``forward`` method and assigns it back onto the model,
   rather than wrapping the whole module. Wrapping the module (``model.model =
   torch.compile(model.model)``) works if you call ``model.model(images)`` directly in your
   own code, but does **not** reliably engage the compiled graph when going through
   ``predict_on_video_file`` / ``predict_frame`` / ``predict_on_label_csv``, because those
   call ``get_loss_inputs_labeled`` rather than ``forward`` -- and calling anything other
   than ``forward``/``__call__`` on a compiled module silently falls back to the *original*,
   uncompiled submodule.

.. note::
   ``torch.compile()`` uses the TorchInductor backend, which generates Triton kernels and
   therefore requires a GPU of CUDA compute capability **7.0 or higher** (Volta and newer).
   On older GPUs -- for example the GTX 10-series (capability 6.x) -- ``compile()`` raises
   a ``RuntimeError`` immediately, rather than failing partway through the first
   prediction. Check your device with
   ``python -c "import torch; print(torch.cuda.get_device_capability())"``.

.. _usage_onnx_runtime:

ONNX Runtime
------------

Export the model's forward pass to ONNX, then run inference through ``onnxruntime``, plugging
the ONNX session back in as the model's forward pass the same way as above:

.. code-block:: python

    import numpy as np
    import onnxruntime as ort
    import torch
    from lightning_pose.api import Model

    model = Model.from_dir("path/to/model_dir")
    real_module = model.model

    resize_h = model.cfg.data.image_resize_dims.height
    resize_w = model.cfg.data.image_resize_dims.width
    # shape is (1, num_views, 3, H, W) instead for a multi-view model
    dummy = torch.randn(1, 3, resize_h, resize_w, device="cuda")

    torch.onnx.export(
        real_module, dummy, "/path/to/model.onnx",
        input_names=["images"], output_names=["heatmaps"],
        dynamic_axes={"images": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )

    session = ort.InferenceSession(
        "/path/to/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    def onnx_forward(images):
        io_binding = session.io_binding()
        io_binding.bind_input(
            name="images", device_type="cuda", device_id=0,
            element_type=np.float32, shape=tuple(images.shape),
            buffer_ptr=images.contiguous().data_ptr(),
        )
        io_binding.bind_output(name="heatmaps", device_type="cuda", device_id=0)
        session.run_with_iobinding(io_binding)
        return torch.from_numpy(io_binding.copy_outputs_to_cpu()[0]).to(images.device)

    real_module.forward = onnx_forward
    result = model.predict_on_video_file("path/to/video.mp4")

Requires ``pip install onnxruntime-gpu`` -- make sure the CUDA version matches your PyTorch
install. Pip's default ``onnxruntime-gpu`` build may target a newer CUDA major version than
your PyTorch install uses; check Microsoft's ``onnxruntime-cuda-12`` package index if you're on
CUDA 12.

.. _usage_tensorrt:

TensorRT
--------

TensorRT engines are built through the same ONNX file, using onnxruntime's
``TensorrtExecutionProvider`` instead of ``CUDAExecutionProvider`` (continuing the snippet
above):

.. code-block:: python

    trt_options = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/path/to/trt_cache",
        "trt_fp16_enable": True,  # set False to retain FP32 precision
        "trt_profile_min_shapes": f"images:1x3x{resize_h}x{resize_w}",
        "trt_profile_opt_shapes": f"images:1x3x{resize_h}x{resize_w}",
        "trt_profile_max_shapes": f"images:1x3x{resize_h}x{resize_w}",
    }
    session = ort.InferenceSession(
        "/path/to/model.onnx",
        providers=[
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    # Always confirm TensorRT actually engaged -- onnxruntime silently falls back to
    # CUDAExecutionProvider if TensorRT can't be used, e.g. a missing library.
    assert "TensorrtExecutionProvider" in session.get_providers()

The first inference call builds and autotunes an engine for the shapes in the profile above,
which can take anywhere from several seconds to a few minutes depending on the model. The
engine is cached to ``trt_engine_cache_path`` and reused on subsequent runs with the same
shapes.

Getting TensorRT installed and working took some real trial and error, mostly around matching
library versions:

- ``pip install tensorrt`` and ``pip install onnxruntime-gpu`` can each default to newer
  CUDA/TensorRT major versions that don't match each other or your PyTorch install. We needed
  ``pip install "tensorrt-cu12<11"`` (landing on TensorRT 10.16.1.11) to match onnxruntime
  1.28.0's expected ``libnvinfer.so.10``.
- TensorRT's shared libraries need to be on ``LD_LIBRARY_PATH`` at runtime, e.g.:

  .. code-block:: bash

      export LD_LIBRARY_PATH=/path/to/site-packages/tensorrt_libs:$LD_LIBRARY_PATH

- Always check ``session.get_providers()`` after creating the session. If TensorRT can't
  load (e.g. a missing library), onnxruntime silently falls back to
  ``CUDAExecutionProvider`` rather than raising an error -- your code will still run and
  produce plausible-looking numbers, just not the ones you think.

.. _caveats:

Caveats
=======

- **These are isolated forward-pass numbers, not end-to-end** ``litpose predict`` **timings.**
  Real inference also includes data loading (DALI) and postprocessing. These forward-pass
  gains aren't guaranteed to translate 1:1 into end-to-end speedups as-is.
- **cuDNN TF32 was left on for the ResNet50 eager-FP32 baseline** (only matmul TF32 was
  disabled), so it's not a fully strict FP32 number -- doesn't change the direction of any
  result here.
- A single dynamic-shape TensorRT profile was used per (model, GPU, precision), covering the
  full batch-size range tested, rather than a separate engine per exact batch size. This is
  simpler but can leave some performance on the table at batch sizes far from the profile's
  optimum -- for example, multiview-FP32-on-L4 was roughly flat (0.91-1.07x) across batch sizes
  rather than showing a clear win.
