.. _increasing_inference_speed:

##########################
Increasing Inference Speed
##########################

In addition to :ref:`running inference at reduced precision <mixed_precision>`, Lightning Pose
models can be accelerated further with additional techniques that speed up model processing --
`torch.compile() <https://pytorch.org/docs/stable/generated/torch.compile.html>`_,
`ONNX Runtime <https://onnxruntime.ai/>`_, and
`TensorRT <https://developer.nvidia.com/tensorrt>`_ -- and with hardware-accelerated video
decoding -- :ref:`PyNvVideoCodec <pynvvc_decoding>`. This page benchmarks the model-processing
techniques together (including eager FP16 for reference) and walks through how to use each one,
assuming you've already trained a model with ``litpose train``; see the
:ref:`PyNvVideoCodec section <pynvvc_decoding>` below for the video-decoding benchmarks and usage.

**TL;DR**

- **All three techniques help, and complexity roughly tracks with payoff.** ``torch.compile()``
  is a one-line change and gives solid gains. ONNX Runtime requires an export step but is the
  simplest option to deploy without a full Python/PyTorch runtime. TensorRT requires the most
  setup (matching CUDA/TensorRT library versions) but wins across every model and GPU tested,
  up to **2.93x** on the 6-view multi-view transformer on an L4 (TensorRT FP16 + DALI vs. eager
  FP32).
- **None of these techniques change the model's predictions.** We compared final keypoint
  predictions against the eager FP32 baseline on real video frames for all three methods --
  max deviation was under 0.08px in every case, consistent with ordinary floating-point kernel
  differences and far below any dataset's typical pixel error (see
  :ref:`Accuracy check <accuracy_check>` below).
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
- **TensorRT** -- same ONNX export, but run through ``onnxruntime``'s ``TensorrtExecutionProvider``,
  which builds an autotuned, hardware-specific inference engine. Most setup, biggest gains. See
  :ref:`Usage: TensorRT <usage_tensorrt>`.
- **PyNvVideoCodec** -- hardware-accelerated video *decoding* via direct NVIDIA Video Codec SDK
  bindings, as an alternative to DALI's video reader. Independent of the model-processing
  techniques above -- see :ref:`Usage: PyNvVideoCodec <pynvvc_decoding>`.
- **These techniques are complementary and can be combined** -- e.g. torch.compile() or
  TensorRT for the model together with PyNvVideoCodec for decoding -- for the best end-to-end
  throughput. See the figure in the following section.

Results
=======

The figure below benchmarks every combination of model architecture, precision/runtime
technique, video decoder, and GPU on real end-to-end ``litpose predict`` calls -- not an
isolated forward pass. Each bar is the mean of 10 timed repeats (1 discarded warmup run);
error bars show standard error of the mean.

.. figure:: https://i.imgur.com/INWfgoi.png
   :alt: Grouped bar chart of end-to-end predict speed across models, techniques, and GPUs

   End-to-end ``litpose predict`` timing for ResNet50 (single-view), ViT-S (single-view), and
   ViT-S (6-view multi-view). Each row shows 5 technique groups (eager FP32, eager FP16,
   torch.compile + FP16, ONNX Runtime FP16, TensorRT FP16), each with 4 bars: L4 + DALI,
   L4 + PyNvVideoCodec, A100 + DALI, A100 + PyNvVideoCodec.

TensorRT FP16 wins overall across every model and GPU, with the largest end-to-end gain
(2.93x) on the 6-view multi-view transformer on L4 -- notably larger than the same technique's
gain on A100, since A100's much faster eager-FP32 baseline leaves less relative headroom to
close. Decoder choice diverges sharply by model: for the single-view models, PyNvVideoCodec is
often the *faster* decoder on A100 once any model-acceleration technique is applied (9-25%
faster than DALI), though it trails DALI somewhat on L4. For the 6-view multi-view model, DALI
wins outright across every technique and GPU -- confirming the multi-view PyNvVideoCodec
slowdown documented below carries through to full end-to-end timing, and the gap is
substantially larger on A100 (44-53% slower) than on L4 (6-19% slower).

.. _caveats:

Caveats
=======

- **cuDNN TF32 was left on for the ResNet50 eager-FP32 baseline** (matmul TF32 is off by
  PyTorch's own default; ``predict_speed_matrix_benchmark.py`` sets neither flag explicitly,
  so both follow PyTorch's defaults). This means the eager-FP32 numbers in the figure above
  are not a fully strict FP32 baseline either -- doesn't change the direction of any result
  here, since TensorRT/ONNX Runtime FP16 still win by a wide margin, but it does mean the
  eager-FP32 bars are a few percent faster than a true FP32 baseline would be for
  convolution-heavy models like ResNet50.
- A single dynamic-shape TensorRT profile was used per (model, GPU, precision), covering the
  full batch-size range tested, rather than a separate engine per exact batch size. This is
  simpler but can leave some performance on the table at batch sizes far from the profile's
  optimum -- for example, multi-view-FP32-on-L4 was roughly flat (0.91-1.07x) across batch sizes
  rather than showing a clear win.
- **ONNX Runtime FP16 + DALI is missing from the figure for the 6-view multi-view model on L4**,
  likely due to a race condition between DALI's CUDA stream and the ONNX Runtime 
  ``CUDAExecutionProvider`` on L4 GPUs -- 
  see issue `483 <https://github.com/paninski-lab/lightning-pose/issues/483>`_.

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

.. _onnx_installation:

Installation
~~~~~~~~~~~~

ONNX Runtime is an optional dependency. ``onnxruntime-gpu`` wheels are built against a specific
CUDA major version at build time, and the wheel does **not** detect or adapt to whatever CUDA
you already have -- so check what your PyTorch install is using first:

.. code-block:: console

    python -c "import torch; print(torch.version.cuda)"

Then use that CUDA version to select the proper installation option from ONNX Runtime's
`installation selector <https://onnxruntime.ai/getting-started>`_.

Exporting additionally requires the ``onnx`` package, which ``onnxruntime-gpu`` does not pull
in as a dependency:

.. code-block:: console

    pip install onnx

Only ``Model.export()`` needs it. A machine that just runs an already-exported ``.onnx`` file
needs ``onnxruntime-gpu`` alone.

Usage
~~~~~

Exporting and loading are separate steps. Export once, then load with ``runtime="onnx"`` for
every subsequent prediction run:

.. code-block:: python

    from lightning_pose.api import Model

    # export once
    model = Model.from_dir("path/to/model_dir")
    model.export("onnx", onnx_precision="fp16")

    # load through ONNX Runtime and predict as normal
    onnx_model = Model.from_dir(
        "path/to/model_dir", runtime="onnx", onnx_precision="fp16"
    )
    result = onnx_model.predict_on_video_file("path/to/video.mp4")

The same two steps from the CLI:

.. code-block:: console

    litpose export /path/to/model_dir --runtime onnx --onnx-precision fp16
    litpose predict /path/to/model_dir /path/to/video.mp4 --runtime onnx --onnx-precision fp16

Exports are written to a fixed location inside the model directory:

.. code-block:: text

    model_dir/
    ├── config.yaml
    ├── tb_logs/.../checkpoints/epoch=214-step=12685-best.ckpt
    └── exports_onnx/
        ├── epoch=214-step=12685-best_fp16.onnx
        └── epoch=214-step=12685-best_fp32.onnx   # only if exported

You never pass export paths yourself -- ``export()`` always writes here and
``from_dir(runtime="onnx")`` always reads from here. The filename is the checkpoint's own stem
plus the export precision, so it stays unambiguous which checkpoint an export came from if a
model directory ever holds exports alongside an updated checkpoint. ``onnx_precision`` may be
omitted when exactly one export exists for the checkpoint, and is required to disambiguate when
several do.

.. note::

   ``precision`` and ``onnx_precision`` are different settings.

   * ``precision`` (``"fp32"``/``"fp16"``/``"bf16"``, also ``litpose predict --precision``)
     controls the autocast precision of an **eager** forward pass. It leaves the weights on
     disk untouched.
   * ``onnx_precision`` (``"fp32"``/``"fp16"``, also ``--onnx-precision``) is the weight
     precision **baked into the exported** ``.onnx`` **file itself**.

   With ``runtime="onnx"``, ``precision`` is ignored -- the exported file's own precision is
   what runs.

.. note::

   ``runtime="onnx"`` rebinds the model's ``forward`` method to the ONNX session rather than
   wrapping the module, for the same reason described in the ``torch.compile()`` note above.

.. note::

   ``torch.compile()`` has no effect on an ONNX Runtime session. Combining ``--compile`` with
   ``--runtime onnx``, or calling ``compile()`` on a model loaded with ``runtime="onnx"``,
   raises rather than silently doing nothing.

.. _usage_tensorrt:

TensorRT
--------

.. _tensorrt_installation:

Installation
~~~~~~~~~~~~~

TensorRT and ``onnxruntime-gpu`` need matching major versions, and ``pip install
tensorrt`` / ``pip install onnxruntime-gpu`` each default independently to whatever is
newest -- so they don't necessarily agree with each other, or with your PyTorch install's
CUDA version. Check what onnxruntime expects before installing TensorRT:

.. code-block:: console

    python -c "import onnxruntime; print(onnxruntime.__version__)"

The combination we verified working is ``onnxruntime-gpu`` 1.28.0 with
``pip install "tensorrt-cu12<11"`` (lands on TensorRT 10.16.1.11, providing
``libnvinfer.so.10``). A newer, unpinned ``pip install tensorrt`` can land on TensorRT
11.x (``libnvinfer.so.11``) instead, which fails at session-creation time with a
missing-symbol error rather than a clear version-mismatch message.

For the full installation matrix (which TensorRT version pairs with which CUDA/cuDNN), see NVIDIA's `TensorRT install guide <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>`_. The ``<11`` pin above matches what onnxruntime-gpu's TensorRT execution provider currently requires. If your onnxruntime-gpu version differs from the one this doc was written against, check onnxruntime's own `TensorRT EP requirements table <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements>`_ first -- it's the maintained source of truth for onnxruntime-gpu <-> TensorRT <-> CUDA compatibility, including versions released after this page was written.

TensorRT's shared libraries also need to be on ``LD_LIBRARY_PATH`` at runtime:

.. code-block:: bash

    # find your site-packages path:
    #   python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))"
    export LD_LIBRARY_PATH=/path/to/site-packages/tensorrt_libs:$LD_LIBRARY_PATH

Building a TensorRT engine starts from the ``.onnx`` file produced by ``model.export(
"onnx", ...)`` -- the ``onnx`` package is needed to create that file, but not to build
the engine from it afterward. A machine that only builds/runs TensorRT engines from an
already-exported ``.onnx`` file needs ``onnxruntime-gpu`` and ``tensorrt`` alone.

Usage
~~~~~

TensorRT builds its engine from an existing ONNX export, so export to ONNX first if you
haven't already:

.. code-block:: python

    from lightning_pose.api import Model

    model = Model.from_dir("path/to/model_dir")
    model.export("onnx", onnx_precision="fp16")
    model.export("tensorrt", onnx_precision="fp16", max_batch_size=32)

    trt_model = Model.from_dir(
        "path/to/model_dir", runtime="tensorrt", onnx_precision="fp16"
    )
    result = trt_model.predict_on_video_file("path/to/video.mp4")

The same from the CLI:

.. code-block:: console

    litpose export /path/to/model_dir --runtime onnx --onnx-precision fp16
    litpose export /path/to/model_dir --runtime tensorrt --onnx-precision fp16 --max-batch-size 32
    litpose predict /path/to/model_dir /path/to/video.mp4 --runtime tensorrt --onnx-precision fp16

``max_batch_size`` sets the upper end of the dynamic-shape batch profile the engine is
built for -- inference at a larger batch size fails. ``opt_batch_size`` (defaults to
``max_batch_size``) is the batch size the engine is tuned to run fastest at; correctness
holds for any batch size in ``[1, max_batch_size]``, but performance is best near
``opt_batch_size``.

The engine cache is written alongside the ONNX exports:

.. code-block:: text

    model_dir/
    ├── config.yaml
    ├── tb_logs/.../checkpoints/epoch=214-step=12685-best.ckpt
    ├── exports_onnx/
    │   └── epoch=214-step=12685-best_fp16.onnx
    └── exports_trt/
        └── epoch=214-step=12685-best_fp16/
            ├── trt_metadata.json
            └── ...                          # engine cache files (managed by onnxruntime)

The first call to ``model.export("tensorrt", ...)`` builds and autotunes the engine, which
can take anywhere from several seconds to a few minutes. ``trt_metadata.json`` records the
GPU name, TensorRT/onnxruntime versions, and batch profile the engine was built with --
``Model.from_dir(..., runtime="tensorrt")`` reads this back and warns if the current GPU
doesn't match, since (unlike an ``.onnx`` file) a built engine is tied to the exact GPU
architecture it was built on and is not portable across machines.

.. note::

   Building always requires a real GPU with a working TensorRT/onnxruntime install --
   there is no CPU fallback tier for ``runtime="tensorrt"``. If
   ``TensorrtExecutionProvider`` can't load, this raises rather than silently running on
   ``CUDAExecutionProvider`` or CPU (unlike plain ONNX Runtime, which does have a CPU
   tier -- see the note in the ONNX Runtime section above about checking
   ``session.get_providers()``).

.. note::

   The accuracy check earlier on this page used TensorRT at FP32. At FP16, quantization
   can shift the heatmap's predicted peak by a few pixels on keypoints the model has no
   real opinion about -- a near-uniform heatmap, likelihood near the noise floor -- a
   "peak-flipping" effect that happens under any precision or kernel change (it
   reproduces on the already-merged ONNX FP16 path too, given a real dataset rather than
   a handful of hand-picked frames) and isn't specific to TensorRT. On
   confidently-tracked keypoints it stays small: repeated FP16 engine builds during
   testing measured mean deviation around 0.2-0.3px, though TensorRT's own build-time
   autotuning adds some run-to-run variance on top -- engine builds aren't perfectly
   deterministic, so don't expect bit-identical predictions from two separately-built
   engines either.


.. _pynvvc_decoding:

PyNvVideoCodec (Video Decoding)
--------------------------------

Everything above speeds up the *model's forward pass*. This section covers a separate,
complementary axis: how the video frames themselves get decoded off disk before they ever
reach the model. By default, ``litpose predict`` decodes video with DALI's GPU video reader.
PyNvVideoCodec (direct NVIDIA Video Codec SDK / NVDEC bindings) is an alternative decoder
backend that talks to the hardware decoder more directly, skipping DALI's pipeline/graph
framework overhead.

.. note::

   ``PreparePynvvc`` (the class backing ``--decoder pynvvc``) reads its window size from
   the existing ``cfg.dali`` config section --
   ``dali.base.predict.sequence_length`` / ``dali.context.predict.sequence_length`` --
   rather than a separate ``cfg.pynvvc`` section. The name is a little confusing, since
   this setting also governs how many frames PyNvVideoCodec reads per batch, but the
   windowing semantics (frames per iteration, overlap for context/MHCRNN models) are
   identical between the two decoder backends, so one config value covers both rather
   than keeping two in sync. If you only ever use ``--decoder pynvvc`` and never DALI,
   you still configure the window size via ``dali.*.predict.sequence_length``.

Installation
~~~~~~~~~~~~

.. code-block:: bash

    pip install PyNvVideoCodec

Requires an NVIDIA GPU from the Turing generation or newer (Turing/Ampere/Ada/Hopper/Blackwell)
and driver version 530.41.03 or newer on Linux. If PyNvVideoCodec can't decode a given video on
the current machine (unsupported GPU generation, driver too old, package not installed, or an
unsupported video format), ``litpose predict`` automatically falls back to DALI rather than
erroring -- see below.

Usage
~~~~~

``--decoder`` is independent of ``--runtime``/``--compile`` above -- it only controls how video
frames are read, not how the model runs on them. From the CLI:

.. code-block:: console

    litpose predict /path/to/model_dir /path/to/video.mp4 --decoder pynvvc
    litpose predict /path/to/model_dir /path/to/video.mp4 --decoder dali

Or from the API:

.. code-block:: python

    model = Model.from_dir("path/to/model_dir")
    result = model.predict_on_video_file("path/to/video.mp4", decoder="pynvvc")

Omitting ``--decoder`` (or passing ``decoder=None``) auto-selects PyNvVideoCodec when it's
usable on the current machine for the given video, falling back to DALI otherwise -- no error,
just a quieter/slower run. Explicitly requesting ``--decoder pynvvc`` on a machine where it
isn't usable raises a clear error instead of silently falling back, so you know your run isn't
using the backend you asked for.
