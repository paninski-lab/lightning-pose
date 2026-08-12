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
  up to **8.32x** on the 6-view multiview transformer on an A100.
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
- **TensorRT** -- same ONNX export, but run through onnxruntime's ``TensorrtExecutionProvider``,
  which builds an autotuned, hardware-specific inference engine. Most setup, biggest gains. See
  :ref:`Usage: TensorRT <usage_tensorrt>`.
- **PyNvVideoCodec** -- hardware-accelerated video *decoding* via direct NVIDIA Video Codec SDK
  bindings, as an alternative to DALI's video reader. Independent of the model-processing
  techniques above -- see :ref:`PyNvVideoCodec (Video Decoding) <pynvvc_decoding>`.
- **These techniques are complementary and can be combined** -- e.g. torch.compile() or
  TensorRT for the model together with PyNvVideoCodec for decoding -- for the best end-to-end
  throughput. Not yet benchmarked together on this page; a combined benchmark is planned for a
  follow-up PR.

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

On a T4, decoding the same video with PyNvVideoCodec measured **8.76x faster** than DALI's GPU
reader (3735.7 vs 426.5 fps, 7 timed runs). This is a decode-throughput-only number, not an
end-to-end ``litpose predict`` speedup -- how much it moves the needle overall depends on how
much of your pipeline's time is spent decoding vs. running the model (see the data-loading
caveat below).

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

Decode speed: L4 and A100, single- and multi-view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decode-only throughput (no model), 7 timed runs each, 64-frame batches:

.. list-table::
   :header-rows: 1

   * - Config
     - GPU
     - DALI
     - PyNvVideoCodec
     - Speedup
   * - Single-view (``mirror-mouse-fused``, 1 view)
     - L4
     - 1516 fps
     - 4442 fps
     - **2.93x**
   * - Single-view (``mirror-mouse-fused``, 1 view)
     - A100
     - 1113 fps
     - 3249 fps
     - **2.92x**
   * - Multi-view (``fly-anipose``, 6 views)
     - L4
     - 5347 fps
     - 3674 fps
     - 0.69x (slower)
   * - Multi-view (``fly-anipose``, 6 views)
     - A100
     - 4487 fps
     - 2684 fps
     - 0.60x (slower)

.. note::

   The multi-view result reverses direction from single-view: PyNvVideoCodec is
   clearly faster for a single video, but *slower* than DALI once 6 views are
   involved. The current implementation reads each view's decoder sequentially in
   a loop (one ``get_batch_frames()`` call after another, not concurrently) --
   which is also exactly what ``--decoder pynvvc`` does in production for
   multiview, so this is a real characteristic of the current implementation, not
   a benchmark artifact. DALI's pipeline appears to handle multiple video streams
   more efficiently under its own internal scheduling. For single-view video,
   ``--decoder pynvvc`` is a clear win; for multiview, it's currently a net loss on
   raw decode speed even though the model's forward pass itself still speeds up
   with reduced precision (see the multiview forward-pass numbers above) --
   something worth revisiting if concurrent per-view decoding is added later.

   Separately, single-view decode speedup is notably lower on L4/A100 here
   (~2.9x) than the T4 number above (8.76x) -- not yet investigated further, but
   worth keeping in mind that the T4 number shouldn't be assumed to generalize to
   newer GPUs.

.. _caveats:

Caveats
=======

- **These are isolated forward-pass numbers, not end-to-end** ``litpose predict`` **timings.**
  Real inference also includes data loading (DALI) and postprocessing. These forward-pass
  gains aren't guaranteed to translate 1:1 into end-to-end speedups as-is. See :ref:`PyNvVideoCodec <pynvvc_decoding>` above for a way to speed up the data-loading half of that gap.
- **cuDNN TF32 was left on for the ResNet50 eager-FP32 baseline** (only matmul TF32 was
  disabled), so it's not a fully strict FP32 number -- doesn't change the direction of any
  result here.
- A single dynamic-shape TensorRT profile was used per (model, GPU, precision), covering the
  full batch-size range tested, rather than a separate engine per exact batch size. This is
  simpler but can leave some performance on the table at batch sizes far from the profile's
  optimum -- for example, multiview-FP32-on-L4 was roughly flat (0.91-1.07x) across batch sizes
  rather than showing a clear win.
