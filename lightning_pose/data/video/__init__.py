"""Video-reading backends for training and prediction.

Three backends currently implement the same reader contract (see below): ``dali.py`` (GPU video
loading via NVIDIA DALI, training + prediction), ``pynvvc.py`` (direct NVDEC access via
PyNvVideoCodec, prediction only), and ``opencv.py`` (CPU decode via OpenCV, prediction only). All
three plug into the same call sites -- ``lightning_pose.data.datamodules.UnlabeledDataModule``
for training, ``lightning_pose.utils.predictions.predict_video`` for prediction -- which select a
backend by name (``--reader``/``reader=`` in the API) or by auto-probing hardware availability.

Backends
--------

- ``dali.py`` -- :class:`~lightning_pose.data.video.dali.PrepareDALI` /
  :class:`~lightning_pose.data.video.dali.LitDaliWrapper`. GPU-accelerated, NVIDIA + Linux
  x86_64 only (``nvidia-dali-cuda110`` is platform-gated in ``pyproject.toml``). The only
  backend used for training; also used for prediction.
- ``pynvvc.py`` -- :class:`~lightning_pose.data.video.pynvvc.PreparePynvvc` /
  :class:`~lightning_pose.data.video.pynvvc.LitPynvvcWrapper`. Direct NVDEC access via
  PyNvVideoCodec, Linux x86_64 plus a supported NVIDIA GPU/driver generation only. Prediction
  only.
- ``opencv.py`` -- :class:`~lightning_pose.data.video.opencv.PrepareOpenCV` /
  :class:`~lightning_pose.data.video.opencv.LitOpenCVWrapper`. CPU decode via
  ``opencv-python-headless``, an unconditional cross-platform dependency -- works on any
  platform/hardware, including macOS, Windows, and GPU-less machines. Slower than DALI/pynvvc on
  a supported NVIDIA box; the guaranteed fallback everywhere else. Prediction only.

Adding a new video reader
--------------------------

Every backend module implements the same contract. To add one:

1. **Two-phase prepare class**: ``Prepare<Name>.__init__(model_type, filenames, resize_dims,
   dali_config, bbox_df=None, ...)`` validates inputs and precomputes windowing parameters
   (raise early, before any GPU/decoder allocation); ``__call__()`` builds and returns a
   ``Lit<Name>Wrapper``. (The ``dali_config`` parameter name is a pre-existing wart --
   ``PreparePynvvc`` already reuses ``cfg.dali``'s windowing section rather than inventing a
   per-backend config block, since the windowing semantics are decoder-agnostic; new backends
   should follow the same precedent rather than fixing it in isolation.)
2. **Iterator wrapper class**: ``Lit<Name>Wrapper`` is iterable, defines ``__len__``, and yields
   :class:`~lightning_pose.data.datatypes.UnlabeledBatchDict` /
   :class:`~lightning_pose.data.datatypes.MultiviewUnlabeledBatchDict` -- FCHW float frames,
   ImageNet-normalized, plus ``transforms`` and ``bbox``. Match ``LitDaliWrapper`` /
   ``LitPynvvcWrapper``'s shape exactly so downstream code (``PredictionHandler``,
   ``trainer.predict``) doesn't need to know which backend produced a batch.
3. **Availability probe (optional)**: ``is_<name>_available(video_path, ...) -> bool``, only
   needed if the backend can be installed but still unusable for a given machine/video (wrong
   hardware/driver generation, as with pynvvc). Skip it for a backend that's simply present or
   absent as a plain package dependency.
4. **Import discipline**: if the backend's package is platform-gated or proprietary (not a
   guaranteed cross-platform install), every import of it must be lazy -- inside the function or
   method body that uses it, never at module top level (see the ``# lazy: avoids ImportError on
   cpu-only installs`` convention in ``dali.py``/``pynvvc.py``). A top-level import of such a
   package anywhere in the package or tests will break ``import lightning_pose`` on machines that
   don't have it. If the package is an unconditional, cross-platform dependency already declared
   in ``pyproject.toml`` (like ``opencv-python-headless``), top-level imports are fine.
5. **Wire it in**:

   - add the name to ``_Reader`` in ``lightning_pose/utils/inference_types.py`` (the single
     source for the CLI's ``--reader`` choices and API type hints);
   - add a dispatch branch -- and, if step 3 applies, a fallback rung -- in ``predict_video()``
     (``lightning_pose/utils/predictions.py``);
   - update the ``--reader`` help text in ``lightning_pose/cli/commands/predict.py``;
   - add this module to the package map above.
6. **Tests**: add ``tests/data/video/test_<name>.py`` mirroring a sibling reader's test module
   one-for-one; mark any test that needs real hardware with ``@pytest.mark.gpu`` so the CPU CI
   workflow (``pytest -m "not gpu"``) still exercises everything else.
"""

__all__: list[str] = []
