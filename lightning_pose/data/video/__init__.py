"""Video-reading backends for training and prediction.

Three backends implement the same reader contract:

- ``dali.py`` (GPU video loading via NVIDIA DALI, training + prediction)
- ``pynvvc.py`` (direct NVDEC access via PyNvVideoCodec, prediction only)
- ``opencv.py`` (CPU decode via OpenCV, prediction only)

:func:`~lightning_pose.data.video.factory.build_video_reader` selects among them
by name (``--reader``/``reader=`` in the API) or by auto-probing hardware availability;
see ``factory.py`` for the dispatch logic and the "Adding a new video reader" recipe.

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
- ``factory.py`` -- :func:`~lightning_pose.data.video.factory.build_video_reader`, the reader-
  selection dispatch used by ``predict_video``. See its docstring for the fallback chain and
  the recipe for adding a new backend.
"""

__all__: list[str] = []
