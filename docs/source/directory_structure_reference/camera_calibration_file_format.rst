.. _camera_calibration_file_format:

Camera calibration files
-------------------------

Each session requires a TOML file in the ``calibrations/`` directory that contains camera
parameters for all views in `Anipose <https://anipose.readthedocs.io/>`_ format.
The TOML file must include one ``[cam_N]`` section
for each camera view, where ``N`` is the camera index (0, 1, 2, etc.).

Each camera section must contain:

* ``name``: A string identifier for the camera (e.g., "cam0", "left", "front")
* ``size``: Array of two integers ``[width, height]`` specifying image dimensions in pixels
* ``matrix``: 3x3 camera intrinsic matrix as nested arrays
* ``distortions``: Array of 5 distortion coefficients ``[k1, k2, p1, p2, k3]``
* ``rotation``: Array of 3 rotation angles in radians (Rodrigues vector)
* ``translation``: Array of 3 translation values ``[x, y, z]`` in world coordinate units

Example TOML calibration file:

.. code-block:: toml

    [cam_0]
    name = "view0"
    size = [2816, 1408]
    matrix = [
        [1993.4, 0.0, 1408.0],
        [0.0, 1993.4, 704.0],
        [1451.1, 993.0, 1.0]
    ]
    distortions = [-0.121, 0.0, 0.0, 0.0, 0.0]
    rotation = [0.830, -2.001, 1.630]
    translation = [-0.001, 0.122, 1.482]

    [cam_1]
    name = "view1"
    size = [2816, 1408]
    matrix = [
        [1915.1, 0.0, 1408.0],
        [0.0, 1915.1, 704.0],
        [1585.2, 835.4, 1.0]
    ]
    distortions = [-0.057, 0.0, 0.0, 0.0, 0.0]
    rotation = [1.883, -0.765, 0.604]
    translation = [0.003, 0.089, 1.545]

    [metadata]
    # Optional metadata section for additional information

The number of camera sections must match the number of views specified in your configuration file.

Automatic calibration discovery
--------------------------------

The CLI automatically discovers calibration files from image paths without any additional
configuration. The session identifier is extracted from the frame's path: frames are
expected to live under ``labeled-data/<session>_<view>/``, and the session is everything
before the last ``_`` in that subfolder name. For example,
``labeled-data/session0_view0/frame00001.png`` yields session ``session0``.

Given the session, it looks for ``calibrations/<session>.toml`` first, then falls back to
``calibration.toml`` at the project root.