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

Calibrations index file
-----------------------

The ``calibrations.csv`` file maps each labeled image to its corresponding calibration file.
This file must have exactly two columns:

* **First column** (no header): The relative path to each labeled image, **without view-specific subdirectories**. This should match the image paths that appear in your labeled data CSV files, but with any view-specific path components removed.

* **Second column** (``file`` header): The relative path to the TOML calibration file for that session.

Example ``calibrations.csv`` format:

.. code-block::

    ,file
    labeled-data/session0/img00000005.png,calibrations/session0.toml
    labeled-data/session0/img00000010.png,calibrations/session0.toml
    labeled-data/session0/img00000230.png,calibrations/session0.toml
    labeled-data/session1/img00000151.png,calibrations/session1.toml
    labeled-data/session1/img00000201.png,calibrations/session1.toml

Note that the first column uses the session name (e.g., ``session0``) rather than the
view-specific directory names (e.g., ``session0_view0``, ``session0_view1``).

You will also need to add the location of this file to your configuration file in order to use
the 3D loss:

.. code-block:: yaml

    data:
      camera_params_file: /path/to/project/calibrations.csv