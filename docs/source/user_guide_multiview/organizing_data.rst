.. _organizing_multiview_data:

####################
Organizing your data
####################

Organizing your data for multi-view models follows a similar structure to the single-view data
organization.

As an example, let's assume a dataset has two camera views from a given session ("session0"),
which we'll call "view0" and "view1".
Lightning Pose assumes the following project directory structure:

.. code-block::

    /path/to/project/
      ├── <LABELED_DATA_DIR>/
      │   ├── session0_view0/
      │   └── session0_view1/
      ├── <VIDEO_DIR>/
      │   ├── session0_view0.mp4
      │   └── session0_view1.mp4
      ├── view0.csv
      └── view1.csv

* ``<LABELED_DATA_DIR>/``: The directory name, any subdirectory names, and image names are all flexible, as long as they are consistent with the first column of `<view_name>.csv` files (see below). As an example, each session/view pair can have its own subdirectory, which contains images that correspond to the labels. The same frames from all the views must have the same names; for example, the images corresponding to time point 39 should be named "<LABELED_DATA_DIR>/session0_view0/img000039.png" and "<LABELED_DATA_DIR>/session0_view1/img000039.png".

* ``<VIDEO_DIR>/``: This is a single directory of videos, which **must** following the naming convention ``<session_name>_<view_name>.csv``. So in our example there should be two videos, named ``session0_view0.mp4`` and ``session0_view1.mp4``.

* ``<view_name>.csv``: For each view (camera) there should be a table with keypoint labels (rows: frames; columns: keypoints). Note that these files can take any name, and need to be listed in the config file under the ``data.csv_file`` section. Each csv file must contain the same set of keypoints, and each must have the same number of rows (corresponding to specific points in time).


The configuration file
=======================

Like the single-view case, users interact with Lighting Pose through a single configuration file.
This file points to data directories, defines the type of models to fit, and specifies a wide range
of hyperparameters.

A template file can be found
`here <https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default_multiview.yaml>`_.
When training a model on a new dataset, you must copy/paste this template onto your local machine
and update the arguments to match your data.

To switch to multi-view from single-view you need to change two ``data`` fields.
Again, assume that we are working with the two-view dataset used as an example above:

.. code-block:: yaml

    data:
      csv_file:
        - view0.csv
        - view1.csv
      view_names:
        - view0
        - view1


* ``csv_file``: list of csv filenames for each view
* ``view_names``: list view names

.. _camera_calibration:

Camera calibration
==================

When using the 3D loss described later, multi-view Lightning Pose requires camera calibration
information to understand the geometric relationships between different camera views.
This calibration data is stored in a specific format using TOML files that follow the Anipose convention.

Required calibration files
--------------------------

Your project directory must include two additional components for camera calibration:

.. code-block::

    /path/to/project/
      ├── <LABELED_DATA_DIR>/
      │   ├── session0_view0/
      │   └── session0_view1/
      ├── <VIDEO_DIR>/
      │   ├── session0_view0.mp4
      │   └── session0_view1.mp4
      ├── view0.csv
      ├── view1.csv
      ├── calibrations.csv          # NEW: calibration index file
      └── calibrations/             # NEW: directory with TOML files
          ├── session0.toml
          ├── session1.toml
          └── ...

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

TOML calibration files
----------------------

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

Bounding boxes
==============

When working with small animals in large arenas, you may have already performed rough tracking 
and cropped the animal out of the larger image. In this scenario, Lightning Pose needs to know 
the bounding box coordinates for each labeled frame to properly apply 3D augmentations and 
loss functions.

Bounding box files are provided in the top-level project directory with the naming convention 
``bboxes_<view_name>.csv``, where ``<view_name>`` matches the view names specified in your 
configuration file.

Your project directory structure with bounding boxes:

.. code-block::

    /path/to/project/
      ├── <LABELED_DATA_DIR>/
      │   ├── session0_view0/
      │   └── session0_view1/
      ├── <VIDEO_DIR>/
      │   ├── session0_view0.mp4
      │   └── session0_view1.mp4
      ├── view0.csv
      ├── view1.csv
      ├── bboxes_view0.csv          # NEW: bounding boxes for view0
      ├── bboxes_view1.csv          # NEW: bounding boxes for view1
      ├── calibrations.csv
      └── calibrations/
          ├── session0.toml
          └── ...

.. note::

    Bounding boxes are only used for 3D augmentation and loss; the Lightning Pose prediction CSV
    files will have (x, y) values with respect to the images fed into the model. You will need to
    manually combine these values with the bounding box information to obtain predictions in the
    original frame coordinates.

Bounding box file format
------------------------

Each bounding box CSV file must have exactly five columns:

* **First column** (no header): The relative path to each labeled image file
* **x**: Upper-left x-coordinate of the bounding box
* **y**: Upper-left y-coordinate of the bounding box  
* **h**: Height of the bounding box
* **w**: Width of the bounding box

Example ``bboxes_view0.csv`` format:

.. code-block::

    ,x,y,h,w
    labeled-data/session0_view0/img00000005.png,1230,117,391,391
    labeled-data/session0_view0/img00000010.png,482,138,425,425
    labeled-data/session0_view0/img00000230.png,1230,117,391,391
    labeled-data/session1_view0/img00000151.png,625,125,405,405
    labeled-data/session1_view0/img00000201.png,1186,118,343,344

The image paths in the first column should match exactly with the paths used in your 
labeled data CSV files.

Configuration
-------------

You must specify the bounding box files in your configuration file:

.. code-block:: yaml

    data:
      bbox_file:
        - bboxes_view0.csv
        - bboxes_view1.csv

The order of bounding box files must match the order of your ``csv_file`` and ``view_names`` 
entries. If you do not have bounding boxes for your data, simply omit the ``bbox_file`` 
field entirely from your configuration file.