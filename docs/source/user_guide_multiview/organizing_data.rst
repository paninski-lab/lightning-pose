.. _organizing_multiview_data:

####################
Organizing your data
####################

Organizing your data for multi-view models follows a similar structure to the single-view data
organization.

For a comprehensive reference on the required directory structure and file formats, see the
:doc:`/source/directory_structure_reference/multiview_structure`.

As an example, let's assume a dataset has two camera views from a given session ("session0"),
which we'll call "view0" and "view1".
Lightning Pose assumes the following project directory structure:

.. code-block::

    /path/to/project/
      ├── project.yaml
      ├── <LABELED_DATA_DIR>/
      │   ├── session0_view0/
      │   └── session0_view1/
      ├── <VIDEO_DIR>/
      │   ├── session0_view0.mp4
      │   └── session0_view1.mp4
      ├── view0.csv
      └── view1.csv

* ``project.yaml``: contains project-level metadata. See :doc:`/source/directory_structure_reference/project_yaml_file_format`.

* ``<LABELED_DATA_DIR>/``: The directory name, any subdirectory names, and image names are all flexible, as long as they are consistent with the first column of `<view_name>.csv` files (see below). As an example, each session/view pair can have its own subdirectory, which contains images that correspond to the labels. The same frames from all the views must have the same names; for example, the images corresponding to time point 39 should be named "<LABELED_DATA_DIR>/session0_view0/img000039.png" and "<LABELED_DATA_DIR>/session0_view1/img000039.png".

* ``<VIDEO_DIR>/``: This is a single directory of videos, which **must** follow the naming convention ``<session_name>_<view_name>.mp4``. So in our example there should be two videos, named ``session0_view0.mp4`` and ``session0_view1.mp4``.

* ``<view_name>.csv``: For each view (camera) there should be a table with keypoint labels (rows: frames; columns: keypoints). Note that these files can take any name, and need to be listed in the config file under the ``data.csv_file`` section. Each csv file must contain the same set of keypoints, and each must have the same number of rows (corresponding to specific points in time). For more details on the format, see :doc:`/source/directory_structure_reference/label_csv_file_format`.

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

Your project directory must include calibration information. See the 
:doc:`/source/directory_structure_reference/camera_calibration_file_format` for detailed 
information on the file format and structure, including how to use the ``calibrations.csv`` 
index file.

The following structure is typically used:

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

Bounding boxes
==============

When working with small animals in large arenas, you may have already performed rough tracking
and cropped the animal out of the larger image. In this scenario, Lightning Pose needs to know
the bounding box coordinates for each labeled frame to properly apply 3D augmentations and
loss functions.

See the :doc:`/source/directory_structure_reference/bounding_box_file_format` for detailed 
specifications.

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