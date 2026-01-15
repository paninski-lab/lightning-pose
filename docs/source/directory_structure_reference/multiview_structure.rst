======================================
Multiview Data Directory Structure
======================================

The detailed documentation for a multiview project data directory.

**1. Base Structure (Required)**

.. code-block:: text

    /path/to/project/
      ├── project.yaml ......................................[1]
      ├── labeled-data/ .....................................[2]
      │   ├── session0_view0/
      │   │   └── frame000001.png
      │   └── session0_view1/
      │       └── frame000001.png
      ├── videos/ ...........................................[3]
      │   ├── session0_view0.mp4
      │   └── session0_view1.mp4
      ├── CollectedData_view0.csv ...........................[4]
      └── CollectedData_view1.csv
      ├── CollectedData_view0.unlabeled .....................[5]
      └── CollectedData_view1.unlabeled

**2. Camera Calibrations (Optional)**

.. code-block:: text

   /path/to/project/
     ├── calibration.toml ...................................[6]
     └── calibrations/ ......................................[7]
         ├── session0.toml
         ├── session1.toml
         └── ...

**3. Bounding Boxes (Optional)**

.. code-block:: text

   /path/to/project/ ........................................[8]
     ├── bboxes_view0.csv
     └── bboxes_view1.csv

----

Detailed Requirements
---------------------

.. _project_yaml_req:

project.yaml file [1]
~~~~~~~~~~~~~~~~~~~~~~
Required for App use. Storage of App project settings. See :doc:`project_yaml_file_format`

.. _labeled_data_req:

Labeled Data folder [2]
~~~~~~~~~~~~~~~~~~~~~~~~

This folder contains extracted frames for labeling and training.
Session and View information embedded in the paths is
used to power multiview features.

In addition to the primary frame being extracted, the 2 frames to the left and right of
are also extracted. These are used when training context
models and for future labeler features that use context information.

Frame indices are zero-padded to have a total of 8 digits.

.. _videos_req:

Video Files [3]
~~~~~~~~~~~~~~~~

Storage for imported video files, stored as ``<SessionKey>_<View>.mp4``.

Video files are transcoded using ``libx264 yuv420p``. For the App viewer
to be frame-accurate, they are also encoded using an Intra frame for every frame
(Group of Pictures size 1).

Used as the default source of videos for unsupervised training,
and the source of videos in the App viewer.

.. _label_csv_file_req:

Label CSV Files [4]
~~~~~~~~~~~~~~~~~~~~

A single multiview label file in the app is represented by multiple singleview label files on disk.
The files are aligned across views: the Nth row in each file is for the same frame across cameras.

The naming convention is ``<LabelFileKey>_<View>.csv`` where a LabelFileKey of "CollectedData"
signifies that the default label file for training.

See `<label_csv_file_format>`_

.. _unlabeled_sidecar_file_req:

Unlabeled Sidecar Files [5]
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These files contain the unlabeled frame queue for the App.
The files are aligned across views: the Nth row in each file is for the same frame across cameras.
They are newline-separated lists of paths.

For example, ``CollectedData_view0.unlabeled`` contains:

.. code-block::

    labeled-data/session0_view0/frame00000001.png
    labeled-data/session0_view0/frame00000123.png
    ...

.. _project_calib_req:

Project-level Camera Calibration File [6]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used as a fallback when the session-level calibration file
is not found.

See `<camera_calibration_file_format>`_.


.. _session_calib_req:

Session-level Camera Calibration Files [7]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `<camera_calibration_file_format>`_.

* Used for App Labeler Multiview features: triangulation/reprojection and bundle adjustment
* Used for CLI 3d losses and augmentation features

For the CLI, it follows the frame-calibration mapping according to the config field ``camera_params_file``.
The App does not use this, instead it maps from Session to Calibration using the directory structure.

.. _bbox_req:

Bounding Boxes [7]
~~~~~~~~~~~~~~~~~~
The bounding box coordinates for each labeled frame to properly apply 3D augmentations and
loss functions.

See `<bounding_box_file_format>`_.
