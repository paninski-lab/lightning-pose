.. _multiview_separate:

################################
Multiview: separate data streams
################################

In addition to the mirrored setups discussed on the previous page, Lightning Pose also supports
more traditional multiview data, where the same scene is captured from different angles with
different cameras.
Each view is treated as an independent input to a single network.
This way, the network can learn from different perspectives and be agnostic to the correlations
between the different views.
Similar to the single view setup, Lightning Pose produces a separate csv file with the predicted
keypoints for each video

.. note::

    As of January 2024, the non-mirrored multiview feature of Lightning Pose does not support
    context frames or unsupervised losses.

Organizing your data
====================

As an example, let’s assume a dataset has two camera views, which we’ll call “view0” and “view1”.
Lightning Pose assumes the following project directory structure:

.. code-block::

    /path/to/project/
      ├── <LABELED_DATA_DIR>/
      │   ├── view0/
      │   └── view1/
      ├── <VIDEO_DIR>/
      ├── view0.csv
      └── view1.csv

* ``<LABELED_DATA_DIR>/``: Each view must have a dedicated folder that contains images that correspond to the labels. The same frames from all the views must have the same names; for example, the images corresponding to time point 39 must be named "<LABELED_DATA_DIR>/view0/img000039.png" and "<LABELED_DATA_DIR>/view1/img000039.png". The directory name, any subdirectory names, and image names are all flexible, as long as they are consistent with the first column of `<view_name>.csv` files.

* ``<view_name>.csv``: For each view (camera) there should be a table with keypoint labels (rows: frames; columns: keypoints). Note that these files can take any name, and need to be listed in the config file under the ``data.csv_file`` section. Each csv file must contain the same set of keypoints, and each must have the same number of rows (corresponding to specific points in time).

* ``<VIDEO_DIR>/``: contains all the videos from any/all views that you would like to be predicted after model training completes.

The configuration file
======================

Like the single view case, users interact with Lighting Pose through a single configuration file.
This file points to data directories, defines the type of models to fit, and specifies a wide range
of hyperparameters.

A template file can be found
`here <https://github.com/danbider/lightning-pose/blob/main/scripts/configs/config_default.yaml>`_.
When training a model on a new dataset, you must copy/paste this template onto your local machine
and update the arguments to match your data.

To switch to multiview from single view you need to change two data parameters.
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

Training and inference
======================

Once the data are properly organized and the config files updated, :ref:`training <training>` and
:ref:`inference <inference>` in this multiview setup proceed exactly the same as for the single
view case.
Because the trained network is view-agnostic,
during inference videos are processed and saved one view at a time.
