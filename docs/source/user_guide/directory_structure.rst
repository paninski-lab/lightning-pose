.. _directory_structure:

####################
Organizing your data
####################

The base Lightning Pose package does **not** contain tools for labeling data; please see our
`browser-based GUI <https://github.com/Lightning-Universe/Pose-app>`_
that supports the full life cycle of a pose estimation project for access to such tools.

Data directory structure
========================

Lightning Pose assumes the following project directory structure, as in the example dataset
`here <https://github.com/danbider/lightning-pose/tree/main/data/mirror-mouse-example>`_.

.. code-block::

    /path/to/project/
      ├── <LABELED_DATA_DIR>/
      ├── <VIDEO_DIR>/
      └── <YOUR_LABELED_FRAMES>.csv

* ``<YOUR_LABELED_FRAMES>.csv``: a table with keypoint labels (rows: frames; columns: keypoints).
  Note that this file can take any name, and needs to be specified in the config file under
  ``data.csv_file``.

* ``<LABELED_DATA_DIR>/``: contains images that correspond to the labels, and can include
  subdirectories.
  The directory name, any subdirectory names, and image names are all flexible, as long as they are
  consistent with the first column of `<YOUR_LABELED_FRAMES>.csv`.

* ``<VIDEO_DIR>/``: when training semi-supervised models, the videos in this directory will be used
  for computing the unsupervised losses.
  This directory can take any name, and needs to be specified in the config file under
  ``data.video_dir``.
  Notes that videos *must* be mp4 files that use the h.264 codec; see more information in the
  :ref:`FAQs<faq_video_formats>`.

Converting DLC projects to Lightning Pose format
================================================

Once you have installed Lightning Pose, you can convert previous DLC projects into the proper
Lightning Pose format by running the following script from the command line
(make sure to activate the conda environment):

.. code-block:: console

    python scripts/converters/dlc2lp.py --dlc_dir=/path/to/dlc_dir --lp_dir=/path/to/lp_dir

.. Note::

    This script is only available through the :ref:`conda from source <conda_from_source>` installation method.

That's it!
After this you will need to update your config file with the correct paths (see next page).

Converting other projects to Lightning Pose format
==================================================
Coming soon. If you have labeled data from other pose estimation packages (like SLEAP or DPK) and
would like to try out Lightning Pose, please
`raise an issue <https://github.com/danbider/lightning-pose/issues>`_.
