.. _sv_organizing_data:

####################
Organizing your data
####################

Data directory structure
========================

.. note::

    The Lightning Pose CLI historically supported flexible data organization, as written in this document.
    The Lightning Pose App requires a more constrained directory structure.

    See the :doc:`/source/directory_structure_reference/singleview_structure` for the detailed reference of the expected directory structure
    that is fully app-compatible.

Lightning Pose assumes the following project directory structure, as in the example dataset
`here <https://github.com/paninski-lab/lightning-pose/tree/main/data/mirror-mouse-example>`_.

.. code-block::

    /path/to/project/
      ├── project.yaml
      ├── <LABELED_DATA_DIR>/
      ├── <VIDEO_DIR>/
      └── <YOUR_LABELED_FRAMES>.csv

* ``project.yaml``: contains project-level metadata. See :doc:`/source/directory_structure_reference/project_yaml_file_format`.

* ``<YOUR_LABELED_FRAMES>.csv``: a table with keypoint labels (rows: frames; columns: keypoints).
  Note that this file can take any name, and needs to be specified in the config file under
  ``data.csv_file``. For more details on the format, see :doc:`/source/directory_structure_reference/label_csv_file_format`.

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
