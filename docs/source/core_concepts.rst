Core concepts
==============

Before moving on to using Lightning Pose, its worth pausing to learn the core components of the system.

The two interfaces: The App and the CLI
----------------------------------------

At the heart of LP is the ``lightning_pose`` python package. Built on top of this are two end-user interfaces: ``litpose``, the command-line interface (CLI), and the App, which is run via ``litpose run_app``.

**How to choose**

* **Use the App** for labeling data, training models, or viewing predictions to evaluate models.
* **Use the CLI** for custom pipelines or more advanced features not available in the app.

The App and CLI are interoperable via working on a common Project.

The Project and its directories
---------------------------------

A Project contains all data related to a pose estimation project in Lightning Pose.
It consists of two directories:

1. **Data directory**: Contains a copy of all data needed to train models.
2. **Model directory**: Contains model weights, metadata needed to run inference, and is the output directory for model predictions.

The model directory is usually just a subdirectory of the data directory, "models". For this reason,
the data directory is the primary project directory.

The ``~/.lightning-pose/projects.toml`` file contains an index of all projects.

For example, this is a fairly typical ``~/.lightning-pose/projects.toml`` file:

.. code-block:: toml

    [my_project]
    data_dir = "/home/username/LPProjects/data"
    # model_dir omitted, defaults to "/home/username/LPProjects/models"

.. note::
  If you move the project directories, you should update the ``projects.toml`` file with their new locations.

The structure of the data and model directories is documented in Reference: Directory Structure.

Important: Naming video files for Lightning Pose
--------------------------------------------------

Video file naming is important. **When you upload files to Lightning Pose,
we only get the filename, so it's important that it fully and uniquely describe the
video being uploaded.**

A simple and effective starting point is:

.. code-block::

    SUBJECT_DATE_TIME_VIEW.mp4

Plan for change
~~~~~~~~~~~~~~~~

You should anticipate any changes you might need to make and plan for them now.
For example, if you anticipate cropping or clipping, you might
say to yourself that you'll embed more metadata like:

.. code-block::

    SUBJECT_DATE_TIME_CLIP_T1-T2_VIEW.mp4

If you have multiple methodologies for cropping, you might further extend
the naming convention like so:

.. code-block::

    SUBJECT_DATE_TIME_CLIP_T1-T2_CROP_cropper_model_1_VIEW.mp4

View name must always be the last token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In multiview projects, we always parse a video filename as SESSION_VIEW.mp4.
View must be the last token in your naming scheme, and must not contain underscores.

Use valid characters
~~~~~~~~~~~~~~~~~~~~~~
Stick with filesystem and URL-safe characters: alphanumeric, -, _.

Use a uniform naming scheme throughout the app
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Whether you upload videos in the Labeler for extracting frames for labeling,
or uploading videos for inference, we store videos in the same directory.
All videos are available to view in the Viewer. Use a uniform naming scheme
across all these tasks, so that names don't conflict in the
project directory, and so that you can easily identify videos and related
files in the Viewer and project directories.
