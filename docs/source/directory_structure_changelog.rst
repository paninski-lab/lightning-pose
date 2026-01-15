==============================
Directory Structure Changelog
==============================

This document details the schema changes to the directory structure
over time.

.. contents:: Versions
   :local:
   :depth: 2

v2.0.5-app-compatible | Jan 14, 2026
------------------------------------

v2.0.5 includes a new app. The App-compatible directory structure is a subset of
what's supported when using just the command-line. The main differences are documented here.

Over time, the App and CLI directory structures will converge towards the App's requirements.
Directory structures that are only supported via the CLI will be deprecated.

Data directory
~~~~~~~~~~~~~~~

1. A ``project.yaml`` file in data directory is required. Without this, a project is CLI-only.

2. The default label file should be named ``CollectedData_<View>.csv`` (``CollectedData.csv`` for singleview).
The general format required by the app is ``<LabelFileKey>_<View>.csv``.
``<View>.csv`` without a LabelFileKey is CLI-only..

3. Multiview calibration files should be stored as ``calibrations/<SessionKey>.toml`` where SessionKey is
matches the video files ``<SessionKey>_<View>.mp4``. In case a calibration file is not found,
the app will fall back on the project-level calibration file, ``DATA_DIR/calibration.toml``.

Alternative file paths for calibration files are CLI-only.

Model directory
~~~~~~~~~~~~~~~~

1. The models directory should immediately contain model directories, instead of
subdirectories used for organization. Specifically, you should move away from
the default depth of 2 structure generated the CLI when --output_dir is unspecified.
For now, maximum depth of 2 structure is supported but discouraged in the App.

Example of depth of 2 structure:

.. code-block::

    /models-directory/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── config.yaml
            ├── tb_logs/
            └── ...

Preferred structure:

.. code-block::

    /models-directory/
    └── ModelName/
        ├── config.yaml
        ├── tb_logs/
        └── ...

.. note::
    Migrating from the old Pose-app requires further changes.
    See the :doc:`migrating_to_app`.