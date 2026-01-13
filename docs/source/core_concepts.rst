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

Key properties of a project
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project directories are designed to be:

1. Portable. You can copy them across machines.
2. Highly structured, according to a documented schema.
3. Modular. You can created a copy a subset of the directory for specific purposes (like inference on a cluster).

The  ``~/.lightning-pose/projects.toml`` file is **not** portable since it contains absolute paths.

