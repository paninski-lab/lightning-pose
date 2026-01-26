.. _cpu_only:

Installing the App CPU-only
=============================

It's possible to run the app without an NVIDIA GPU for labeling and viewing only.
(Model operations not supported as they require a GPU.)

Install the app
----------------

The full lightning-pose package will not install without an NVIDIA GPU,
but we can install a lightweight version that will allow the app to run.

.. code-block:: bash

   # Skips nvidia dependencies
   pip install lightning-pose --no-deps
   pip install lightning-pose-app

   # Verify your system has ffmpeg
   ffmpeg -v

Run the app
------------

Run the app as normal. Model training/inference
will not work but the labeler and viewer should work.

.. code-block:: bash

   litpose run_app


Optional: Add an existing project directory
---------------------------------------------

After installing, you may want to import a project
from an existing lightning pose installation.

1. First, :doc:`transfer the project directory <transfer_project>` to your computer.
2. Then :doc:`add the project to the projects.toml <add_existing_project>` so that it is accessible from the app.
