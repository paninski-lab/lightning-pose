.. _example_dataset

Try an example dataset
=======================

You can download a project that has already been
setup with Labels, a couple of small models, and Predictions.

Extract to a directory and add that directory to ``~/.lightning-pose/projects.toml``
For example:

.. code-block:: toml

  [fly-anipose]
  data_dir = "/home/username/Downloads/fly-anipose/"

After this, the app will recognize the project and allow you to test
all of its capabilities.

.. code-block:: console

    litpose run_app

The server is now running.
Open a browser and navigate to ``http://localhost:8000`` if the server is local,
or navigate to the cloud URL for port 8000.

