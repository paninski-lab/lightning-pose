.. _fiftyone:

########
FiftyOne
########

The `FiftyOne app <https://voxel51.com/>`_ visualizes the predictions of one or multiple
trained models, overlaid on labeled frames.

Creating ``FiftyOne.Dataset`` for predictions
=============================================

The first step is to create a ``FiftyOne.Dataset`` object
(i.e., a Mongo database pointing to images, keypoint predictions, names, and confidences).
Run the following command from inside the ``lightning-pose`` directory
(make sure you have activated your conda environment):

.. code-block:: console

    python scripts/create_fiftyone_dataset.py \
    --config-path=<PATH/TO/YOUR/CONFIGS/DIR> \
    --config-name=<CONFIG_NAME.yaml> \
    eval.fiftyone.dataset_name=<YOUR_DATASET_NAME> \
    eval.hydra_paths=["</ABSOLUTE/PATH/TO/HYDRA/DIR/1>", "</ABSOLUTE/PATH/TO/HYDRA/DIR/1>"] \
    eval.fiftyone.model_display_names=["<NAME_FOR_MODEL_1>","<NAME_FOR_MODEL_2>"] \
    eval.fiftyone.launch_app_from_script=true

* ``config-path/config-name``: these are used the same as the training and inference scripts
* ``eval.fiftyone.dataset_name``: unique name of ``FiftyOne.Dataset`` object
* ``eval.hydra_paths``: list of *absolute* paths to directories of the trained models you want to use for prediction. Each directory should contain a ``predictions.csv`` file.

  You can also use the relative form

  .. code-block:: console

      eval.hydra_paths: ["YYYY-MM-DD/HH-MM-SS/", "YYYY-MM-DD/HH-MM-SS/"]

  which will look in the ``lightning-pose/outputs`` directory for these subdirectories.

* ``eval.fiftyone.model_display_names``: meaningful display names for the models above, e.g.

  .. code-block:: console

      eval.fiftyone.model_display_names: ["supervised", "semi-supervised"]

* ``eval.fiftyone.launch_app_from_script``: if ``true``, the ``FiftyOne`` app will launch after
  dataset creation. To open the app, follow the link provided in the terminal; it should look
  something like

  .. code-block:: console

      http://localhost:5151/

.. note::

    These arguments can also be edited and saved in the config files if needed.

Launching the FiftyOne app from ipython
=======================================

You can access previously created ``FiftyOne.Dataset`` objects from the terminal.

Open an ``ipython`` session from your terminal:

.. code-block:: console

    ipython

Inside the interactive python session, we import fiftyone, load the dataset we created,
and launch the app:

.. code-block:: python

    import fiftyone as fo
    dataset = fo.load_dataset("<YOUR_DATASET_NAME>")
    session = fo.launch_app(dataset)

The app should automatically open in your browser.

If you have forgotten your dataset names, you can list them in ipython:

.. code-block:: python

    fo.list_datasets()
