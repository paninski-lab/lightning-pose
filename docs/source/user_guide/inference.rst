.. _inference:

#########
Inference
#########

Once you have trained a model you'll likely want to run inference on new videos.

Similar to training, there are several tools for running inference:

#. A set of high-level functions used for processing videos and creating labeled clips. You can combine these to create your own custom inference script. This is required if you used the :ref:`pip package <pip_package>` installation method.
#. An example inference script provided in the :ref:`conda from source <conda_from_source>` installation method. This demonstrates how to combine the high-level functions.

.. note::

    The steps below assume the :ref:`conda from source <conda_from_source>` installation method.
    If you did not use this installation method, see the
    `example inference script <https://github.com/danbider/lightning-pose/blob/main/scripts/predict_new_vids.py>`_.
    You can also see how video inference is handled in the
    `example train script <https://github.com/danbider/lightning-pose/blob/main/scripts/train_hydra.py>`_.

Inference with example data
===========================

To run inference with a model trained on the example dataset, run the following command from
inside the ``lightning-pose`` directory
(make sure you have activated your conda environment):

.. code-block:: console

    python scripts/predict_new_vids.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"]

This overwrites the config field ``eval.hydra_paths``, which is a list that contains the relative
paths of the model folders you want to run inference with
(you will need to replace "YYYY-MM-DD/HH-MM-SS/" with the timestamp of your own model).

Inference with your data
========================

In order to use this script more generally, you need to update several config fields:

#. ``eval.hydra_paths``: path to models to use for prediction
#. ``eval.test_videos_directory``: path to a `directory` containing videos to run inference on
#. ``eval.save_vids_after_training``: if ``true``, the script will also save a copy of the full video with model predictions overlaid.

The results will be stored in the model directory.

As with training, you either directly edit your config file and run:

.. code-block:: console

    python scripts/predict_new_vids.py --config-path=<PATH/TO/YOUR/CONFIGS/DIR> --config-name=<CONFIG_NAME.yaml>

or override these arguments in the command line:

.. code-block:: console

    python scripts/predict_new_vids.py --config-path=<PATH/TO/YOUR/CONFIGS/DIR> --config-name=<CONFIG_NAME.yaml> eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"] eval.test_videos_directory=/absolute/path/to/videos

.. note::

  Videos *must* be mp4 files that use the h.264 codec; see more information in the
  :ref:`FAQs<faq_video_formats>`.
