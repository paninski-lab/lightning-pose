.. _training:

########
Training
########

Lightning Pose provides several tools for training models:

#. A set of high-level functions used for creating data loaders, models, trainers, etc. You can combine these to create your own custom training script. This is required if you used the :ref:`pip package <pip_package>` installation method.
#. An example training script provided in the :ref:`conda from source <conda_from_source>` installation method. This demonstrates how to combine the high-level functions for model training and evaluation.

.. note::

    The steps below assume the :ref:`conda from source <conda_from_source>` installation method.
    If you did not use this installation method, see the
    `example training script <https://github.com/danbider/lightning-pose/blob/main/scripts/train_hydra.py>`_.


Train with example data
=======================

To train a model on the example dataset provided with the Lightning Pose package,
run the following command from inside the ``lightning-pose`` directory
(make sure you have activated your conda environment):

.. code-block:: console

    python scripts/train_hydra.py

Note there are no arguments - this tells the script to default to the example data.

Train with your data
====================

To train a model on your own dataset, follow these steps:

#. Ensure your data is in the :ref:`proper data format <directory_structure>`.
#. Copy the file ``scripts/configs/config_default.yaml`` to another directory and rename it. You will then need to update the various fields to match your dataset (see :ref:`The configuration file <config_file>` section). See other config files in ``scripts/configs/`` for examples.
#. Train your model from the terminal and overwrite the config path and config name with your newly created file:

   .. code-block:: console

       python scripts/train_hydra.py --config-path=<PATH/TO/YOUR/CONFIGS/DIR> --config-name=<CONFIG_NAME.yaml>

You can find more information on the structure of the output model directory
:ref:`below <model_directory_structure>`.

Working with ``hydra``
======================

All of the scripts in the ``scripts`` directory rely on the ``hydra`` package to manage
arguments in config files.
You have two options: directly edit the config file, or override it from the command line.

#. **Edit** the config file, and save it.
   Then run the script without arguments:

   .. code-block:: console

       python scripts/train_hydra.py

#. **Override** the argument from the command line; for example, if you want to use a maximum of 11
   epochs instead of the default number (not recommended):

   .. code-block:: console

       python scripts/train_hydra.py training.max_epochs=11

   Or, for your own dataset,

   .. code-block::

       python scripts/train_hydra.py --config-path=<PATH/TO/YOUR/CONFIGS/DIR> --config-name=<CONFIG_NAME.yaml> training.max_epochs=11

We also recommend trying out training with resizing to smaller images first;
this allows for larger batch sizes/fewer Out Of Memory errors on the GPU:

.. code-block:: console

    python scripts/train_hydra.py --config-path=<PATH/TO/YOUR/CONFIGS/DIR> --config-name=<CONFIG_NAME.yaml> data.image_resize_dims.height=256 data.image_resize_dims.width=256

See more documentation on the config file fields :ref:`here <config_file>`. A couple of fields that
are specific to the provided training script, but important to consider:

* ``eval.predict_vids_after_training``: if ``true``, automatically run inference after training on all videos located in the directory given by ``eval.test_videos_directory``; results are saved to the model directory
* ``eval.save_vids_after_training``: if ``true`` (as well as ``eval.predict_vids_after_training``) the keypoints predicted during the inference step will be overlaid on the videos and saved with inference outputs to the model directory

Tensorboard
===========

The outputs of the training script, namely the model checkpoints and tensorboard logs,
will be saved in the ``lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS/tb_logs`` directory by default.
(Note: this behavior can be changed by updating ``hydra.run.dir`` in the config file to an
absolute path of your choosing.)

To view the logged losses with tensorboard in your browser, in the command line, run:

.. code-block:: console

    tensorboard --logdir outputs/YYYY-MM-DD/

where you use the date in which you ran the model.
Click on the provided link in the terminal, which will look something like
``http://localhost:6006/``.
Note that if you save the model at a different directory, just use that directory after
``--logdir``.

.. note::

    If you don't see all your models in tensorboard,
    hit the refresh button on the top right corner of the screen,
    and the other models should appear.

Metrics are plotted as a function of step/batch. Validation metrics are typically recorded less
frequently than train metrics.
The frequency of these checks are controlled by ``cfg.training.log_every_n_steps`` (training)
and ``cfg.training.check_val_every_n_epoch`` (validation).

**Available metrics**

The following are the important metrics for all model types
(supervised, context, semi-supervised, etc.):

* ``train_supervised_loss``: this is the same as ``train_heatmap_mse_loss_weighted``, which is the
  mean square error (MSE) between the true and predicted heatmaps on labeled training data
* ``train_supervised_rmse``: the root mean square error (RMSE) between the true and predicted
  (x, y) coordinates on labeled training data; scale is in pixels
* ``val_supervised_loss``: this is the same as ``val_heatmap_mse_loss_weighted``, which is the
  MSE between the true and predicted heatmaps on labeled validation data
* ``val_supervised_rmse``: the RMSE between the true and predicted (x, y) coordinates on labeled
  validation data; scale is in pixels

The following are important metrics for the semi-supervised models:

* ``train_pca_multiview_loss_weighted``: the ``train_pca_multiview_loss`` (in pixels), which
  measures multiview consistency, multplied by the loss weight set in the configuration file.
  This metric is only computed on batches of unlabeled training data.
* ``train_pca_singleview_loss_weighted``: the ``train_pca_singleview_loss`` (in pixels), which
  measures pose plausibility, multplied by the loss weight set in the configuration file.
  This metric is only computed on batches of unlabeled training data.
* ``train_temporal_loss_weighted``: the ``train_temporal_loss`` (in pixels), which
  measures temporal smoothness, multplied by the loss weight set in the configuration file.
  This metric is only computed on batches of unlabeled training data.
* ``total_unsupervised_importance``: a weight on all *weighted* unsupervised losses that linearly
  increases from 0 to 1 over 100 epochs
* ``total_loss``: weighted supervised loss (``train_heatmap_mse_loss_weighted``) plus
  ``total_unsupervised_importance`` times the sum of all applicable weighted unsupervised losses


.. _model_directory_structure:

Model directory structure
=========================

If you train a model using our script ``lightning-pose/scripts/train_hydra.py``,
a directory will be created with the following structure.
The default is to save models in a directory called ``outputs`` inside the Lightning Pose
directory; to change this, update the config fields ``hydra.run.dir`` and ``hydra.sweep.dir``
with absolute paths of your choosing.

.. code-block::

    /path/to/models/YYYY-MM-DD/HH-MM-SS/
      ├── tb_logs/
      ├── video_preds/
      │   └── labeled_videos/
      ├── config.yaml
      ├── predictions.csv
      ├── predictions_pca_multiview_error.csv
      ├── predictions_pca_singleview_error.csv
      └── predictions_pixel_error.csv

* ``tb_logs/``: model weights

* ``video_preds/``: predictions and metrics from videos. The config field ``eval.test_videos_directory`` points to a directory of videos; if ``eval.predict_vids_after_training`` is set to ``true``, all videos in the indicated direcotry will be run through the model upon training completion and results stored here.

* ``video_preds/labeled_videos/``: labeled mp4s. The config field ``eval.test_videos_directory`` points to a directory of videos; if ``eval.save_vids_after_training`` is set to ``true``, all videos in the indicated direcotry will be run through the model upon training completion and results stored here.

* ``predictions.csv``: predictions on labeled data. The right-most column records the train/val/test split that each example belongs to.

* ``predictions_pixel_error.csv``: Euclidean distance between the predictions in ``predictions.csv`` and the labeled keypoints (in ``<YOUR_LABELED_FRAMES>.csv``) per keypoint and frame.

We also compute all unsupervised losses, where applicable, and store them
(per keypoint and frame) in the following csvs:

* ``predictions_pca_multiview_error.csv``: pca multiview reprojection error between predictions and labeled keypoints

* ``predictions_pca_singleview_error.csv``: pca singleview reprojection error between predictions and labeled keypoints
