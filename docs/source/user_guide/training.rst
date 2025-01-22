.. _training:

########
Training
########

If this is your first time training a model, you'll need to:

#. Organize your data (per the document in :ref:`directory_structure`)
#. Create a valid config file

After that, you are ready to train a model.

Create a valid config file
==========================

Copy the default config (`config_default.yaml`_)
to a local file and modify the ``data`` section to point to your own dataset. For example:

.. code-block:: yaml

   data:
      image_resize_dims:
        height: 256
        width: 256
      data_dir: /home/user1/data/
      video_dir: /home/user1/data/videos
      csv_file: labeled_frames.csv
      downsample_factor: 2
      # total number of keypoints
      num_keypoints: 3
      keypoint_names:
          - paw_left
          - paw_right
          - nose_tip

.. _config_default.yaml: https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml

Sections other than ``data`` have reasonable defaults for getting started,
but can be modified as well. For the full reference of fields, see :ref:`config_file`.

Train a model
=============

Since version 1.7.0, installing lightning-pose also installs ``litpose``,
a command-line tool built on top of the :ref:`lightning_pose_api`.

To train a model, just point ``litpose train`` at your config file:

.. code-block:: shell

  # Replace 'config_default.yaml' with the path to your config file.
  litpose train config_default.yaml

The model will be saved in ``./outputs/{YYYY-MM-DD}/{HH:MM:SS}/``, creating the folder if it does not already exist.
To customize the output directory, use the ``--output_dir OUTPUT_DIR`` flag of the command.

.. code-block:: shell

  # Save to 'outputs/lp_test_1'
  litpose train config_default.yaml --output_dir outputs/lp_test_1

.. note::

    If the command ``litpose`` is not found, ensure that you've activated the conda
    environment with lightning-pose installed, and that you're using version >= 1.7.0
    (verify this using ``pip show lightning-pose``).

For the full listing of training options, run ``litpose train --help``.

Config overrides
----------------

If you want to override some config values before training, you can use the ``--overrides`` flag.
This uses hydra under the hood, so refer to the `hydra syntax for config overrides`_.

.. _hydra syntax for config overrides: https://hydra.cc/docs/advanced/override_grammar/basic/

.. code-block:: shell

  # Train for only 5 epochs
  litpose train config_default.yaml --overrides training.min_epochs=5 training.max_epochs=5

  # Train a supervised model
  litpose train config_default.yaml --output_dir outputs/supervised --overrides \
    model.losses_to_use=null

Post-training flags
-------------------

After training, lightning pose can automatically predict on some videos
and save out videos labeled with its predictions. The config settings that control this behavior are:

* ``eval.predict_vids_after_training``: if ``true``, automatically run inference after training on
  all videos located in the directory given by ``eval.test_videos_directory``; results are saved
  to the model directory
* ``eval.save_vids_after_training``: if ``true`` (as well as ``eval.predict_vids_after_training``)
  the keypoints predicted during the inference step will be overlaid on the videos and saved with
  inference outputs to the model directory


.. _training-on-sample-dataset:

Training on sample dataset
============================

To quickly try lightning-pose without your own dataset, the lightning-pose git repository provides a small
sample dataset. Clone the repository and run the train command pointed at our sample config:

.. code-block:: shell

    # (Skip this if you've already cloned, i.e. to install from source.)
    git clone https://github.com/paninski-lab/lightning-pose

    # Run from a directory containing the lightning-pose repo.
    litpose train lightning-pose/scripts/configs/config_mirror-mouse-example.yaml

Tensorboard
===========

Training metrics such as losses are logged in ``model_dir/tb_logs``.
To view the logged losses via tensorboard, run:

.. code-block:: shell

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

.. code-block::

    /path/to/model/YYYY-MM-DD/HH-MM-SS/
      ├── tb_logs/
      ├── video_preds/
      │   └── labeled_videos/
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