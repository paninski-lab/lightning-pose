.. _config_file:

######################
The configuration file
######################

Users interact with Lighting Pose through a single configuration file. This file points to data
directories, defines the type of models to fit, and specifies a wide range of hyperparameters.

A template file can be found
`here <https://github.com/danbider/lightning-pose/blob/main/scripts/configs/config_default.yaml>`_.
When training a model on a new dataset, you must copy/paste this template onto your local machine
and update the arguments to match your data.

The config file contains several sections:

* ``data``: information about where data is stored, keypoint names, etc.
* ``training``: batch size, training epochs, image augmentation, etc.
* ``model``: backbone architecture, unsupervised losses to use, etc.
* ``dali``: batch sizes for unlabeled video data
* ``losses``: hyperparameters for unsupervised losses
* ``eval``: paths for video inference and fiftyone app

Model/training parameters
=========================

Below is a list of some commonly modified arguments related to model architecture/training.

* ``training.train_batch_size``: batch size for labeled data
* ``training.min_epochs`` / ``training.max_epochs``: length of training
* ``model.model_type``:
    * regression: model directly outputs an (x, y) prediction for each keypoint; not recommended
    * heatmap: model outputs a 2D heatmap for each keypoint
    * heatmap_mhcrnn: the "multi-head convolutional RNN", this model takes a temporal window of frames as input, and outputs two heatmaps: one "context-aware" and one "static". The prediction with the highest confidence is automatically chosen.
* ``model.losses_to_use``: defines the unsupervised losses. An empty list indicates a fully supervised model. Each element of the list corresponds to an unsupervised loss. For example, ``model.losses_to_use=[pca_multiview,temporal]`` will fit both a pca_multiview loss and a temporal loss. Options include:
    * pca_multiview: penalize inconsistencies between multiple camera views
    * pca_singleview: penalize implausible body configurations
    * temporal: penalize large temporal jumps

See the :ref:`Unsupervised losses <unsupervised_losses>` section for more details on the various
losses and their associated hyperparameters.


Video loading parameters
========================

Some arguments relate to video loading, both for semi-supervised models and when predicting new
videos with any of the models:

* ``dali.base.train.sequence_length`` - number of unlabeled frames per batch in ``regression`` and ``heatmap`` models (i.e. "base" models that do not use temporal context frames)
* ``dali.base.predict.sequence_length`` - batch size when predicting on a new video with a "base" model
* ``dali.context.train.batch_size`` - number of unlabeled frames per batch in ``heatmap_mhcrnn`` model (i.e. "context" models that utilize temporal context frames); each frame in this batch will be accompanied by context frames, so the true batch size will actually be larger than this number
* ``dali.context.predict.sequence_length`` - batch size when predicting on a new video with a "context" model
