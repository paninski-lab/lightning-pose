.. _config_file:

######################
The configuration file
######################

Users interact with Lighting Pose through a single configuration file. This file points to data
directories, defines the type of models to fit, and specifies a wide range of hyperparameters.

A template file can be found
`here <https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml>`_.
When training a model on a new dataset, you must copy/paste this template onto your local machine
and update the arguments to match your data.

The config file contains several sections:

* ``data``: information about where data is stored, keypoint names, etc.
* ``training``: batch size, training epochs, image augmentation, etc.
* ``model``: backbone architecture, unsupervised losses to use, etc.
* ``dali``: batch sizes for unlabeled video data
* ``losses``: hyperparameters for unsupervised losses
* ``eval``: paths for video inference and fiftyone app

Data parameters
===============

All of these parameters except ``downsample_factor`` are dataset-specific and will need to be
provided.

* ``data.image_resize_dims.height/width`` (*int*): images (and videos) will be resized to the
  specified height and width before being processed by the network.
  Supported values are {64, 128, 256, 384, 512}.
  The height and width need not be identical.
  Some points to keep in mind when selecting these values:
  if the resized images are too small, you will lose resolution/details;
  if they are too large, the model takes longer to train and might not train as well.

* ``data.data_dir/video_dir`` (*str*): update these to reflect your (absolute) local paths

* ``data.csv_file`` (*str*): location of labels csv file; this should be relative to
  ``data.data_dir``

* ``data.downsample_factor`` (*int, default: 2*): factor by which to downsample the heatmaps
  relative to ``data.image_resize_dims``

* ``data.num_keypoints`` (*int*): the number of body parts.
  If using a mirrored setup, this should be the number of body parts summed across all views.
  If using a multiview setup, this number should indicate the number of keyponts per view
  (must be the same across all views).

* ``data.keypoint_names`` (*list*): keypoint names should reflect the actual names/order in the
  csv file.
  This field is necessary if, for example, you are running inference on a machine that does not
  have the training data saved on it.

* ``data.mirrored_column_matches`` (*list*): see the
  :ref:`Multiview PCA documentation <unsup_loss_pcamv>`

* ``data.columns_for_singleview_pca`` (*list*): see the
  :ref:`Pose PCA documentation <unsup_loss_pcasv>`


Training parameters
===================

The following parameters relate to model training.
Reasonable defaults are provided, though parameters like the batch sizes
(``train_batch_size``, ``val_batch_size``, ``test_batch_size``)
may need modification depending on the size of the data and the available compute resources.
See the :ref:`FAQs <faq_oom>` for more information on memory management.

* ``training.imgaug`` (*str, default: dlc*): select from one of several predefined image/video
  augmentation pipelines:

  * default: resizing only
  * dlc: imgaug pipeline implmented in DLC 2.0 package
  * dlc-lr: dlc augmentations plus horizontal flips
  * dlc-top-down: dlc augmentations plus additional vertical and horizontal flips

  You can also define custom augmentation pipelines following
  :ref:`these instructions <custom_imgaug_pipeline>`.

* ``training.train_batch_size`` (*int, default: 16*): batch size for labeled data during training

* ``training.val_batch_size`` (*int, default: 32*): batch size for labeled data during validation

* ``training.test_batch_size`` (*int, default: 32*): batch size for labeled data during test

* ``training.train_prob`` (*float, default: 0.95*): fraction of labeled data used for training

* ``training.val_prob`` (*float, default: 0.05*): fraction of labeled data used for validation;
  any remaining frames not assigned to train or validation sets are assigned to the test set

* ``training.train_frames`` (*float or int, default: 1*): this parameter determines how many of the
  frames assigned to to training data (using ``train_prob``) are actually used for training.
  This option is generally more useful for testing new algorithms rather than training production
  models.
  If the value is a float between 0 and 1 then it is interpreted as the fraction of total train frames.
  If the value is an integer greater than 1 then it is interpreted as the number of total train frames.

    .. _config_num_gpus:

* ``training.num_gpus`` (*int, default: 1*): the number of GPUs for
  :ref:`multi-GPU training <multi_gpu_training>`

* ``training.num_workers`` (*int, default: num_cpus*): number of cpu workers for data loaders

* ``training.unfreezing_epoch`` (*int, default: 20*): epoch at which backbone network weights begin
  updating. A value >0 allows the smaller number of parameters in the heatmap head to adjust to
  the backbone outputs first.

* ``training.min_epochs`` / ``training.max_epochs`` (*int, default: 300*): length of training.
  An epoch is one full pass through the dataset.
  As an example, if you have 400 labeled frames, and ``training.train_batch_size=10``, then your
  dataset is divided into 400/10 = 40 batches.
  One "batch" in this case is equivalent to one "iteration" for DeepLabCut.
  Therefore, 300 epochs, at 40 batches per epoch, is equal to 300*40=12k total batches
  (or iterations).

* ``training.log_every_n_steps`` (*int, default: 10*): frequency to log training metrics for
  tensorboard (one step is one batch)

* ``training.check_val_every_n_epochs`` (*int, default: 5*): frequency to log validation metrics
  for tensorboard

* ``training.ckpt_every_n_epochs`` (*int or null, default: null*): save model weights every n
  epochs; must be divisible by ``training.check_val_every_n_epochs`` above.
  If null, only the best weights will be saved after training, where "best" is defined as the
  weights from the epoch with the lowest validation loss.

* ``training.early_stopping`` (*bool, default: false*): if false, the default is to train for the
  max number of epochs and save out the best model according to the validation loss; if true, early
  stopping will exit training if the validation loss continues to increase for a given number of
  validation checks (see ``training.early_stop_patience`` below).

* ``training.early_stop_patience`` (*int, default: 3*): number of validation checks over which to
  assess validation metrics for early stopping; this number, multiplied by
  ``training.ckpt_every_n_epochs``, gives the number of epochs over which the validation loss must
  increase before exiting.

* ``training.rng_seed_data_pt`` (*int, default: 0*): rng seed for splitting labeled data into
  train/val/test

* ``training.rng_seed_model_pt`` (*int, default: 0*): rng seed for weight initialization of the head

* ``training.lr_scheduler`` (*str, default: multisteplr*): reduce the learning rate by a certain
  factor after a given number of epochs (see ``training.lr_scheduler_params.multisteplr`` below)

* ``training.lr_scheduler_params.multistep_lr``: milestones: epochs at which to reduce learning
  rate; gamma: factor by which to multiply learning rate at each milestone

* ``training.uniform_heatmaps_for_nan_keypoints`` (*bool, default: true*): how to treat missing
  hand labels.
  Setting this to true will encourage the model to output uniform heatmaps for keypoints that do
  not have ground truth labels; this will generally lead to low-confidence predictions when a
  keypoint is occluded.
  Setting this to false will drop missing keypoints from the loss computation rather than
  encouraging uniform heatmaps. This generally leads to high confidence predictions even when a
  keypoint is occluded. Using false may be preferrable if occulsions are brief in time and you want
  the network to guess where the keypoint should be (rather than signaling uncertainty).

* ``training.accumulate_grad_batches`` (*int, default: 1*): (experimental) number of batches to
  accumulate gradients for before updating weights. Simulates larger batch sizes with
  memory-constrained GPUs.
  This parameter is not included in the config by default and should be added manually to the
  ``training`` section.

Model parameters
================

The following parameters relate to model architecture and unsupervised losses.


* ``model.losses_to_use`` (*list, default: []*): defines the unsupervised losses.
  An empty list indicates a fully supervised model.
  Each element of the list corresponds to an unsupervised loss.
  For example, ``model.losses_to_use=[pca_multiview,temporal]`` will fit both a pca_multiview loss
  and a temporal loss. Options include:

    * pca_multiview: penalize inconsistencies between multiple camera views
    * pca_singleview: penalize implausible body configurations
    * temporal: penalize large temporal jumps

  See the :ref:`unsupervised losses<unsupervised_losses>` page for more details on the various
  losses and their associated hyperparameters.


* ``model.backbone`` (*str, default: resnet50_animal_ap10k*): a variety of pretrained backbones are
  available:

    * resnet50_animal_ap10k: ResNet-50 pretrained on the AP-10k dataset (Yu et al 2021, AP-10k: A Benchmark for Animal Pose Estimation in the Wild)
    * resnet18: ResNet-18 pretrained on ImageNet
    * resnet34: ResNet-34 pretrained on ImageNet
    * resnet50: ResNet-50 pretrained on ImageNet
    * resnet101: ResNet-101 pretrained on ImageNet
    * resnet152: ResNet-152 pretrained on ImageNet
    * resnet50_contrastive: ResNet-50 pretrained on ImageNet using SimCLR
    * resnet50_animal_apose: ResNet-50 pretrained on an animal pose dataset (Cao et al 2019, Cross-Domain Adaptation for Animal Pose Estimation)
    * resnet50_human_jhmdb: ResNet-50 pretrained on JHMDB dataset (Jhuang et al 2013, Towards Understanding Action Recognition)
    * resnet50_human_res_rle: a regression-based ResNet-50 pretrained on MPii dataset (Andriluka et al 2014, 2D Human Pose Estimation: New Benchmark and State of the Art Analysis)
    * resnet50_human_top_rle: a heatmap-based ResNet-50 pretrained on MPii dataset (Xiao et al 2014, Simple Baselines for Human Pose Estimation and Tracking)
    * resnet50_human_hand: ResNet-50 pretrained on OneHand10k dataset (Wang et al 2018, Mask-pose Cascaded CNN for 2d Hand Pose Estimation from Single Color Image)
    * efficientnet_b0: EfficientNet-B0 pretrained on ImageNet
    * efficientnet_b1: EfficientNet-B1 pretrained on ImageNet
    * efficientnet_b2: EfficientNet-B2 pretrained on ImageNet
    * vit_b_sam: Segment Anything Model (Vision Transformer Base)

  Note: the file size for a single ResNet-50 network is approximately 275 MB.


* ``model.model_type`` (*str, default: heatmap*):

    * regression: model directly outputs an (x, y) prediction for each keypoint; not recommended
    * heatmap: model outputs a 2D heatmap for each keypoint
    * heatmap_mhcrnn: the "multi-head convolutional RNN", this model takes a temporal window of
      frames as input, and outputs two heatmaps: one "context-aware" and one "static".
      The prediction with the highest confidence is automatically chosen.
      See the :ref:`Temporal Context Network<mhcrnn>` page for more information.

* ``model.heatmap_loss_type`` (*str, default: mse*): (experimental) loss to compute difference
  between ground truth and predicted heatmaps

* ``model.model_name`` (*str, default: test*): directory name for model saving

* ``model.checkpoint`` (*str or null, default: null*): to initialize weights from an existing
  checkpoint, update this parameter to the absolute path of a pytorch .ckpt file


Video loading parameters
========================

Some parameters relate to video loading, both for semi-supervised models and when predicting new
videos with any of the models.
The parameters may need modification depending on the size of the data and the available compute
resources.
See the :ref:`FAQs <faq_oom>` for more information on memory management.

* ``dali.base.train.sequence_length`` (*int, default: 32*): number of unlabeled frames per batch in
  "regression" and "heatmap" models (i.e. "base" models that do not use temporal context frames)
* ``dali.base.predict.sequence_length`` (*int, default: 96*): batch size when predicting on a new
  video with a base model
* ``dali.context.train.batch_size`` (*int, default: 16*): number of unlabeled frames per batch in
  heatmap_mhcrnn model (i.e. "context" models that utilize temporal context frames)
* ``dali.context.predict.sequence_length`` (*int, default: 96*): batch size when predicting on a
  new video with a "context" model

Evaluation
==========

The following parameters are used for general evaluation.

* ``eval.predict_vids_after_training`` (*bool, default: true*): if true, after training run
  inference on all videos located in ``eval.test_videos_directory`` (see below)

* ``eval.test_videos_directory`` (*str, default: null*): absolute path to a video directory
  containing videos for post-training prediction.

* ``eval.save_vids_after_training`` (*bool, default: false*): save out an mp4 file with predictions
  overlaid after running post-training prediction.

* ``eval.colormap`` (*str, default: cool*): colormap options for labeled videos; options include
  sequential colormaps (viridis, plasma, magma, inferno, cool, etc) and diverging colormaps (RdBu,
  coolwarm, Spectral, etc)

* ``eval.confidence_thresh_for_vid`` (*float, default: 0.9*): predictions with confidence below this
  value will not be plotted in the labeled videos

* ``eval.fiftyone.dataset_name`` (*str, default: test*): name of the FiftyOne dataset

* ``eval.fiftyone.model_display_names`` (*list, default: [test_model]*): shorthand name for each of
  the models specified in ``hydra_paths``

* ``eval.hydra_paths`` (*list, default: []*): absolute paths to model directories, only for use with
  scripts/create_fiftyone_dataset.py (see :ref:`FiftyOne <fiftyone>` docs).