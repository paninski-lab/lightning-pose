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

Data parameters
===============

* ``data.image_orig_dims.height/width``: the current version of Lightning Pose requires all
  training images to be the same size.
  We are working on an updated version without this requirement.
  However, if you plan to use the PCA losses (Pose PCA or multiview PCA) then all training images
  **must** be the same size, otherwise the PCA subspace will erroneously contain variance related
  to image size.

* ``data.image_resize_dims.height/width``: images (and videos) will be resized to the specified
  height and width before being processed by the network.
  Supported values are {64, 128, 256, 384, 512}.
  The height and width need not be identical.
  Some points to keep in mind when selecting these values:
  if the resized images are too small, you will lose resolution/details;
  if they are too large, the model takes longer to train and might not train as well.

* ``data.data_dir/video_dir``: update these to reflect your local paths

* ``data.num_keypoints``: the number of body parts.
  If using a mirrored setup, this should be the number of body parts summed across all views.
  If using a multiview setup, this number should indicate the number of keyponts per view
  (must be the same across all views).

* ``data.keypoint_names``: keypoint names should reflect the actual names/order in the csv file.
  This field is necessary if, for example, you are running inference on a machine that does not
  have the training data saved on it.

* ``data.columns_for_singleview_pca``: see the :ref:`Pose PCA documentation <unsup_loss_pcasv>`

* ``data.mirrored_column_matches``: see the :ref:`Multiview PCA documentation <unsup_loss_pcamv>`


Model/training parameters
=========================

Below is a list of some commonly modified arguments related to model architecture/training.

* ``training.train_batch_size``: batch size for labeled data

* ``training.min_epochs`` / ``training.max_epochs``: length of training.
  An epoch is one full pass through the dataset.
  As an example, if you have 400 labeled frames, and ``training.train_batch_size=10``, then your
  dataset is divided into 400/10 = 40 batches.
  One "batch" in this case is equivalent to one "iteration" for DeepLabCut.
  Therefore, 300 epochs, at 40 batches per epoch, is equal to 300*40=12k total batches
  (or iterations).

* ``model.model_type``:

    * regression: model directly outputs an (x, y) prediction for each keypoint; not recommended
    * heatmap: model outputs a 2D heatmap for each keypoint
    * heatmap_mhcrnn: the "multi-head convolutional RNN", this model takes a temporal window of
      frames as input, and outputs two heatmaps: one "context-aware" and one "static".
      The prediction with the highest confidence is automatically chosen.

* ``model.losses_to_use``: defines the unsupervised losses.
  An empty list indicates a fully supervised model.
  Each element of the list corresponds to an unsupervised loss.
  For example, ``model.losses_to_use=[pca_multiview,temporal]`` will fit both a pca_multiview loss
  and a temporal loss. Options include:

    * pca_multiview: penalize inconsistencies between multiple camera views
    * pca_singleview: penalize implausible body configurations
    * temporal: penalize large temporal jumps

* ``model.checkpoint``: to initialize weights from an existing checkpoint, update this parameter
  to the absolute path of a pytorch .ckpt file

* ``model.backbone``: a variety of pretrained backbones are available:

    * resnet50_animal_ap10k (recommended): ResNet-50 pretrained on the AP-10k dataset (Yu et al 2021, AP-10k: A Benchmark for Animal Pose Estimation in the Wild)
    * resnet18: ResNet-18 pretrained on ImageNet
    * resnet34: ResNet-34 pretrained on ImageNet
    * resnet50: ResNet-50 pretrained on ImageNet
    * resnet101: ResNet-101 pretrained on ImageNet
    * resnet152: ResNet-152 pretrained on ImageNet
    * resnet50_contrastive: ResNet-50 pretrained on ImageNet using SimCLR
    * resnet50_animal_apose: ResNet-50 pretrained on an animal pose dataset (Cao et al 2019, Cross-Domain Adaptation for Animal Pose Estimation)
    * resnet50_human_jhmdb: ResNet-50 pretrained on JHMDB dataset (Jhuang et al 2013, Towards Understanding Action Recognition)
    * resnet50_human_res_rle: ResNet-50 pretrained on MPii dataset (Andriluka et al 2014, 2D Human Pose Estimation: New Benchmark and State of the Art Analysis)
    * resnet50_human_hand: ResNet-50 pretrained on OneHand10k dataset (Wang et al 2018, Mask-pose Cascaded CNN for 2d Hand Pose Estimation from Single Color Image)
    * efficientnet_b0: EfficientNet-B0 pretrained on ImageNet
    * efficientnet_b1: EfficientNet-B1 pretrained on ImageNet
    * efficientnet_b2: EfficientNet-B2 pretrained on ImageNet
    * vit_b_sam: Segment Anything Model (Vision Transformer Base)

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
