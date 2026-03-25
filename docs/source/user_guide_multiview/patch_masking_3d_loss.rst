.. _patch_masking_3d_loss:

##########################
Patch masking and 3D loss
##########################

This section covers advanced training techniques specific to multi-view setups, including
patch masking strategies and 3D loss functions that leverage the geometric relationships
between camera views.

Both of these techniques require you to use the multi-view transformer (MVT) model,
specified in the configuration file:

.. code-block:: yaml

    model:
        backbone: vits_dino
        model_type: heatmap_multiview_transformer

The backbone can be any of the available backbones that start with the string "vit"
(see options :ref:`here <config_file_model>`),
indicating Vision Transformer.
The "heatmap_multiview_transformer" will then use the specified backbone to process all camera
view simultaneously.


Patch masking
=============

The self-attention of the MVT enables the network to utilize information from multiple views, which
is particularly advantageous for handling occlusions.
To encourage the model to develop this cross-view reasoning during training, we introduce a pixel
space patch masking scheme inspired by the success of masked autoencoders and dropout.
We use a training curriculum that starts with a short warmup period where no patches are masked
(controlled by ``training.patch_mask.init_epoch`` in the config file), then increase the ratio of
masked patches over the course of training
(controlled by ``training.patch_mask.init_ratio`` and ``training.patch_mask.final_ratio``).
This technique creates gradients that flow through the attention mechanism and encourage
cross-view information propagation, which in turn develops internal representations that capture
statistical relationships between the different views.

.. code-block:: yaml

    training:
        patch_mask:
            init_epoch: 40     # epoch to start patch masking
            final_epoch: 300   # epoch when patch masking reaches maximum
            init_ratio: 0.0    # initial masking ratio
            final_ratio: 0.5   # final masking ratio

To turn patch masking off, set ``final_ratio: 0.0``.

3D augmentations and loss
=========================

.. note::

    As of March 2026, the unsupervised losses introduced in the original Lightning Pose paper have
    not yet been implemented for the ``multi-view transformer`` model, including the
    ``pca_multiview`` loss.

The MVT produces a 2D heatmap for each keypoint in each view.
Without explicit geometric constraints, it is possible for these individual 2D predictions to be
geometrically inconsistent with each other.
If we have access to camera parameters, we can use this additional information to
encourage geometric consistency in the outputs
(see the :ref:`camera calibration section <camera_calibration>` for details on required data
formats for camera calibration; note also that bounding box information must be shared if the
training images are cropped from larger frames).

The 3D loss requires geometrically consistent input images, which precludes applying geometric
augmentations like rotation to each view independently.
Instead, we triangulate the ground truth labels and augment the 3D poses by translating and scaling in 3D space.
The augmented 3D pose is then projected back to individual 2D views.
These augmentations do not affect the camera parameters;
rather, they are equivalent to keeping the cameras fixed and scaling and translating the subject within the scene.
For each view, we then estimate the affine transformation from the original to augmented 2D keypoints,
and apply this transformation to the original image.

To enable 3D augmentations, add the ``imgaug_3d`` field to the ``training`` section of your configuration
file and set it to `true`:

.. code-block:: yaml

    training:
        imgaug: dlc
        imgaug_3d: true

To compute the 3D reprojection loss, we:

1. take the soft argmax of the 2D heatmaps to get predicted coordinates.
2. for each keypoint, and for each pair of views, we triangulate the predictions into 3D
3. project the predicted 3D points back into 2D coordinates for each view
4. turn these reprojected coordinates into heatmaps
5. computes the mean square error between the reprojected and ground truth heatmaps.

The advantage of this loss is that it is on the same scale as the standard supervised heatmap loss,
which may make for easier hyperparameter tuning.

The default ``log_weight`` value of 1.0 should be a reasonable place to start; if the training curve
for this loss is unstable (for example it doesn't decrease, or spikes to a large value during training),
you can _decrease_ the effect of the 3D loss by _increasing_ the log_weight; we recommend a secondary
value of 1.5.

.. code-block:: yaml

    losses:
        supervised_reprojection_heatmap_mse:
            log_weight: 1.0

To turn this loss off (but, for example, continue to use 3D augmentations), set
``log_weight: null`` in the config file.
