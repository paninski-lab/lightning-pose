.. _custom_imgaug_pipeline:

#############################
Custom augmentation pipelines
#############################

Overview
========

Data augmentation is essential for robust pose estimation.
Augmentations fall into two categories:

* **Image-only augmentations**: Modify images while preserving keypoint positions
  (e.g., Gaussian noise, pixel dropout)
* **Image-and-keypoint augmentations**: Transform both images and ground truth keypoints
  (e.g., cropping, rotation)

Built-in pipelines
==================

Lightning Pose provides predefined augmentation pipelines accessible via the ``training.imgaug``
field in the config file.
Set this to a string corresponding to one of the available options (see the
:ref:`configuration file documentation<config_file>`).

Creating custom pipelines
=========================
To define a custom augmentation pipeline, replace the string value in ``training.imgaug`` with a
list of augmentations.
For each augmentation, specify:

1. The transformation name (from the `imgaug package <https://imgaug.readthedocs.io/>`_)
2. Application probability
3. Required arguments

Simple example
--------------
This example applies a
`Rotate <https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html#rotate>`_
transformation that randomly rotates images and keypoints between -10 and 10 degrees with 50%
probability.

.. code-block:: yaml

    training:
      imgaug:
        Rotate:
          p: 0.5
          kwargs:
            rotate: [-10, 10]

* ``p``: Probability of applying the transformation
* ``kwargs`` arguments passed to the Rotate function.

Complex example: 'dlc' pipeline
-------------------------------

This more complex example combines multiple transformations.
The order listed in the config file determines the sequence in which transformations are applied.

.. note::

    Parameters defined as tuples in the imgaug documentation must use square brackets in the config
    file due to OmegaConf's YAML parsing behavior.

.. code-block:: yaml

    training:
      imgaug:
        Rotate:
          p: 0.4
          kwargs:
            rotate: [-25, 25]
        MotionBlur:
          p: 0.5
          kwargs:
            k: 5
            angle: [-90, 90]
        CoarseDropout:
          p: 0.5
          kwargs:
            p: 0.02
            size_percent: 0.3
            per_channel: 0.5
        CoarseSalt:
          p: 0.5
          kwargs:
            p: 0.01
            size_percent: [0.05, 0.1]
        CoarsePepper:
          p: 0.5
          kwargs:
            p: 0.01
            size_percent: [0.05, 0.1]
        ElasticTransformation:
          p: 0.5
          kwargs:
            alpha: [0, 10]
            sigma: 5
        AllChannelsHistogramEqualization:
          p: 0.1
        AllChannelsCLAHE:
          p: 0.1
        Emboss:
          p: 0.1
          kwargs:
            alpha: [0, 0.5]
            strength: [0.5, 1.5]
        CropAndPad:
          p: 0.4
          kwargs:
            percent: [-0.15, 0.15]
            keep_size: false

Horizontal flip for lateralized keypoints
==========================================

Many animal pose datasets have **lateralized keypoints** — pairs of landmarks that appear on the
left and right sides of the body (e.g. ``ear_left`` / ``ear_right``, ``paw_front_left`` /
``paw_front_right``).
Randomly flipping images horizontally is a powerful augmentation for these datasets, but a naive
flip must also **swap the left/right labels**: after a flip the right ear appears on the left side
of the image and must therefore be relabeled as ``ear_left``.

Enable this with the ``training.imgaug_hflip`` flag:

.. code-block:: yaml

    training:
      imgaug_hflip: true

With ``imgaug_hflip: true``, each training image is horizontally flipped with probability 0.5.
All keypoint x-coordinates are mirrored, and any ``_left`` / ``_right`` pair is additionally
swapped so that label identity is preserved.

Naming convention
-----------------

Lightning Pose detects lateralized pairs by suffix:

* Keypoints whose names end in ``_left`` are paired with the corresponding ``_right`` keypoint
  (e.g. ``ear_tip_left`` ↔ ``ear_tip_right``).
* Keypoints that do **not** end in ``_left`` or ``_right`` (e.g. ``nose_tip``, ``spine_mid``)
  are treated as midline landmarks: their x-coordinate is mirrored but their label is unchanged.

If any ``_left`` keypoint has no matching ``_right`` partner (or vice versa) an error is raised
at dataset construction time.

.. note::

    ``imgaug_hflip`` is **not** supported for multiview models and will raise an error if set to
    ``true`` in a multiview config.
