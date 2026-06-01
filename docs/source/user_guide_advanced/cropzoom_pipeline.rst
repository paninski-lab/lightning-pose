########################
Cropzoom pipeline
########################

For setups where an animal is freely moving in a large arena,
it's advantageous to crop around the animal before running pose estimation.
Lightning Pose calls this technique "cropzoom". This document describes how
to set up a such a pipeline.

Conceptual overview
===================

A cropzoom pipeline consists of two Lightning Pose models:
a "detector model" and a "pose model".

* The detector model operates on the full image of the arena.
* The pose model operates on the cropped animal.

These two models are trained and predicted like any other
Lightning Pose model. We provide additional tools that help you compose these models:

* ``litpose create_bbox``: Given the detector model's predictions, computes per-frame
  bounding boxes and saves them as CSV files.
* ``litpose smooth_bbox``: *(optional)* Applies temporal smoothing to bbox CSV files,
  which can reduce jitter in the cropped region.
* ``litpose crop``: Given a directory of bbox CSV files, crops the animal out of each
  frame or video.
* ``litpose remap``: Given the pose model's predictions and the crop bounding boxes,
  remaps the predictions to the original coordinate space.

For the full command-line reference for these tools, see the CLI page sections:
:ref:`Create bbox <cli-create-bbox>`, :ref:`Smooth bbox <cli-smooth-bbox>`,
:ref:`Crop <cli-crop>`, and :ref:`Remap <cli-remap>`.

Training
--------

Training involves:

1. Train a "detector model".
2. Predict on training data using the detector model.
3. Create bounding boxes from detector predictions.
4. Crop training data for the pose model.
5. Train a "pose model".

Inference
---------

Inference involves:

1. Predict using the "detector model".
2. Create bounding boxes from detector predictions.
3. *(optional)* Smooth the bounding boxes.
4. Crop the data using the bounding boxes.
5. Predict on the cropped data using the "pose model".
6. Remap the pose model's predictions to the original coordinate space.


Bounding box sizing
===================

The ``litpose create_bbox`` command supports two ways to size the bounding box around the
animal.  The two options are mutually exclusive; if neither is provided,
``--crop_ratio=2.0`` is used.

``--crop_ratio`` (default)
    Sizes the bounding box relative to the per-frame span of the detected keypoints.
    A value of 2.0 (the default) produces a box twice as wide and tall as the spread of
    keypoints. Use this when the animal's apparent size varies across frames.

    .. code-block:: bash

        litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4 --crop_ratio 2.0

``--crop_size``
    Produces a fixed square bounding box of the given pixel size, centred on the
    per-frame mean of the detected keypoints. Use this when the animal occupies a
    roughly consistent region of the frame and you want uniform crop dimensions.

    .. code-block:: bash

        litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4 --crop_size 200

Bounding box smoothing
======================

After running ``litpose create_bbox``, you can optionally smooth the resulting bbox files
with ``litpose smooth_bbox``.  Smoothing reduces per-frame jitter in the crop region,
which can improve pose estimation quality when the detector produces noisy predictions.

Smoothed bboxes are written to a **new directory** alongside a ``metadata.json`` file
that records the smoothing parameters.  You can then pass this directory to
``litpose crop`` via ``--bbox_dir``.

.. code-block:: bash

    litpose smooth_bbox $MODEL_DIR/$DETECTOR_MODEL/video_preds \
        --output_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth_w5 \
        --window 5

Using bboxes from an external source
=====================================

Because ``litpose crop`` accepts a ``--bbox_dir`` argument, you can use bounding boxes
produced by any external tool, not just ``litpose create_bbox``
(e.g., `idtracker.ai <https://idtracker.ai/latest/>`_, `SAM3 <https://ai.meta.com/research/sam3/>`_, etc.).
Place your bbox CSV files in a directory following the naming convention below, then pass
``--bbox_dir`` to ``litpose crop``:

* **Videos**: ``<video_stem>_bbox.csv`` (one file per video)
* **Labeled frames**: ``bbox.csv``

Each bbox CSV must have columns ``x``, ``y``, ``h``, ``w`` (top-left corner and size in
pixels), with one row per frame.

Example
=======

This is a basic example of how you can setup a cropzoom pipeline.
Paths to CSV and MP4 files below should be replaced with your files.
The example is illustrative only. In reality you might be interested in
making modifications to this such as:

1. Using different model type, backbone, image_resize_dims for
   your detector model and pose model. This can be accomplished using
   different config files for the detector and pose model.
2. Limiting ``train_frames`` and ``max_epochs`` for testing purposes.
3. Choosing ``--crop_ratio`` or ``--crop_size`` to suit your data (see `Bounding box sizing`_ above).

We'll use some bash variables to avoid repeating paths below:

.. code-block:: bash

    MODEL_DIR=outputs/chickadee/cropzoom
    DETECTOR_MODEL=detector_0
    POSE_MODEL=pose_supervised_0

Training script
---------------

.. code-block:: bash

    #!/bin/bash

    # Train the detector model.
    litpose train config.yaml --output_dir $MODEL_DIR/$DETECTOR_MODEL

    # Predict on training data with the detector model.
    litpose predict $MODEL_DIR/$DETECTOR_MODEL data/CollectedData.csv

    # Create bounding boxes from detector predictions.
    # Use --crop_ratio (default) or --crop_size to control bounding box size.
    litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/CollectedData.csv

    # Crop images for pose model training.
    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/CollectedData.csv

    # Train the pose model.
    litpose train config.yaml --output_dir $MODEL_DIR/$POSE_MODEL \
        --detector_model $MODEL_DIR/$DETECTOR_MODEL

For command-line options of the commands used above, see
:ref:`Create bbox <cli-create-bbox>` and :ref:`Crop <cli-crop>`.

Prediction on videos script
---------------------------

.. code-block:: bash

    #!/bin/bash

    litpose predict $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.short.mp4

    litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.short.mp4

    # Optional: smooth bboxes before cropping.
    litpose smooth_bbox $MODEL_DIR/$DETECTOR_MODEL/video_preds \
        --output_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth

    # Crop using raw bboxes (default):
    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.short.mp4

    # Or crop using smoothed bboxes:
    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.short.mp4 \
        --bbox_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth

    litpose predict $MODEL_DIR/$POSE_MODEL $MODEL_DIR/$DETECTOR_MODEL/cropped_videos/cropped_test_vid.short.mp4

    litpose remap $MODEL_DIR/$POSE_MODEL/video_preds/cropped_TRQ177_200624_112234_lBack.short.csv \
        $MODEL_DIR/$DETECTOR_MODEL/video_preds/test_vid.short_bbox.csv

For detailed command-line options, see :ref:`Create bbox <cli-create-bbox>`,
:ref:`Smooth bbox <cli-smooth-bbox>`, :ref:`Crop <cli-crop>`, and :ref:`Remap <cli-remap>`.

Prediction on OOD Labeled Data
------------------------------

Say you have new labeled data for OoD animals, at `data/CollectedData_new.csv`,
and you want to predict on these frames as well as compute pixel error.

.. code-block:: bash

    #!/bin/bash

    litpose predict $MODEL_DIR/$DETECTOR_MODEL data/CollectedData_new.csv

    litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/CollectedData_new.csv

    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/CollectedData_new.csv

    litpose predict $MODEL_DIR/$POSE_MODEL \
      $MODEL_DIR/$DETECTOR_MODEL/image_preds/CollectedData_new.csv/cropped_CollectedData_new.csv

    litpose remap $MODEL_DIR/$POSE_MODEL/image_preds/cropped_CollectedData_new.csv/predictions.csv \
      $MODEL_DIR/$DETECTOR_MODEL/image_preds/CollectedData_new.csv/bbox.csv

Limitations
===========

* Pose models do not yet support PCA Multiview loss.
