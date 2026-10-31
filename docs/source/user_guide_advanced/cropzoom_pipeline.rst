########################
Cropzoom pipeline
########################

For setups where an animal is freely moving in a large arena,
it's advantageous to crop around the animal before running pose estimation.
Lightning Pose calls this technique "cropzoom". This document describes how
to set up such a pipeline.

.. tip::

   This pipeline works equally well with bounding boxes from any external source
   (e.g., `idtracker.ai <https://idtracker.ai/latest/>`_,
   `SAM3 <https://ai.meta.com/research/sam3/>`_, or your own custom detection scripts).
   Wherever the examples below call ``litpose predict [detector_model]`` and
   ``litpose create_bbox [detector_model]``, substitute your own scripts that produce
   bbox CSV files in the required format (see `Using bboxes from an external source`_
   below). All downstream steps are identical.

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

Alternatively, ``litpose predict --bbox_dir`` combines the crop, predict, and remap steps
into a single command, and ``litpose predict --detector`` goes one step further and also runs
the detector, so a video goes from raw frames to original-coordinate predictions in a single
in-memory pass — see :ref:`prediction-on-videos` for details.

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

.. note::

   Steps 4–6 can be replaced by a single ``litpose predict --bbox_dir`` call, and all of
   steps 1–6 (except smoothing) by a single ``litpose predict --detector`` call —
   see :ref:`prediction-on-videos` in the example below.


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
``litpose crop`` or ``litpose predict`` via ``--bbox_dir``.

.. code-block:: bash

    litpose smooth_bbox $MODEL_DIR/$DETECTOR_MODEL/video_preds \
        --output_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth_w5 \
        --window 5

Using bboxes from an external source
=====================================

Because ``litpose crop`` and ``litpose predict`` both accept a ``--bbox_dir`` argument,
you can use bounding boxes produced by any external tool, not just ``litpose create_bbox``
(e.g., `idtracker.ai <https://idtracker.ai/latest/>`_,
`SAM3 <https://ai.meta.com/research/sam3/>`_, etc.).
Place your bbox CSV files in a directory following the naming convention below, then pass
``--bbox_dir`` to either command:

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

.. _prediction-on-videos:

Prediction on videos
--------------------

.. tab-set::

   .. tab-item:: From original videos (1-stage)

      A single command loads both models, runs the detector on every frame, computes a
      bounding box from its keypoints, crops the full-resolution frame, runs the pose model
      on the crop, and maps the keypoints back to the original coordinate space. The positional
      model directory is the **pose** model; ``--detector`` points at the detector model.

      **Pros**

      - One command, one pass over the video, nothing intermediate written to disk except the
        final outputs: ``<video_stem>.csv`` (predictions in original coordinates) and
        ``<video_stem>_bbox.csv`` (the per-frame boxes), both saved under
        ``$MODEL_DIR/$POSE_MODEL/video_preds``. The bbox file can be reused with
        ``--bbox_dir`` or ``litpose crop``.
      - The detector and pose model are independent: they may use different
        ``image_resize_dims`` and different keypoints.

      **Cons**

      - Bounding boxes are **not smoothed**. If you want smoothing, use the multi-command
        workflows in the other tabs.
      - Detector confidence is **ignored by default**: every anchor keypoint's predicted
        position is used, however low its confidence. This reproduces ``litpose create_bbox``,
        so the same detector gives the same boxes whether you run one command or several.
        To treat low-confidence keypoints as missing (for example, a beak that is hidden when
        the bird faces away from the camera), set ``--detector_confidence_thresh``. Anchor
        keypoints below the threshold are then dropped, and the box is centred on and sized
        from the remaining ones. A frame left with fewer than 2 usable anchor keypoints
        reuses the previous frame's box (the full frame, if there is no previous box).
      - Full-resolution frames are held in memory while they are cropped, and both models
        are loaded at once. If inference runs out of GPU memory, lower ``--batch_size``
        (default 16 for this command).
      - Videos only (not labeled-frame CSVs), single-view and non-context models only,
        eager runtime only.

      .. code-block:: bash

         #!/bin/bash

         litpose predict $MODEL_DIR/$POSE_MODEL data/videos/test_vid.mp4 \
             --detector $MODEL_DIR/$DETECTOR_MODEL

         # Control the bounding box with the same options as `litpose create_bbox`:
         litpose predict $MODEL_DIR/$POSE_MODEL data/videos/test_vid.mp4 \
             --detector $MODEL_DIR/$DETECTOR_MODEL --crop_size 200 --anchor_keypoints nose,tail

   .. tab-item:: From original videos (2-stage)

      **Pros**

      - Runs entirely in memory — no intermediate cropped files are written to disk.
      - Predictions are returned in the original coordinate space; no remap step is needed.

      **Cons**

      - Without intermediate cropped files, it is harder to inspect the detector output
        or diagnose issues at each stage of the pipeline.

      .. code-block:: bash

         #!/bin/bash

         litpose predict $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4

         litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4

         # Optional: smooth bboxes.
         litpose smooth_bbox $MODEL_DIR/$DETECTOR_MODEL/video_preds \
             --output_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth

         # Predict using raw bboxes:
         litpose predict $MODEL_DIR/$POSE_MODEL data/videos/test_vid.mp4 \
             --bbox_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds

         # Or predict using smoothed bboxes:
         litpose predict $MODEL_DIR/$POSE_MODEL data/videos/test_vid.mp4 \
             --bbox_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth

   .. tab-item:: From cropped videos

      **Pros**

      - Intermediate cropped videos are stored on disk, making it straightforward to
        inspect each stage of the pipeline and diagnose potential issues in the detector
        or pose estimator.

      **Cons**

      - Cropped videos are written to disk, requiring additional storage and compute.
      - An extra remap step is needed to convert predictions back to the original
        coordinate space.

      .. code-block:: bash

         #!/bin/bash

         litpose predict $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4

         litpose create_bbox $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4

         # Optional: smooth bboxes before cropping.
         litpose smooth_bbox $MODEL_DIR/$DETECTOR_MODEL/video_preds \
             --output_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth

         # Crop using raw bboxes (default):
         litpose crop $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4

         # Or crop using smoothed bboxes:
         litpose crop $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.mp4 \
             --bbox_dir $MODEL_DIR/$DETECTOR_MODEL/video_preds/bboxes_smooth

         litpose predict $MODEL_DIR/$POSE_MODEL \
             $MODEL_DIR/$DETECTOR_MODEL/cropped_videos/cropped_test_vid.mp4

         litpose remap $MODEL_DIR/$POSE_MODEL/video_preds/cropped_test_vid.csv \
             $MODEL_DIR/$DETECTOR_MODEL/video_preds/test_vid_bbox.csv

For detailed command-line options, see :ref:`Create bbox <cli-create-bbox>`,
:ref:`Smooth bbox <cli-smooth-bbox>`, :ref:`Crop <cli-crop>`, and :ref:`Remap <cli-remap>`.

Multiview models
================

``litpose predict --bbox_dir`` also works with multiview models. Each view is cropped to its
own bounding boxes, so the views may have different resolutions. The bbox directory must contain
one ``<video_stem>_bbox.csv`` for *every view's video* (e.g. ``session_viewA_bbox.csv`` and
``session_viewB_bbox.csv`` for ``session_viewA.mp4`` and ``session_viewB.mp4``), each with one row
per frame of its video. Predictions are saved in each view's original coordinate space.

.. code-block:: console

    litpose predict $MODEL_DIR/$POSE_MODEL data/videos/ --bbox_dir data/bboxes

The same is available from the Python API with
``Model.predict_on_video_file_multiview(videos, bbox_file_per_view=bbox_files)``, where
``bbox_files`` is in the same order as ``videos``.

.. note::

   Predicting with bounding boxes decodes full-resolution frames (cropping happens after
   decoding), so memory use grows with the video resolution, the number of views, and the batch
   size. If inference runs out of GPU memory, lower the batch size with
   ``litpose predict --batch_size``.

Limitations
===========

* ``litpose predict --detector`` does not yet support multiview models, context models,
  or bounding-box smoothing.
* Pose models do not yet support PCA Multiview loss.
