########################
Cropzoom pipeline
########################

For setups where an animal is freely moving in a large arena,
it's advantageous to crop around the animal before running pose estimation.
Lightning-pose calls this technique "cropzoom". This document describes how
to set up a such a pipeline.

Conceptual overview
===================

A cropzoom pipeline consists of two lightning-pose models:
a "detector model" and a "pose model".

* The detector model operates on the full image of the arena.
* The pose model operates on the cropped animal.

These two models are trained and predicted like any other
lightning pose model. We provide additional tools that help you compose these models:

* ``litpose crop``: Given the detector model's predictions, crops around the animal.
* ``litpose remap``: Given the pose model's predictions and the crop bounding boxes,
  remaps the predictions to the original coordinate space.

For the full command-line reference for these tools, see the CLI page sections: :ref:`Crop <cli-crop>` and :ref:`Remap <cli-remap>`.

Training
--------

Training involves:

1. Train a "detector model"
2. Crop training data for the pose model.
3. Train a "pose model".

Inference
---------

Inference involves:

1. Predict using the "detector model"
2. Crop the data using the above predictions
3. Predict on the cropped data using the "pose model".
4. Remap the pose model's predictions to the original coordinate space.


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

    # Crop data for pose model training.
    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/CollectedData.csv

    # Train the pose model.
    litpose train config.yaml --output_dir $MODEL_DIR/$POSE_MODEL \
        --detector_model=$MODEL_DIR/$DETECTOR_MODEL

For command-line options of the ``litpose crop`` command used above, see :ref:`the CLI Crop section <cli-crop>`.

Prediction on videos script
---------------------------

.. code-block:: bash

    #!/bin/bash

    litpose predict $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.short.mp4

    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/videos/test_vid.short.mp4

    litpose predict $MODEL_DIR/$POSE_MODEL $MODEL_DIR/$DETECTOR_MODEL/cropped_videos/cropped_test_vid.short.mp4

    litpose remap $MODEL_DIR/$POSE_MODEL/video_preds/cropped_TRQ177_200624_112234_lBack.short.csv \
        $MODEL_DIR/$DETECTOR_MODEL/video_preds/test_vid.short_bbox.csv

For detailed command-line options, see :ref:`Crop <cli-crop>` and :ref:`Remap <cli-remap>`.

Prediction on OOD Labeled Data
------------------------------

Say you have new labeled data for OoD animals, at `data/CollectedData_new.csv`,
and you want to predict on these frames as well as compute pixel error.

.. code-block:: bash

    #!/bin/bash

    litpose predict $MODEL_DIR/$DETECTOR_MODEL data/CollectedData_new.csv

    litpose crop $MODEL_DIR/$DETECTOR_MODEL data/CollectedData_new.csv

    litpose predict $MODEL_DIR/$POSE_MODEL \
      $MODEL_DIR/$DETECTOR_MODEL/image_preds/CollectedData_new.csv/cropped_CollectedData_new.csv

    litpose remap $MODEL_DIR/$POSE_MODEL/image_preds/cropped_CollectedData_new.csv/predictions.csv \
      $MODEL_DIR/$DETECTOR_MODEL/image_preds/CollectedData_new.csv/bbox.csv

Limitations
===========

* Pose models do not yet support PCA Multiview loss.