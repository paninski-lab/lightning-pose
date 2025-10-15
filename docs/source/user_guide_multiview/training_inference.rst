.. _multiview_training_inference:

######################
Training and inference
######################

This section covers the training and inference procedures specific to multi-view setups.

Training
========

Once the data are properly organized and the config files updated, :ref:`training <training>` in 
this multi-view setup proceeds exactly the same as for the single-view case.

The multi-view transformer architecture processes all camera views simultaneously during training,
learning cross-view correlations and improving robustness to occlusions.
Given the transformer-based architecture, this model requires a larger memory footprint which
scales quadratically with the number of views and the (resized) resolution of the images.
If you encounter Out of Memory (OOM) errors, try reducing your training batch size.

Inference
=========

:ref:`Inference <inference>` in the multi-view setup follows the same general procedure as the
single-view case, with some important considerations.

Videos for all views must exist in the same directory.
For exmaple, if you are running inference on `sessionX` with two views, `view0` and `view1`, your
video data must be stored as:

.. code-block::

    /path/to/videos/
      ├── sessionX_view0.mp4
      ├── sessionX_view1.mp4
      └── ...

Lightning Pose produces a separate csv file with the predicted keypoints for each video, 
maintaining the same file structure as the input data organization.