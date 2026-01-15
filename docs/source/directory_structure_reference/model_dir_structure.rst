Model Directory Structure
==========================

.. code-block:: text

   /models/model_dir/
   ├── config.yaml
   |
   # Singleview
   ├── predictions.csv
   # Multiview
   ├── predictions_view0.csv
   ├── predictions_view1.csv
   |
   ├── tb_logs/ ...
   ├── video_preds/
   │   ├── session0_view0.mp4/
   │   |   └── predictions.csv
   │   └── session0_view1.mp4/
   └── image_preds/
       └── CollectedData.csv/
            └── predictions.csv


Detailed descriptions
-----------------------

* ``tb_logs/``: model weights
* ``video_preds/``: predictions and metrics from videos. The config field ``eval.test_videos_directory`` points to a directory of videos; if ``eval.predict_vids_after_training`` is set to ``true``, all videos in the indicated direcotry will be run through the model upon training completion and results stored here.
* ``video_preds/labeled_videos/``: labeled mp4s. The config field ``eval.test_videos_directory`` points to a directory of videos; if ``eval.save_vids_after_training`` is set to ``true``, all videos in the indicated direcotry will be run through the model upon training completion and results stored here.
* ``predictions.csv``: predictions on labeled data. The right-most column records the train/val/test split that each example belongs to.
* ``predictions_pixel_error.csv``: Euclidean distance between the predictions in ``predictions.csv`` and the labeled keypoints (in ``<YOUR_LABELED_FRAMES>.csv``) per keypoint and frame.

We also compute all unsupervised losses, where applicable, and store them
(per keypoint and frame) in the following csvs:

* ``predictions_pca_multiview_error.csv``: pca multiview reprojection error between predictions and labeled keypoints
* ``predictions_pca_singleview_error.csv``: pca singleview reprojection error between predictions and labeled keypoints

