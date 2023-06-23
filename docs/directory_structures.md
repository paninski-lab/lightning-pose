# Directory structures for Lightning Pose

## Data directory structure

Lightning Pose assumes the following project directory structure, as in the example dataset
provided in `lightning-pose/data/mirror-mouse-example`.

    /path/to/project/
      ├── <LABELED_DATA_DIR>/
      ├── <VIDEO_DIR>/
      └── <YOUR_LABELED_FRAMES>.csv

* `<YOUR_LABELED_FRAMES>.csv`: a table with keypoint labels (rows: frames; columns: keypoints). 
Note that this file can take any name, and needs to be specified in the config file under 
`data.csv_file`.

* `<LABELED_DATA_DIR>/`: contains images that correspond to the labels, and can include subdirectories.
The directory name, any subdirectory names, and image names are all flexible, as long as they are
consistent with the first column of `<YOUR_LABELED_FRAMES>.csv`.

* `<VIDEO_DIR>/`: when training semi-supervised models, the videos in this directory will be used 
for computing the unsupervised losses. This directory can take any name, and needs to be specified 
in the config file under `data.video_dir`.

## Converting DLC projects to Lightning Pose format
Once you have installed Lightning Pose, you can convert previous DLC projects into the proper 
Lightning Pose format by running the following script from the command line 
(make sure to activate the conda environment):
```console
python scripts/converters/dlc2lp.py --dlc_dir=/path/to/dlc_dir --lp_dir=/path/to/lp_dir
```
That's it! After this you will need to update your config file with the correct paths.

### Converting other projects to Lightning Pose format
Coming soon. If you have labeled data from other pose estimation packages (like SLEAP or DPK) and
would like to try out Lightning Pose, please 
[raise an issue](https://github.com/danbider/lightning-pose/issues).

## Model directory structure

If you train a model using our script `lightning-pose/scripts/train_hydra.py`, a directory will be
created with the following structure. The default is to save models in a directory called `outputs`
inside the Lightning Pose directory; to change this, update the config fields `hydra.run.dir` and
`hydra.sweep.dir` with absolute paths of your choosing.

    /path/to/models/YYYY-MM-DD/HH-MM-SS/
      ├── tb_logs/
      ├── video_preds/
      │   └── labeled_videos/
      ├── config.yaml
      ├── predictions.csv
      ├── predictions_pca_multiview_error.csv
      ├── predictions_pca_singleview_error.csv
      └── predictions_pixel_error.csv
      
* `tb_logs/`: model weights

* `video_preds/`: predictions and metrics from videos. 
The config field `eval.test_videos_directory` points to a directory of videos;
if `eval.predict_vids_after_training` is set to `true`, all videos in the indicated direcotry will
be run through the model upon training completion and results stored here.

* `video_preds/labeled_videos/`: labeled mp4s. 
The config field `eval.test_videos_directory` points to a directory of videos;
if `eval.save_vids_after_training` is set to `true`, all videos in the indicated direcotry will
be run through the model upon training completion and results stored here. 

* `predictions.csv`: predictions on labeled data

* `predictions_pixel_error.csv`: Euclidean distance between the predictions in `predictions.csv` 
and the labeled keypoints (in `<YOUR_LABELED_FRAMES>.csv`) per keypoint and frame.

We also compute all supervised losses, where applicable, and store them (per keypoint and frame) in
the following csvs:
* `predictions_pca_multiview_error.csv`: pca multiview reprojection error between predictions and
labeled keypoints

* `predictions_pca_singleview_error.csv`: pca singleview reprojection error between predictions and
labeled keypoints
