# Labeled Frame Diagnostics

Analyze predictions of one or more networks on the `train/test/val` images.
From within `lightning-pose/lightning_pose/apps`, run:
```
streamlit run labeled_frame_diagnostics.py -- --model_dir <ABOLUTE_PATH_TO_HYDRA_OUTPUTS_DIRECTORY>
```
The only argument needed is `--model_dir`, which tells the app where to find models and their predictions. `<ABOLUTE_PATH_TO_HYDRA_OUTPUTS_DIRECTORY>` should contain hydra subfolders of the type `YYYY-MM-DD/HH-MM-SS`.

(The lightning-pose output folder for a single model is typically: `/path/to/lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS`, where inside the last folder there are prediction `.csv`s.)

The app shows:
- plot of a selected metric (e.g. pixel errors, confidences) for each network and each body part, using bar/box/violin/etc plots.
- scatterplot of a selected metric between two networks

# Video Diagnostics
Visualizes multiple networks' predictions on a test video.
From within `lightning-pose/lightning_pose/apps`, run:
```
streamlit run video_diagnostics.py -- --model_dir <ABOLUTE_PATH_TO_HYDRA_OUTPUTS_DIRECTORY>
```
where `--model_dir` is explained above.
The app shows:
- timeseries of predictions/confidences/losses of a selected keypoint (x and y coord) for each network
- boxplot/histogram of confidences/losses for each network and each body part