![Wide Lightning Pose Logo](assets/images/LightningPose_horizontal_light.png)
Convolutional Networks for pose tracking implemented in **Pytorch Lightning**, 
supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.

### Built with the coolest Deep Learning packages
* `pytorch-lightning` for multiple-GPU training and to minimize boilerplate code
* `nvidia-DALI` for accelerated GPU dataloading
* `Hydra` to orchestrate the config files and log experiments
* `kornia` for differntiable computer vision ops
* `torchtyping` for type and shape assertions of `torch` tensors
* `FiftyOne` for visualizing model predictions
* `Tensorboard` to visually diagnoze training performance

## Required Hardware
Your (potentially remote) machine has at least one GPU and **CUDA 11** installed. This 
is a requirement for **NVIDIA DALI**. 

Provide more GPUs and we will use them.

## Installation

First create a Conda environment in which this package and its dependencies will be installed. 
As you would do for any other repository --

Create a conda environment:

```console 
foo@bar:~$ conda create --name <YOUR_ENVIRONMENT_NAME>
```

and activate it:

```console
foo@bar:~$ conda activate <YOUR_ENVIRONMENT_NAME>
```

Move into the folder where you want to place the repository folder, and then download it from GitHub:

```console
foo@bar:~$ cd <SOME_FOLDER>
foo@bar:~$ git clone https://github.com/danbider/lightning-pose.git
```

Then move into the newly-created repository folder, and install dependencies:

```console
foo@bar:~$ cd lightning-pose
foo@bar:~$ pip install -r requirements.txt
```

You should be ready to go! You may verify that all the unit tests are passing on your 
machine by running

```console
foo@bar:~$ pytest
```

## Datasets
NEEDS UPDATE
* `BaseDataset`: images + keypoint coordinates.
* `HeatmapDataset`: images + heatmaps.
* `SemiSupervisedDataset`: images + sequences of unlabeled videos + heatmaps.

## Models 
NEEDS UPDATE
* `RegressionTracker`: images -> labeled keypoint coordinates.
* `HeatmapTracker`: images -> labeled heatmaps.
* `SemiSupervisedHeatmapTracker`: images + sequences of unlabeled videos -> labeled heatmaps + unlabeled heatmaps. Supports multiple losses on the unlabeled videos.

## Working with `hydra`

For all of the scripts in our `scripts` folder, we rely on `hydra` to manage arguments in hierarchical config files. You have two options: edit the config file, or override it from the command line.

* **Edit** a hydra config, that is, any of the files in `scripts/configs/config_folder/config_name.yaml`, and save it. Then run the script without arguments, e.g.,:
```console
foo@bar:~$ python scripts/train_hydra.py
```

* **Override** the argument from the command line:
```console
foo@bar:~$ python scripts/train_hydra.py training.max_epochs=11
```
If you happen to want to use a maximum of 11 epochs instead the default number (not recommended).

## Training

```console
foo@bar:~$ python scripts/train_hydra.py
```

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs, 
will be saved at the `lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS/tb_logs` directory.

To view the logged losses with tensorboard in your browser, in the command line, run:

```console
foo@bar:~$ tensorboard --logdir outputs/YYYY-MM-DD/
```

where you use the date in which you ran the model. Click on the provided link in the
terminal, which will look something like `http://localhost:6006/`.
Note that if you save the model at a different directory, just use that directory after `--logdir`.

## Visualize train/test/val predictions

You can visualize the predictions of one or multiple trained models on the `train/test/val` 
images using the `FiftyOne` app.

You will need to specify:
1. `eval.hydra_paths`: path to trained models to use for prediction. 

Generally, using `Hydra` we can either edit the config `.yaml` files or override them 
from the command line. The argument of ineterest is `

### Option 1: Edit the config

Edit `scripts/configs/eval/eval_params.yaml` like so:
```
hydra_paths: [
"YYYY-MM-DD/HH-MM-SS/", "YYYY-MM-DD/HH-MM-SS/",
]
```
where you specify the relative paths for `hydra` folders within the `lightning-pose/outputs` folder. 
Then from command line, run:
```console
foo@bar:~$ python scripts/launch_diagnostics.py
```
Alternatively, override from the command line:
```console
foo@bar:~$ python scripts/launch_diagnostics.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"] \
data.data_dir='/absolute/path/to/data_dir' \
data.video_dir='/absolute/path/to/video_dir' 
``` 
As with `Tensorboard`, click on the link provided in the terminal to launch the diagnostics
in your browser.

### FiftyOne app
The app will open and will show `FRAME LABELS` to the left. Click the `v` next to it. It will drop down a menu with keypoint names, and a filter with confidence. Play around with these. Once you're happy, you can click on the orange bookmark icon to save the filters you applied. Then from code, you can call `session.view.export(...)`. NOTE: not yet supported.

## Predict keypoints on new videos
With a trained model and a path to a new video, you can generate predictions for each 
frame and save it as a `.csv` or `.h5` file. 
To do so for the example dataset, run:

```console
foo@bar:~$ python scripts/predict_new_vids.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"]
```

using the same hydra path as before.

In order to use this script more generally, you need to specify several paths:
1. `eval.hydra_paths`: path to models to use for prediction
2. `eval.test_videos_directory`: path to a *folder* with new videos (not a single video)
3. `eval.saved_vid_preds_dir`: optional path specifying where to save prediction csv files. If `null`, the predictions will be saved in `eval.test_videos_directory`.

As above, you could directly edit `scripts/configs/eval/eval_params.yaml` and run
```console
foo@bar:~$ python scripts/predict_new_vids.py 
```
or override these arguments in the command line.

```console
foo@bar:~$ python scripts/predict_new_vids.py eval.hydra_paths=["2022-01-18/01-03-45"] \
eval.test_videos_directory="/absolute/path/to/unlabeled_videos" \
eval.saved_vid_preds_dir="/absolute/path/to/dir"
```

## Overlay predicted keypoints on new videos
With the pose predictions output by the previous step, you can now overlay these 
predictions on the video. 
To do so for the example dataset, run:

```console
foo@bar:~$ python scripts/render_labeled_vids.py eval.model_display_names=["test_model"] \
data.data_dir="/absolute/path/to/data_dir" \
eval.video_file_to_plot="/absolute/path/to/vid.mp4" \
eval.pred_csv_files_to_plot:["/absolute/path/1/to.csv", "/absolute/path/1/to.csv"]

```

using the same hydra path as before. This script will by default save a labeled video in the same directory as the video it analyzes, and will also launch the fiftyone app to further explore the video(s) in the browser. 

