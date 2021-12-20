![Wide Lightning Pose Logo](assets/images/LightningPose_horizontal_light.png)
Convolutional Networks for pose tracking implemented in **Pytorch Lightning**, supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.

### Built with the coolest Deep Learning packages
* `pytorch-lightning` for multiple-GPU training and to minimize boilerplate code
* `nvidia-DALI` for accelerated GPU dataloading
* `Hydra` to orchestrate the config files and log experiments
* `kornia` for differntiable computer vision ops
* `torchtyping` for type and shape assertions of `torch` tensors
* `FiftyOne` for visualizing model predictions
* `Tensorboard` to visually diagnoze training performance

## Required Hardware
Your (potentially remote) machine has at least one GPU and **CUDA 11** installed. This is a requirement for **NVIDIA DALI**. 

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

You should be ready to go! You may verify that all the unit tests are passing on your machine by running

```console
foo@bar:~$ pytest
```

## Datasets
* `BaseDataset`: images + keypoint coordinates.
* `HeatmapDataset`: images + heatmaps.
* `SemiSupervisedDataset`: images + sequences of unlabeled videos + heatmaps.

## Models 
* `RegressionTracker`: images -> labeled keypoint coordinates.
* `HeatmapTracker`: images -> labeled heatmaps.
* `SemiSupervisedHeatmapTracker`: images + sequences of unlabeled videos -> labeled heatmaps + unlabeled heatmaps. Supports multiple losses on the unlabeled videos.


## Training

The generic script for training models in our package is `scripts/train_hydra.py`.
The script relies on **Hydra** to manage arguments in hierarchical config files. You can run over an argument from the config file, for example, `training.max_epochs`, by calling

```console
foo@bar: ~$python scripts/train_hydra.py training.max_epochs=11
```

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs, will be saved at the `lightning-pose/outputs/YYYY-MM-DD/tb_logs` directory.

To view the logged losses with tensorboard, in the command line, run:

```console
foo@bar:~$ tensorboard --logdir outputs/YYYY-MM-DD/
```

where you use the date in which you ran the model.

## Visualize train/test/val predictions:

You can visualize the predictions of one or multiple trained models on the `train/test/val` images using the `FiftyOne` app.

You will need to specify:
1. `eval.hydra_paths`: path to trained models to use for prediction. 

Generally, using `Hydra` we can either edit the config `.yaml` files or override them from command line. 

### Option 1: Edit the config

Edit `scripts/configs/eval/eval_params.yaml` like so:
```
hydra_paths: [
"YYYY-MM-DD/HH-MM-SS/", "YYYY-MM-DD/HH-MM-SS/",
]
```
where you specify the relative paths for `hydra` folders within the `lightning-pose/outputs` folder. Then from command line, run:
```console
foo@bar:~$ python scripts/launch_diagnostics.py
```

### Option 2: Override from command line
Specify `hydra_paths` in the command line, overriding the `.yaml`:
```console
foo@bar:~$ python scripts/launch_diagnostics.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"]
``` 
where again, `hydra_paths` should be a list of strings with folder names within `lightning-pose/outputs`.

## Predict keypoints on new videos
With a trained model and a path to a new video, you can generate predictions for each frame and save it as a `.csv` or `.h5` file. 
To do so for the example dataset, run:

```console
foo@bar:~$ python scripts/predict_new_vids.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"]
```

using the same hydra path as before.

In order to use this script more generally, you need to specify several paths:
1. `eval.hydra_paths`: path to models to use for prediction: 
2. `eval.path_to_test_videos`: path to a *folder* with new videos (not a single video)
3. `path_to_save_predictions`: optional path specifying where to save prediction csv files. If `null`, the predictions will be saved in `eval.path_to_test_videos`.

As above, you could directly edit `scripts/configs/eval/eval_params.yaml` and run
```console
foo@bar:~$ python scripts/predict_new_vids.py 
```
or override these arguments in the command line.

```console
foo@bar:~$ python scripts/predict_new_vids.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"] eval.path_to_save_predictions="/path/to/your/file.csv" eval.path_to_test_videos="path/to/test/vids"
```

## Overlay predicted keypoints on new videos
With the pose predictions output by the previous step, you can now overlay these predictions on the video. 
To do so for the example dataset, run:

```console
foo@bar:~$ python scripts/render_labeled_vids.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"]
```

using the same hydra path as before.
