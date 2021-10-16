# Lightning Pose
Convolutional Networks for pose tracking implemented in **Pytorch-Lightning**, supporting massively accelerated training on *unlabeled* videos using **NVIDIA DALI**.
## Hardware
Your (potentially remote) machine has at least one GPU and **CUDA 11** installed. This is a requirement for **NVIDIA DALI**.
## Installation
First create a Conda environment in which this package and its dependencies will be installed. 
As you would do for any other repository, first --

Create the environment:

```conda create --name <YOUR_ENVIRONMENT_NAME>```

Activate the environment:

```conda activate <YOUR_ENVIRONMENT_NAME>```

Move into the folder where you want to place the repository folder:

```cd <SOME_FOLDER>```

From within that folder, download a local version of the GitHub folder:

```git clone https://github.com/danbider/lightning-pose.git```

Then move into the new package folder:

```cd lightning-pose```

Install our package and its dependencies:

`pip install -r requirements.txt`

You should be ready to go! You may verify that all the unit tests are passing on your machine by running

```pytest```

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

```python scripts/train_hydra.py training.max_epochs=11```.

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs, will be saved at the `outputs/YYYY-MM-DD/tb_logs` directory.

To view the logged losses with tensorboard, in the command line, run:

```tensorboard --logdir outputs/YYYY-MM-DD/```

where you use the date in which you ran the model.

## Prediction and visualization

Visualize the models' predictions on the `train/test/val` datasets using the `FiftyOne` app: 

```python scripts/predict_compare.py```





