# pose-estimation-nets
Scalable pose estimation based on **Pytorch-Lightning**, supporting training on massive unlabeled videos using **NVIDIA DALI**.
## Hardware
We assume that you are running on a machine that has at least one GPU and **CUDA 11** installed. This is a requirement for **NVIDIA DALI**.
## Installation
First create a Conda environment in which this package and its dependencies will be installed. 
As you would do for any other repository, first --

Create the environment:

```conda create --name pose-estimation-nets```

Activate the environment:

```conda activate pose-estimation-nets```

Move into the folder where you want to place the repository folder:

```cd <SOMEFOLDER>```

From within that folder, download a local version of the GitHub folder:

```git clone https://github.com/danbider/pose-estimation-nets.git```

Then move into the new package folder:

```cd pose-estimation-nets```

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
The script relies on the **Hydra** to manage arguments in hierarchical config files. You can run over an argument from the config file by calling

```python scripts/train_hydra.py training.max_epochs=11```.

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs, will be saved at the `outputs/YYYY-MM-DD/tb_logs` directory.

## Prediction and visualization

Visualizing the models' predictions on the `train/test/val` datasets is done using the `FiftyOne` app. To generate these predictions, run

```python scripts/predict_compare.py```

**TODO: current version of this script relies on manual paths and models, we should extend this to flexible checkpoints and model types.**




