![](https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png)
Pose estimation models implemented in **Pytorch Lightning**, supporting massively accelerated training on _unlabeled_ videos using **NVIDIA DALI**. 
The whole process is orchestrated by **Hydra**. 
Models can be evaluated with **TensorBoard**, **FiftyOne**, and **Streamlit**.

Preprint: [Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling, and cloud-native open-source tools](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1)

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/danbider/lightning-pose)
![PyPI](https://img.shields.io/pypi/v/lightning-pose)

## Try our demo!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb)

Train a network on an example dataset and visualize the results in Google Colab.

## Community

Lightning Pose is primarily maintained by 
[Dan Biderman](https://dan-biderman.netlify.app) (Columbia University) 
and 
[Matt Whiteway](https://themattinthehatt.github.io/) (Columbia University). 
Come chat with us in Discord.

## Getting Started
This package provides tools for training and evaluating models on already labeled data and 
unlabeled video clips. 
See the [documentation](docs/directory_structures.md) for the data formats required by 
Lightning Pose (and how to convert a DeepLabCut dataset into a Lightning Pose dataset). 

We also offer a [browser-based application](https://github.com/Lightning-Universe/Pose-app) that 
supports the full life cycle of a pose estimation project, from data annotation to model training 
(with Lightning Pose) to diagnostics visualizations.

## Requirements

Your (potentially remote) machine has a Linux operating system, 
at least one GPU and **CUDA 11.0-12.x** installed. 
This is a requirement for **NVIDIA DALI**.

## Installation

First create a Conda environment in which this package and its dependencies will be installed.
```console
conda create --name <YOUR_ENVIRONMENT_NAME> python=3.8
```

and activate it:
```console
conda activate <YOUR_ENVIRONMENT_NAME>
```

Move into the folder where you want to place the repository folder, and then download it from GitHub:
```console
cd <SOME_FOLDER>
git clone https://github.com/danbider/lightning-pose.git
```

Then move into the newly-created repository folder:
```console
cd lightning-pose
```
and install dependencies using one of the lines below that suits your needs best:
* `pip install -e . `: basic installation, covers most use-cases (note the period!)
* `pip install -e .[dev] `: basic install + dev tools
* `pip install -e .[extra_models] `: basic install + tools for loading resnet-50 simclr weights
* `pip install -e .[dev,extra_models] `: install all available requirements

This installation might take between 3-10 minutes, depending on your machine and internet connection.

If you are using Ubuntu 22.04 or newer, you'll need an additional update for the Fiftyone package:
```console
pip install fiftyone-db-ubuntu2204
```

Now you should be ready to go! You may verify that all the unit tests are passing on your
machine by running
```console
pytest
```

## Docker users

Use the appropriate Dockerfiles in this directory to build a Docker image:
```console
docker build -f Dockerfile.cuda11 -t my-image:cuda11 .
```

```console
docker build -f Dockerfile.cuda12 -t my-image:cuda12 .
```

Run code inside a container (following [this tutorial](https://docs.docker.com/get-started/)):
```console
docker run -it --rm --gpus all my-image:cuda11
```

```console
docker run -it --rm --gpus all --shm-size 256m my-image:cuda12
```

For a g4dn.xlarge AWS EC2 instance adding the flag `--shm-size=256m` will provide the necessary 
memory to execute. 
The '--gpus all' flag is necessary to allow Docker to access the required drivers for Nvidia DALI 
to work properly.

## Training

To train a model on the example dataset provided with this repo, run the following command:
```console
python scripts/train_hydra.py
```

To train a model on your own dataset, follow these steps:
1. ensure your data is in the [proper data format](docs/directory_structures.md)
2. copy the file `scripts/configs/config_default.yaml` to another directory and rename it. 
You will then need to update the various fields to match your dataset (such as image height and width). 
See other config files in `scripts/configs/` for examples.
3. train your model from the terminal and overwrite the config path and config name with your newly 
created file:
```console
python scripts/train_hydra.py --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" --config-name="<CONFIG_NAME.yaml>"
```

You can find more information on the structure of the model directories 
[here](docs/directory_structures.md#model-directory-structure).

## Working with `hydra`

For all of the scripts in our `scripts` folder, we rely on `hydra` to manage arguments in
config files. You have two options: directly edit the config file, or override it from the command
line.

- **Edit** the hydra config, that is, any of the parameters in, e.g., 
`scripts/configs/config_mirror-mouse-example.yaml`, and save it. 
Then run the script without arguments:
```console
python scripts/train_hydra.py
```

- **Override** the argument from the command line; for example, if you want to use a maximum of 11
  epochs instead of the default number (not recommended):
```console
python scripts/train_hydra.py training.max_epochs=11
```

Or, for your own dataset,
```console
python scripts/train_hydra.py --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" --config-name="<CONFIG_NAME.yaml> training.max_epochs=11
```

We also recommend trying out training with automatic resizing to smaller images first; 
this allows for larger batch sizes/fewer Out Of Memory errors on the GPU:
```console
python scripts/train_hydra.py --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" --config-name="<CONFIG_NAME.yaml> data.image_resize_dims.height=256 data.image_resize_dims.width=256
```

See more documentation on the config file fields [here](docs/config.md).

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs,
will be saved at the `lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS/tb_logs` directory. (Note: this
behavior can be changed by updating `hydra.run.dir` in the config yaml to an absolute path of your
choosing.)

To view the logged losses with tensorboard in your browser, in the command line, run:
```console
tensorboard --logdir outputs/YYYY-MM-DD/
```
where you use the date in which you ran the model. 
Click on the provided link in the terminal, which will look something like `http://localhost:6006/`.
Note that if you save the model at a different directory, just use that directory after `--logdir`.

## Predict keypoints on new videos

With a trained model and a path to a new video, you can generate predictions for each frame and 
save it as a `.csv` file.
To do so for the example dataset, run:
```console
python scripts/predict_new_vids.py eval.hydra_paths=["YYYY-MM-DD/HH-MM-SS/"]
```
using the same hydra path as before.

In order to use this script more generally, you need to specify several paths:
1. `eval.hydra_paths`: path to models to use for prediction
2. `eval.test_videos_directory`: path to a _folder_ with new videos (not a single video)
3. `eval.saved_vid_preds_dir`: optional path specifying where to save prediction csv files. If `null`, the predictions will be saved in `eval.test_videos_directory`.

As above, you could directly edit `scripts/configs/config_toy-dataset.yaml` and run
```console
python scripts/predict_new_vids.py
```

or override these arguments in the command line:
```console
scripts/predict_new_vids.py eval.hydra_paths=["2022-01-18/01-03-45"] \
eval.test_videos_directory="/absolute/path/to/videos" \
eval.saved_vid_preds_dir="/absolute/path/to/dir"
```

## Diagnostics

Beyond providing access to loss values throughout training with Tensorboard, the Lightning Pose
package also offers several diagnostic tools to compare the performance of trained models on
labeled frames and unlabeled videos.

##### Fiftyone

This component provides tools for visualizing the predictions of one or more trained
models on labeled frames or on test videos.

See the documentation [here](docs/fiftyone.md).

##### Streamlit

This component provides tools for quantifying model performance across a range of
metrics for both labeled frames and unlabeled videos:

- Pixel error (labeled data only)
- Temporal norm (unlabeled data only)
- Pose PCA error (if `data.columns_for_singleview_pca` is not `null` in the config file)
- Multi-view consistency error (if `data.mirrored_column_matches` is not `null` in the config
  file)

See the documentation [here](docs/apps.md).
