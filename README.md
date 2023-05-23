![](https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png)
Pose estimation models implemented in **Pytorch Lightning**,
supporting massively accelerated training on _unlabeled_ videos using **NVIDIA DALI**. The whole thing is orchestrated by **Hydra**. Models can be diagnosed with **TensorBoard**, **FiftyOne**, and **Streamlit**.

Preprint: [Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling, and cloud-native open-source tools](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1)

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/danbider/lightning-pose)
![PyPI](https://img.shields.io/pypi/v/lightning-pose)

## Try our demo!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb)

Train a network on a toy dataset and visualize the results in Google Colab.

## Community

Lightning Pose is primarily maintained by [Dan Biderman](https://dan-biderman.netlify.app) (Columbia University) and [Matt Whiteway](https://themattinthehatt.github.io/) (Columbia University). Come chat with us in Discord.

## Requirements

Your (potentially remote) machine has a Linux operating system, at least one GPU and **CUDA 11.0-12.x** installed. This
is a requirement for **NVIDIA DALI**.

## Installation

First create a Conda environment in which this package and its dependencies will be installed.
As you would do for any other repository --

Create a conda environment:

```console
foo@bar:~$ conda create --name <YOUR_ENVIRONMENT_NAME> python=3.8
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
foo@bar:~$ pip install -e .
```

You should be ready to go! You may verify that all the unit tests are passing on your
machine by running

```console
foo@bar:~$ pytest
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

For a g4dn.xlarge AWS EC2 instance adding the flag `--shm-size=256m` will provide the necessary memory to execute. The '--gpus all' flag is necessary to allow Docker to access the required drivers for Nvidia DALI to work properly. 

## Working with `hydra`

For all of the scripts in our `scripts` folder, we rely on `hydra` to manage arguments in
config files. You have two options: directly edit the config file, or override it from the command
line.

- **Edit** the hydra config, that is, any of the parameters in `scripts/configs/config.yaml`,
  and save it. Then run the script without arguments, e.g.,:

```console
foo@bar:~$ python scripts/train_hydra.py
```

- **Override** the argument from the command line; for example, if you want to use a maximum of 11
  epochs instead of the default number (not recommended):

```console
foo@bar:~$ python scripts/train_hydra.py training.max_epochs=11
```

See more documentation on the arguments [here](docs/config.md).

## Training

```console
foo@bar:~$ python scripts/train_hydra.py
```

In case your config file isn't located in `lightning-pose/scripts/configs`, which is common if you
have multiple projects, run:

```console
foo@bar:~$ python scripts/train_hydra.py \
  --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" \
  --config-name="<CONFIG_NAME.yaml>"
```

## Logs and saved models

The outputs of the training script, namely the model checkpoints and `Tensorboard` logs,
will be saved at the `lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS/tb_logs` directory. (Note: this
behavior can be changed by updating `hydra.run.dir` in the config yaml to an absolute path of your
choosing.)

To view the logged losses with tensorboard in your browser, in the command line, run:

```console
foo@bar:~$ tensorboard --logdir outputs/YYYY-MM-DD/
```

where you use the date in which you ran the model. Click on the provided link in the
terminal, which will look something like `http://localhost:6006/`.
Note that if you save the model at a different directory, just use that directory after `--logdir`.

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
2. `eval.test_videos_directory`: path to a _folder_ with new videos (not a single video)
3. `eval.saved_vid_preds_dir`: optional path specifying where to save prediction csv files. If `null`, the predictions will be saved in `eval.test_videos_directory`.

As above, you could directly edit `scripts/configs/config.yaml` and run

```console
foo@bar:~$ python scripts/predict_new_vids.py
```

or override these arguments in the command line.

```console
foo@bar:~$ python scripts/predict_new_vids.py eval.hydra_paths=["2022-01-18/01-03-45"] \
eval.test_videos_directory="/absolute/path/to/unlabeled_videos" \
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
