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

## Requirements
Your (potentially remote) machine has a Linux operating system, at least one GPU and **CUDA 11** installed. This 
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
foo@bar:~$ pip install -r requirements.txt
```

You should be ready to go! You may verify that all the unit tests are passing on your 
machine by running

```console
foo@bar:~$ pytest
```

## Working with `hydra`

For all of the scripts in our `scripts` folder, we rely on `hydra` to manage arguments in 
config files. You have two options: directly edit the config file, or override it from the command 
line.

* **Edit** the hydra config, that is, any of the parameters in `scripts/configs/config.yaml`, 
and save it. Then run the script without arguments, e.g.,:
```console
foo@bar:~$ python scripts/train_hydra.py
```

* **Override** the argument from the command line; for example, if you want to use a maximum of 11
epochs instead of the default number (not recommended):
```console
foo@bar:~$ python scripts/train_hydra.py training.max_epochs=11
```

### Important arguments
Below is a list of some commonly modified arguments related to model architecture/training. See
the file `scripts/configs/config_default.yaml` for a complete list of arguments and their defaults. 
When training a model on a new dataset, you must copy/paste this default config and update the 
arguments to match your data.
* training.train_batch_size (default: `16`) - batch size for labeled data
* training.min_epochs (default: `300`)
* training.max_epochs (default: `750`)
* model.model_type (default: `heatmap`)
    * regression: model directly outputs an (x, y) prediction for each keypoint; not recommended
    * heatmap: model outputs a 2D heatmap for each keypoint
    * heatmap_mhcrnn: the "multi-head convolutional RNN", this model takes a temporal window of 
    frames as input, and outputs two heatmaps: one "context-aware" and one "static". The prediction 
    with the highest confidence is automatically chosen. Must also set `model.do_context=True`.
* model.do_context (default: `False`) - set to `True` when using `model.model_type=heatmap_mhcrnn`.
* model.losses_to_use (default: `[]`) - this argument relates to the unsupervised losses. An empty 
list indicates a fully supervised model. Each element of the list corresponds to an unsupervised
loss. For example,
`model.losses_to_use=[pca_multiview,temporal]` will fit both a pca_multiview loss and a temporal 
loss. Options include:
    * pca_multiview: penalize inconsistencies between multiple camera views
    * pca_singleview: penalize implausible body configurations
    * temporal: penalize large temporal jumps

See the `losses` section of `scripts/configs/config_default.yaml` for more details on the various 
losses and their associated hyperparameters. The default values in the config file are reasonable 
for a range of datasets.

Some arguments related to video loading, both for semi-supervised models and when predicting new 
videos with any of the models:
* dali.base.train.sequence_length (default: `32`) - number of unlabeled frames per batch in 
`regression` and `heatmap` models (i.e. "base" models that do not use temporal context frames)
* dali.base.predict.sequence_length (default: `96`) - batch size when predicting on a new video with 
a "base" model
* dali.context.train.batch_size (default: `16`) - number of unlabeled frames per batch in 
`heatmap_mhcrnn` model (i.e. "context" models that utilize temporal context frames); each frame in 
this batch will be accompanied by context frames, so the true batch size will actually be larger 
than this number
* dali.context.predict.sequence_length (default: `96`) - batch size when predicting on a ndew video
with a "context" model

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
2. `eval.test_videos_directory`: path to a *folder* with new videos (not a single video)
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

1. Fiftyone: this component provides tools for visualizing the predictions of one or more trained
models on labeled frames or on test videos. See the documentation [here](docs/fiftyone.md). 
2. Streamlit: this component provides tools for quantifying model performance across a range of
metrics for both labeled frames and unlabeled videos:
    * Pixel error (labeled data only)
    * Temporal norm (unlabeled data only)
    * Pose PCA error (if `data.columns_for_singleview_pca` is not `null` in the config file)
    * Multi-view consistency error (if `data.mirrored_column_matches` is not `null` in the config 
    file)
See the documentation [here](docs/apps.md)
