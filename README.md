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

### Important configs
* ``

## Training

```console
foo@bar:~$ python scripts/train_hydra.py
```
In case your config file isn't at `lightning-pose/scripts/configs`, which is common if you have multiple projects.

```console
foo@bar:~$ python scripts/train_hydra.py \
  --config-path="<PATH/TO/YOUR/CONFIGS/DIR>" \
  --config-name="<CONFIG_NAME_.yaml>"
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

## `FiftyOne`
You can visualize the predictions of one or multiple trained models, either on the `train/test/val` 
images or on test videos. 
Add details about the app...
The first step is to create a `FiftyOne.Dataset` (i.e., a Mongo database pointing to images, keypoint predictions, names, and confidences.)
### Creating `FiftyOne.Dataset` for train/test/val predictions
```console
foo@bar:~$ python scripts/create_fiftyone_dataset.py \
eval.fiftyone.dataset_to_create="images" \
eval.fiftyone.dataset_name=<YOUR_DATASET_NAME> \
eval.fiftyone.build_speed="slow" \
eval.hydra_paths=["</ABSOLUTE/PATH/TO/HYDRA/DIR/1>", "</ABSOLUTE/PATH/TO/HYDRA/DIR/1>"] \
eval.fiftyone.model_display_names=["<NAME_FOR_MODEL_1>","<NAME_FOR_MODEL_2>"]
```
These arguments could also be edited and saved in the config files if needed.
Note that `eval.hydra_paths` are absolute paths to directories with trained models you want to use for prediction. Each directory contains a `predictions.csv` file. 
You can also use the relative form 
```
eval.hydra_paths: ["YYYY-MM-DD/HH-MM-SS/", "YYYY-MM-DD/HH-MM-SS/"]
```
which will look in the `lightning-pose/outputs` directory for these subdirectories.
You can choose meaningful display names for the models above using `eval.fiftyone.model_display_names`, e.g., 
```
eval.fiftyone.model_display_names: ["supervised", "temporal"]
```
The output of this command will include `Created FiftyOne dataset called: <NAME>.`. Use that name when you load the dataset from python.

### Launching the FiftyOne app
Open an `ipython` session from your terminal. 
```console
foo@bar:~$ ipython
```
Now in Python, we import fiftyone, load the dataset we created, and launch the app
```
In [1]: import fiftyone as fo
In [2]: dataset = fo.load_dataset("YOUR_DATASET_NAME") # loads an existing dataset created by scripts/create_fiftyone_dataset.py which prints its name
In [3]: session = fo.launch_app(dataset) # launches the app

# Do stuff in the App..., and click the bookmark when you finish

# Say you want to export images to disc after you've done some filtering in the app

In [4]: view = session.view # point just to the current view

# define a config file for style
In [5]: import fiftyone.utils.annotations as foua
In [6]: config = foua.DrawConfig(
        {
            "keypoints_size": 9, # can adjust this number after inspecting images
            "show_keypoints_names": False,
            "show_keypoints_labels": False,
            "show_keypoints_attr_names": False,
            "per_keypoints_label_colors": False,
        }
    )
In [7]: export_dir = "/ABSOLUTE/PATH/TO/DIR" # a directory where you want the images saved
In [8]: label_fields = ["YOUR_LABELED_FIELD_1", "YOUR_LABELED_FIELD_1", ... ] # "LABELS" in the app, i.e., model preds and/or ground truth data
In [9]: view.draw_labels(export_dir, label_fields=label_fields, config=config)
```

When you're in the app: the app will show `LABELS` (for images) or `FRAME LABELS` (for videos) on the left. Click the downward arrow next to it. It will drop down a menu which (if `eval.fiftyone.build_speed == "slow"`) will allow you to filter by `Labels` (keypoint names), or `Confidence`. When `eval.fiftyone_build_speed == "fast"`) we do not store `Labels` and `Confidence` information. Play around with these; a typical good threshold is `0.05-0.1` Once you're happy, you can click on the orange bookmark icon to save the filters you applied. Then from code, you can call `view = session.view` and proceed from there.


### Creating `FiftyOne.Dataset` for videos
```console
foo@bar:~$ python scripts/create_fiftyone_dataset.py \
eval.fiftyone.dataset_to_create="videos" \
eval.fiftyone.dataset_name=<YOUR_DATASET_NAME> \
eval.fiftyone.build_speed="slow" \
eval.hydra_paths=["</ABSOLUTE/PATH/TO/HYDRA/DIR/1>","</ABSOLUTE/PATH/TO/HYDRA/DIR/1>"] \
eval.fiftyone.model_display_names=["<NAME_FOR_MODEL_1>","<NAME_FOR_MODEL_2>"]
eval.test_videos_directory="</ABSOLUTE/PATH/TO/VIDEOS/DIR>" \
eval.video_file_to_plot="</ABSOLUTE/PATH/TO/VIDEO.mp4>" \
eval.pred_csv_files_to_plot=["</ABSOLUTE/PATH/TO/PREDS_1.csv>","</ABSOLUTE/PATH/TO/PREDS_2.csv>"]
```
Again, you may just edit the config and run
```console
foo@bar:~$ python scripts/create_fiftyone_dataset.py
```
**Note**: for videos longer than a few minutes, creating a detailed `FiftyOne.Dataset` may take a very long time. Until next releases, please implement your own visualization method for longer videos.

Now, open `ipython` and launch the app similarly to the above.
Note, that the app may complain about the video format, in which case, within ipython, reencode the video:
```
In [8]: fouv.reencode_video("</ABSOLUTE/PATH/TO/VIDEO.mp4>", output_path="</NEW/ABSOLUTE/PATH/TO/VIDEO.mp4>")
```
If you want to save a video to local directory,
# TODO -- verify
```
foua.draw_labeled_video(dataset[fo_video_class.video], outpath, config=config)
In [9]: view.draw_labels(export_dir, label_fields=label_fields, config=config)

```
### Filtering according to confidence and bodypart
```
import fiftyone as fo
from fiftyone import ViewField as F

def simul_filter(sample_collection, fields, confidence=None, labels=None):
    view = sample_collection.view()
    for field in fields:
        if labels is not None:
            view = view.filter_labels(field, F("label").is_in(labels))

        if confidence is not None:
            view = view.filter_labels(field, F("confidence") > confidence)

    return view

dataset = fo.load_dataset(...)

session = fo.launch_app(dataset)

session.view = simul_filter(dataset, ["long", "list", "of", "fields"], confidence=0.5, labels=["nose"])
```

