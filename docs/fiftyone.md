# FiftyOne
You can visualize the predictions of one or multiple trained models, either on the `train/test/val` 
images or on test videos, using the [FifyOne](https://voxel51.com/) package. Add details about the app...

The first step is to create a `FiftyOne.Dataset` (i.e., a Mongo database pointing to images, keypoint predictions, names, and confidences.)
### Creating `FiftyOne.Dataset` for train/test/val predictions
```console
python scripts/create_fiftyone_dataset.py \
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
python scripts/create_fiftyone_dataset.py \
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
python scripts/create_fiftyone_dataset.py
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

