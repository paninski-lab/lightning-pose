###################
Inference
###################

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
