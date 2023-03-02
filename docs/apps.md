# Labeled Frame Diagnostics
You can analyze predictions of one or multiple trained models on the `train/test/val` images using this streamlit app.

The app creates plots for:
    - plot of a selected metric (e.g. pixel errors) for each model (bar/box/violin/etc)
    - scatterplot of a selected metric between two models

### Specifying models to compare
Users select an arbitrary number of model output folders and pass these through the command line to compute metrics for each of these models. Users can also specify names for each of these models using command line arguments. The two command line arguments taken by this streamlit app are `model_folders` and `model_names`. For each model that the user would like to compare, they would specify the model location through its outputs folder absolute path and pass this to model_folders. Optionally, the user can also specify a name for the model using the model_names argument.

The model_folder arguments require the ABSOLUTE paths to the model outputs folder. The lightning-pose output folder structure for a single model is typically structured as follows: `/path/to/lightning-pose/outputs/YYYY-MM-DD/HH-MM-SS`, where inside the last folder there are prediction csvs. The absolute path to the `HH-MM-SS` folder must be specified for metrics for the model to be properly outputted to the streamlit app.

### Running app from the command line for train/test/val metrics
Multiple models can be compared as follows, where the model folder for each model must be preceded by "--model_folders" and the name for each model to be displayed in the app must be preceded by "--model_names".
```console
foo@bar:~$ streamlit run /path/to/lightning_pose/apps/labeled_frame_diagnostics.py --
--model_folders=/absolute/path/to/model0/outputs --model_names=model0
--model_folders=/absolute/path/to/model1/outputs --model_names=model1
```
To compare more models simply add more inputs to --model_folders and optionally a corresponding input to --model_names. 


# Video Diagnostics
You can analyze predictions of one or multiple trained models on the video predictions outputted after training using this streamlit app.

The app creates plots for:
- time series/likelihoods of a selected keypoint (x or y coord) for each model
- boxplot/histogram of temporal norms for each model
- boxplot/histogram of multiview pca reprojection errors for each model

### Specifying models to compare
This operates identically to labeled_frame_diagnostics above. 

### Running app from the command line for train/test/val metrics
Multiple models can be compared as follows, where the model folder for each model must be preceded by "--model_folders" and the name for each model to be displayed in the app must be preceded by "--model_names".
```console
foo@bar:~$ streamlit run /path/to/lightning_pose/apps/video_diagnostics.py --
--model_folders=/absolute/path/to/model0/outputs --model_names=model0
--model_folders=/absolute/path/to/model1/outputs --model_names=model1
```
To compare more models simply add more inputs to --model_folders and optionally a corresponding input to --model_names. 
