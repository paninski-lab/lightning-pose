# Cropzoom changes in this branch

## Usage example
```shell
# Train a regular model; this will act as a detector later.
litpose train chickadee.yaml --output_dir outputs/chickadee/single_0 --overrides training.train_frames=500

# Generate detector crop outputs.
litpose predict outputs/chickadee/single_0 CollectedData_merged.csv CollectedData_merged_new.csv --detector_mode

# Train a pose model using the above model's detector outputs. 
# This will internally modify config data_dir and csv_file. 
litpose train chickadee.yaml --output_dir outputs/chickadee/pose_0_1.8 --overrides training.train_frames=500 --detector_model=outputs/chickadee/single_0

# Run pose model on OOD data.
# Training has already run this on InD data, but it did not automatically
# do OoD data due to the _new file being in a different directory. 
litpose predict outputs/chickadee/pose_0_1.8 outputs/chickadee/single_0/image_preds/CollectedData_merged_new.csv/cropped_CollectedData_merged_new.csv

# Remap predictions. Currently manual, to be moved into `litpose predict`.
# Generates remapped_predictions.csv next to predictions.csv in pose model dir.
python remap_predictions.py

# Lastly: go to paw_ood_pixel_error_simple.ipynb and run the notebook.
# Note you may need to adjust the paths. I was using sshfs to map lightning studio `outputs` to local `fuse_outputs`.
```

## Limitations

1. Does not yet crop videos. We have some video cropping code in cropzoom but it has not been
   integrated into the rest of the code, and it does not yet do dynamic cropping (dynamic animal size).
2. Pose model prediction does not dynamically invoke detector model. It relies on detector artifacts
   already existing in detector model directory.
3. Pose PCA loss was not working for some reason - got a cryptic error. IIRC this was for the pose model.
   
## Major chanegs

1. Predictions.csv moved to within `image_preds`. Follows already documented structure:
   `image_preds/<csv_file_name>/predictions.csv`. `bbox.csv`, `cropped_<csv_file_name>` are
   siblings to predictions. This was to simplify the logic of looking up pose artifacts.
   For backwards compatibility, copy predictions.csv to its previous location.
2. Fix bug where context frames cropping was overwriting regular labeled frame cropping.
2. Dynamic crop images (dynamic animal size).
3. Make cropping multi-process.

## Jupyter notebooks

1. `paw_ood_pixel_error_simple.py`: Generates pixel error vs ensemble stddev plots. 
2. `cropped_predictions.ipynb`: Visualizes cropping, labels, predictions as skeletons.
   Generates an image gallery website which can be served using `python -m http.server`.

## Stash of workflow scripts

### Lightning job sweep.py script

Used this to run lightning jobs to train and evaluate ensembles. 

### Moving artifacts from lightning job to studio drive

```
cp -rn --no-preserve=ownership,mode /teamspace/jobs/single-2-train-and-ood-predict/artifacts/lightning-pose/outputs/chickadee/pose_2_1.8  outputs/chickadee
```

### sshfs to mount lightning drive locally

```
sudo sshfs -F ~/.ssh/config -v -o allow_other,default_permissions s_01j9ptzvcj2nfjjyzvjnm9nf6h@ssh.lightning.ai:/teamspace/studios/this_studio/lightning-pose/outputs fuse_outputs/
```

For cropped images, mount for debugging cropping only:
```
sudo sshfs -F ~/.ssh/config -v -o allow_other,default_permissions s_01j9ptzvcj2nfjjyzvjnm9nf6h@ssh.lightning.ai:/teamspace/jobs/single-0-train-and-ood-predict/artifacts/lightning-pose/outputs/chickadee/single_0/cropped_images fuse_outputs/chickadee/single_0/cropped_images
```

## Future thoughts

1. Consider a Virtual Model which adheres to some Model interface. 
* train
* predict_frames
* predict_video

However it generates its own model_dir separate from detector
and pose models. It stores predictions (and losses) in the original
coordinate space in `image_preds`, `video_preds`.

Details:
* A CropZoomModel consists of a detector model and a pose model.
* Training a CropZoomModel from scratch involves combining all of the
  steps we currently do.
* Prediction should remap coordinates, instead of that being a separate step
  as it is now.

2. Consider a better caching mechanism for detector artifacts.
Currently we assume cropped files to be at a location. There
should be an elegant way to ask for a cropped file that will
use the detector to generate it if it has not yet been generated.
Elegance would come from this cache abstracting away the implementation
detail of the files.



![](https://github.com/danbider/lightning-pose/raw/main/docs/images/LightningPose_horizontal_light.png)

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/danbider/lightning-pose)
[![Documentation Status](https://readthedocs.org/projects/lightning-pose/badge/?version=latest)](https://lightning-pose.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/lightning-pose)
![PyPI Downloads](https://static.pepy.tech/badge/lightning-pose/week)


Pose estimation models implemented in **Pytorch Lightning**, supporting massively accelerated training on _unlabeled_ videos using **NVIDIA DALI**. 
Models can be evaluated with **TensorBoard**, **FiftyOne**, and **Streamlit**.

As of June 2024, Lightning Pose is now [published in Nature Methods](https://rdcu.be/dLP3z)!

## Try our demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb)

Train a network on an example dataset and visualize the results in Google Colab.

## Getting Started
Please see the [Lightning Pose documentation](https://lightning-pose.readthedocs.io/) 
for installation instructions and user guides.
Note that the Lightning Pose package provides tools for training and evaluating models on 
_already labeled data_ and unlabeled video clips. 

We also offer a [browser-based application](https://github.com/Lightning-Universe/Pose-app) that 
supports the full life cycle of a pose estimation project, from data annotation to model training 
to diagnostic visualizations.

The Lightning Pose team also actively develops the 
[Ensemble Kalman Smoother (EKS)](https://github.com/paninski-lab/eks), 
a simple and performant post-processor that works with any pose estimation package including 
Lightning Pose, DeepLabCut, and SLEAP.

## Community

Lightning Pose is primarily maintained by 
[Dan Biderman](https://dan-biderman.netlify.app) (Columbia University) 
and 
[Matt Whiteway](https://themattinthehatt.github.io/) (Columbia University). 

Lightning Pose is under active development and we welcome community contributions.
Whether you want to implement some of your own ideas or help out with our [development roadmap](docs/roadmap.md), please get in touch with us on Discord (see contributing guidelines [here](CONTRIBUTING.md)). 
