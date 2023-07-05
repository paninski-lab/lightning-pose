# Additional documentation

Here we provide some additional documentation on the Lightning Pose repo. The first set of docs are
for users, and include information on some of the visualization tools inside the repo. The second
set of docs are intended for developers, and include information on how to add new losses and 
models.

## Docs for users
* [Data formats](directory_structures.md): how frames, videos, and labels need to be organized for 
use by Lightning Pose. 
Also see the example dataset provided in `lightning-pose/data/mirror-mouse-example`.
This page also describes the structure of the model directories output by Lightning Pose.

* [Fiftyone](fiftyone.md): plot model predictions overlaid on labeled frames or unlabeled videos

* [Streamlit](apps.md): plot metrics on labeled data (pixel errors, confidences, pca reprojection 
errors) and unlabeled data (confidences, pca reprojection errors, temporal norms)


## Docs for developers

* [How to add a loss](add_a_loss.md): 
where to store the loss code, 
how to make it visible to users,
how to provide hyperparameters,
and how to ensure the loss values are automatically logged during training and computed on labeled
and unlabeled data after training

* [How to add a model](add_a_model.md): 
where to store the model code,
how to make it visible to users, 
how to provide hyperparameters for model construction, 
and how to connect it to data loaders and losses

* [General contributing guidelines](contributing.md): pull requests, testing, linting, etc.
