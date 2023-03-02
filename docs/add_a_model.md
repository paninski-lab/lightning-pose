# How to add a new model to Lightning Pose

To detail the steps that are necessary to fully incorporate a new model into the 
Lightning Pose package, we will first detail some of the underlying infrastructure, and then use 
the standard heatmap model as an example.


## The `BaseFeatureExtractor` class
This class (found in `lightning_pose/models/base.py`)
contains a backbone neural network that, as its name suggests, extracts features from 
images that can be used for downstream keypoint prediction. There is one key method in this class,
`get_representations`, which takes in a batch of images and outputs a batch of 
representations. Notably, this method can handle a variety of batch types:
* individual images for supervised training of a baseline model
* individual images and their context frames for  supervised training of a TCN model
* sequence of frames for video inference with a baseline model
* sequence of frames for video inference with a TCN model
See `BaseFeatureExtractor.get_representations` for more documentation on the input/output behavior
of this function.

Users can currently choose from different backbone architectures (EfficientNets and ResNets) 
and different initializations for the ResNet50 
(classification with ImageNet, pose estimation with AP10k, etc.).
If you would like to add a new backbone architecture (say, a transformer) or a new initialization 
(say, AP36k) to `BaseFeatureExtractor` these modifications will be immediately accessible to all
 downstream models (semi-supervised pose estimators, TCN, etc.)   

## The `BaseSupervisedTracker` class
This class (found in `lightning_pose/models/base.py`) 
inherits from `BaseFeatureExtractor`, and contains a mix of already-implemented methods
to assist with model training and several abstract methods which should be implemented by children
classes (i.e. your new model). For example, the `training_step` method (used by `Lightning` in 
their `Trainer` class) simply calls the `evaluate_labeled` method which performs three steps:
1. process the input batch and return pose features/heatmaps/coordinates 
(needs to be implemented by downstream models)
2. send these model outputs to the loss factory to compute and log losses (already implemented)
3. compute and and log root mean square error for monitoring during training (already implemented)
When implementing a new model, you must implement a method called `get_loss_inputs_labeled` that 
will perform step 1.
Note that the `BaseSupervisedTracker` does _not_ implement a "head" that transforms features into
pose predictions; that is what individual models will implement.

## The `SemiSupervisedTrackerMixin` object
[Note: you can skip this section on a first pass]
Finally, if you wish to implement a model that is compatible with any of the unsupervised losses
you will need to use the `SemiSupervisedTrackerMixin` (found in `lightning_pose/models/base.py`). 
This ``mixin'' is not a complete class on its own but rather implements a `training_step` that 
takes unlabeled data into account. 
This training step consists of two parts:
1. evaluate the network on a labeled batch
2. evaluate the network on an unlabeled batch
For step 1, we fall back to the `evaluate_labeled` method described above. For step 2, there is a
new method in this mixin called `evaluate_unlabeled` that goes through the same process of pushing
data through the model and computing and logging the losses.

## Implementing your own model
Now that we've covered some necessary background, let's look at the how to implement a fully 
supervised and semi-supervised heatmap tracker.

#### Fully supervised model
The `HeatmapTracker` class (found in `lightning_pose/models/heatmap_tracker.py`) inherits from 
`BaseSupervisedTracker`, giving it access to the base feature extractor and the training step
(as well as validation and test step methods). Our job is to implement a "head" network that takes
the features as input and outputs pose predictions (in this case through the use of heatmaps).
The `__init__` method should call the `__init__` method of the parent class 
(`BaseSupervisedTracker`), which will construct the backbone feature extractor.
You will need to implement one or more methods that construct the head, and initialize the head in
this `__init__` method. For example, `HeatmapTracker` contains methods that construct upsampling
layers that transform the features into 2D heatmaps, one per keypoint. This class also contains a
method that takes the soft argmax of each heatmap to produce an (x, y) coordinate for each 
keypoint.

> NOTE: if you want to use unsupervised losses in your model the transformation from features to 
(x, y) coordinate MUST be differentiable!

Another method that you must implement is `get_loss_inputs_labeled` (which will be called by the
`training_step` method of the parent class). For the `HeatmapTracker` class this method comprises 
two parts:
1. process batch through both feature extractor and head to get heatmaps
2. process heatmaps to get (x, y) coordinates

Importantly, this method must return a dict with a set of standard key names, which will be
used by downstream losses. Your model does not need to return all the keys listed below, but must 
return the keys used by the losses you choose.
* "heatmaps_targ": target (ground truth) heatmaps for each frame/keypoint in the batch
* "heatmaps_pred": predicted heatmaps
* "keypoints_targ": target (ground truth) coordinates
* "keypoints_targ": target (ground truth) coordinates
* "confidences": uncertainty estimate associated with each keypoint

The final method that you must implement is the `predict_step`; this tells the model how to 
transform a batch of frames into (x, y) coordinates (and optionally confidences). This method may
look exactly like `get_loss_inputs_labeled` with a slightly different return format; alternatively,
this method can implement non-differentiable operations to choose the final coordinates, such as a
hard argmax instead of the soft argmax required for training.

#### Semi-supervised model


## How to integrate your model into the Lightning Pose pipeline
Once you've implemented your model, the next step is to integrate it into the larger repo so that 
it can take advantage of the available training and evaluation infrastructure. We will describe 
this process from the outside in.

#### Step 1: update the yaml config file
#### Step 2:
#### Step 3:
