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
supervised and semi-supervised heatmap tracker. The easiest way to parse this information is to 
open up the file `lightning_pose/models/heatmap_tracker.py` and follow along with the text below to
see how these ideas are implemented in practice. 

A second example can be found in 
`lightning_pose/models/heatmap_tracker_mhcrnn.py`, which implements the fully- and semi-supervised 
versions of the temporal context network. 

Finally, a third example can be found in `lightning_pose/models/regression_tracker.py`, which 
implements a fully- and semi-supervised trackers that omit heatmaps and directly predict (x, y)
coordinates. 

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
The `SemiSupervisedHeatmapTracker` class (found in `lightning_pose/models/heatmap_tracker.py`) 
inherits from _both_ `HeatmapTracker` - which gives it access to the feature extractor, heatmap
prediction head, and labeled data loss computation - and `SemiSupervisedTrackerMixin` - which 
gives it access to unlabeled data loss computation. This class (as well as any other class you
want to build that uses unsupervised losses) must implement two methods. The first is the 
`__init__` method, which should call the `__init__` method of the parent class(es), and also builds 
the unsupervised losses themselves. 
This is fully taken care of by the loss factory object, which must be an input to
the semi-supervised model's constructor (more info on this below). 
The second method is called `get_loss_inputs_unlabeled` and, like its sibling function 
`get_loss_inputs_labeled` in the fully supervised model, is responsible for two steps:
1. process batch through both feature extractor and head to get heatmaps
2. process heatmaps to get (x, y) coordinates

As above, this method will return a dict with a set of standard key names, which will not
include target heatmaps or keypoints since those are formed from labeled data and we are dealing 
exclusively with unlabeled data in this method.

You do _not_ need to implement a `predict_step` method since the "semi-supervised" aspect of this
model only affects training and not inference; therefore during inference the `predict_step` of the
fully supervised model will be used.

## How to integrate your model into the Lightning Pose pipeline
Once you've implemented your model, the next step is to integrate it into the larger repo so that 
it can take advantage of the available training and evaluation infrastructure. We will describe 
this process from the outside in.

#### Step 1: update the yaml config file
The default configuration file at `lightning_pose/scripts/configs/config_default.yaml` enumerates 
all possible hyperparameters needed for building and training a model. If your new model requires
additional hyperparameters that you wish to control externally, include these in the config file.
Inside the pipeline, when initializing the model, you will have access to every key-value pair in
this file.

There is a field `model.model_type` which you can use to specify your model - the current supported
values are "regression", "heatmap", and "heatmap_mhcrnn". Add your new model name to this list. If 
your model requires context frames, ensure that you also set `model.do_context: true`, which will
build a data generator that serves context frames to the model with both labeled and, if needed,
unlabeled data.

The basic training script can be found at `scripts/train_hydra.py`. You do not
need to update anything in this script to accommodate your new model, but this script uses several
helper functions that we will update next.
 
#### Step 2: update `get_dataset`
The first helper function you need to update is `lightning_pose.utils.scripts.get_dataset`, 
which creates a torch Dataset object associated with your model. 
For example, the regression-based models do not need a dataset that returns heatmaps, whereas the 
heatmap-based models do. 
In this function you will see the `if/else` statement that creates a dataset based on the model 
type; include your model in this `if/else` statement.

#### Step 3: update `get_loss_factories`
If your model requires heatmaps for training, in order to ensure the heatmap losses are properly 
logged you need to add your model to the first `if/else` statement in the function
`lightning_pose.utils.scripts.get_loss_factories` (you will see "heatmap" and "heatmap_mhcrnn" 
models already represented there). 
Note that if your model uses heatmaps you will also be able to select from several heatmap losses 
in the config file using the `model.heatmap_loss_type` key.

#### Step 4: update `get_model`
This next helper function - `lightning_pose.utils.scripts.get_model` - is what translates the 
key-value pairs from the config file to constructing the actual model. 
You will see examples of all other models in this function; include your model accordingly.

#### Step 5: update `get_model_class`
Finally, there is helper function `lightning_pose.utils.predictions.get_model_class` which is used
to seamlessly load model parameters from checkpoint files. Again, there are various `if/else`
statements where your model should be incorporated.

#### Step 6: optional and miscellanious additons
* if you find yourself needing to write a new DALI dataloader to support your model training, you
might also need to update `lightning_pose.utils.predictions.PredictionHandler`
* if your model uses heatmaps and you would like to save out heatmaps for each keypoint/frame when
running inference on a new video, you'll need to update the legacy function 
`lightning_pose.utils.predictions._predict_frames`, which will be called by the function
`lightning_pose.utils.predictions.predict_single_video` when using `save_heatmaps=True`

#### Step 7: ADD UNIT TESTS!
Not only is this good coding practice, it makes debugging your model easier! Make a new file in the 
directory `tests/models` that follow the same pattern as the other files there. We provide many 
convenience functions that allow you to set up units tests for fully supervised models, context 
models, and semi-supervised models (and combinations thereof). 
Let's take the fully-supervised heatmap model as an example; once you write the test you can run it
from the command line like so:
```bash
pytest tests/models/test_heatmap_tracker.py::test_supervised_heatmap
```
This test will build your model using the helper functions above (like `get_model`) and train it
for several epochs using the toy dataset that comes packaged with this repo.


And that's it!
