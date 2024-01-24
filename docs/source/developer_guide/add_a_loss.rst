###################
Add a loss
###################

The Lightning Pose package is organized to cleanly separate losses from other apsects of the code -
model building, data loading, training, etc. In order to add a new loss all you need to implement
is a new loss class in the file ``lightning_pose/losses/losses.py``. We will first detail the base
loss class, and then demonstrate how to add a new loss using the temporal loss as an example.


The ``Loss`` class
==================

The :class:`~lightning_pose.losses.losses.Loss` class contains the blueprint for all losses
(i.e. any loss you would like to implement will inherit from this class). We will go through the
relevant methods here at a high level, then discuss their implementation in the next section.

All Lightning Pose models have a method called ``get_loss_inputs_labeled`` which pushes a batch of
labeled data through the network and returns a data dictionary which can contain, for example, the
ground truth and predicted (x, y) coordinates, ground truth and predicted heatmaps, etc.
Semi-supervised models also have a method called ``get_loss_inputs_unlabeled`` that will return its
own data dictionary of model outputs on unlabeled video.
These dictionaries are then passed to a :class:`~lightning_pose.losses.factory.LossFactory` object,
which loops over all selected losses and,
for each loss, uses the labeled or unlabeled data dictionary as input to the loss's ``__call__``
method. The ``__call__`` method also takes a "stage" argument that will be externally set as
"train", "val", or "test", and is used for proper logging of the loss.

Each new loss must implement its own ``__call__`` method, and this should follow a standard recipe
using some or all of the methods below.

#. ``remove_nans()`` (supervised only): find NaNs in the labeled data which correspond to occluded keypoints with no labels, and drop these data from the batch.
#. ``compute_loss()``: actual loss computation on keypoints, heatmaps, features, etc.
#. ``rectify_epsilon()``: loss values below a given epsilon are set to zero
#. ``reduce_loss()``: take sum or mean over various dimensions of batch
#. ``log_loss()``: record the loss value and given stage (train/val/test) for this batch
#. return weighted scalar loss and log dictionary to loss factory

The :class:`~lightning_pose.losses.factory.LossFactory` will add the weighted scalar loss to the
overall loss and log the results.


Implementing a new loss
=======================

To implement a new loss you will need to subclass the :class:`~lightning_pose.losses.losses.Loss`
class; the most important methods to modify are ``__call__``, ``__init__``, and of course your own
implementation of ``compute_loss()``.

The ``__call__`` method
-----------------------

Because the models return their outputs via dictionaries that are shared across losses, each new
loss must use the same data argument names (note any given loss may only use a subset of these):

* ``keypoints_targ``: shape (batch, 2 * num_keypoints)
* ``keypoints_pred``: shape (batch, 2 * num_keypoints); predicted (x, y) coordinates in the original image space; PCA losses need to act on these predictions, rather than the augmented predictions which alter the correlations between body parts and views
* ``keypoints_pred_augmented``: shape (batch, 2 * num_keypoints); this contains predicted (x, y) coordinates that still contain geometric data augmentations (scaling, rotation, etc.); note that these match ``heatmaps_pred`` and ``heatmaps_targ``, which also still contain data augmentations
* ``heatmaps_targ``: shape (batch, num_keypoints, heatmap_height, heatmap_width)
* ``heatmaps_pred``: shape (batch, num_keypoints, heatmap_height, heatmap_width)
* ``confidences``: shape (batch, num_keypoints)
* ``stage``: this will be passed to ``log_loss()``
* kwargs: it is important to include this as it will take care of any entries in the data dictionary that are not directly used in the given loss

If we look at the ``__call__`` method of the :class:`~lightning_pose.losses.losses.TemporalLoss`
class, we see that it takes as data arguments ``keypoints_pred`` and ``confidences``.

.. code-block:: python

    def __call__(
        self,
        keypoints_pred: TensorType["batch", "two_x_num_keypoints"],
        confidences: TensorType["batch", "num_keypoints"] = None,
        stage: Optional[Literal["train", "val", "test"]] = None,
        **kwargs,
    ) -> Tuple[TensorType[()], List[dict]]:

        elementwise_loss = self.compute_loss(predictions=keypoints_pred)
        # do remove nans with loss to remove temporal difference values
        clean_loss = (
            self.remove_nans(loss=elementwise_loss, confidences=confidences)
            if confidences is not None
            else elementwise_loss
        )
        epsilon_insensitive_loss = self.rectify_epsilon(loss=clean_loss)
        scalar_loss = self.reduce_loss(epsilon_insensitive_loss, method="mean")
        logs = self.log_loss(loss=scalar_loss, stage=stage)
        return self.weight * scalar_loss, logs


We now walk through the steps outlined above for this specific example.

#. Because this loss is not computed on labeled data we skip step 1 above (nan removal).

#. We first compute the loss by calling ``self.compute_loss()`` and pass in the keypoints. Inspection of ``TemporalLoss.compute_loss()`` shows that this is where the temporal norm between successive timepoints is computed. These element-wise norms are returned to the ``__call__`` function. Next, we perform this loss's version of ``remove_nans``: if a string of predictions has low confidence, it may be because the keypoint in question is occluded or out of the frame. We do not want to penalize these low-confidence predictions, and therefore set the loss at these timepoints to zero.

#. Next, we send the loss (still stored on a frame-by-frame and keypoint-by-keypoint basis) to the ``rectify_epsilon`` method. Because each keypoint will move from one frame to the next under natural movement, we may want to only penalize temporal norms above a given threshold (i.e. epsilon).

#. Once rectification has occurred (if desired), the loss is finally sent to the ``reduce_loss`` method which will take the mean or sum over frames and keypoints.

#. The resulting scalar loss and "stage" argument are sent to the ``log_loss()`` method for proper recording.

#. Finally, we return the weighted loss and the logs. Note that the loss factory will automatically record the weighted loss as well, so that you will have access to both quantities in Tensorboard.

The ``__init__`` method
-----------------------

Now that we've seen the meat of the loss, let's look into the ``__init__`` method.

.. code-block:: python

    def __init__(
        self,
        data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
        epsilon: Union[float, List[float]] = 0.0,
        prob_threshold: float = 0.0,
        log_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(data_module=data_module, epsilon=epsilon, log_weight=log_weight)
        self.loss_name = "temporal"
        self.prob_threshold = torch.tensor(prob_threshold, dtype=torch.float, device=self.device)

The arguements here should mirror those of the base
:class:`~lightning_pose.losses.losses.Loss` class:

* ``data_module``: optional, and unused by many losses; for an example use-case, see the PCA losses
* ``epsilon``: for rectification; can be set to zero for no rectification
* ``prob_threshold``: specific to this loss
* ``log_weight``: hyperparameter that controls the weight of the loss in the final cost function

In the :class:`~lightning_pose.losses.losses.TemporalLoss` constructor, we see three simple
actions.

#. calling the ``__init__`` function of the parent class
#. defining the loss name as a string (this is used for logging)
#. setting the probability threshold that is used when we call ``remove_nans``

If your loss requires its own set of parameters - for example, the PCA losses contain the PCA
eigenvectors - these need to be defined in the loss's ``__init__`` function.

Integrating a new loss into the Lightning Pose pipeline
=======================================================

Once you've implemented a new loss, the next step is to integrate it into the larger repo so that
users can select it for training.

.. note::

    The current models only return keypoints and, if applicable, heatmaps and confidences.
    If you wish to construct a loss that acts on other representations created by the model,
    you will need to update the ``get_loss_inputs_labeled`` and ``get_loss_inputs_unlabeled``
    methods in the various models so that they include these data in their return dicts
    (and hence expose these quantities to the loss classes).

Step 1: update the config file
------------------------------

The default configuration file at ``lightning_pose/scripts/configs/config_default.yaml`` enumerates
all possible hyperparameters needed for building and training a model.
In particular, there is a field called ``losses`` under which you can add your new loss and its
associated hyperparameters.
For example, the temporal loss function entry looks like this:

.. code-block:: yaml

    losses:
      temporal:
        # weight in front of temporal loss
        log_weight: 5.0
        # for epsilon insensitive rectification (in pixels; diffs below this are not penalized)
        epsilon: 20.0
        # nan removal value (in prob; heatmaps with max prob values are removed)
        prob_threshold: 0.05

Some notes:

* the loss name in the config file (``temporal`` here) should match the ``self.loss_name`` string defined in your loss's ``__init__`` method
* ``log_weight`` is a standard field used by all losses
* ``epsilon`` is a standard field used by all losses; can be zero
* any other field under your loss name will be passed to your loss's ``__init__`` function as a key-value pair

Step 2: update ``get_loss_classes``
-----------------------------------
The first helper function you need to update is
:meth:`~lightning_pose.losses.losses.get_loss_classes`,
which creates a mapping from the loss name to the loss class.
Add your new loss to the dictionary.

Step 3: update ``get_loss_factories``
-------------------------------------
The next helper function you may need to update is
:meth:`~lightning_pose.utils.scripts.get_loss_factories`,
which creates the supervised and unsupervised loss factories from the config file.
If your loss requires parameters from other parts of the config file (such as image dimensions
from the ``data`` field) you can add those key-value pairs to the constructor input in the
``if/elif`` block (see other examples in that function).

Step 4: update ``compute_metrics`` (optional)
---------------------------------------------
The base training script ``scripts/train_hydra.py`` will automatically compute a set of metrics on
all labeled data and unlabeled videos upon training completion.
To add your new metric to this operation, you must update
:meth:`~lightning_pose.utils.scripts.compute_metrics`.
In that fucntion you will see how other metrics such as pixel error, temporal norm, and pca
reprojection errors are included.
This may require you to adapt your loss and include it in the :mod:`lightning_pose.metrics` module.

Step 5: ADD UNIT TESTS!
-----------------------
Not only is this good coding practice, it makes debugging your loss easier!
Make a new function in the file ``tests/losses/losses.py`` that follows the same pattern as the
other functions there.
Let's take the temporal loss as an example again; once you write the test you can run it from the
command line like so:

.. code-block:: console

    pytest tests/losses/test_losses.py::test_temporal_loss

Make sure to include as many corner-cases as possible in your test suite.

And that's it!
