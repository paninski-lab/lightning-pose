.. _lightning_pose_api:

##################
Lightning Pose API
##################


Train function
==============

.. autofunction:: lightning_pose.train.train

To train a model using ``config.yaml`` and output to ``outputs/doc_model``:
    .. code-block:: python

        import os
        from lightning_pose.train import train
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("config.yaml")
        os.chdir("outputs/doc_model")
        train(cfg)

To override settings before training:
    .. code-block:: python

        cfg = OmegaConf.load("config.yaml")
        overrides = {
            "training": {
                "min_epochs": 5,
                "max_epochs": 5
            }
        }
        cfg = OmegaConf.merge(cfg, overrides)
        train(cfg)

Training returns a Model object, which is described next.

Model class
===========

The ``Model`` class provides an easy-to-use interface to a lightning-pose
model. It supports running inference and accessing model metadata.
The set of supported Model operations will expand as we continue development.

You create a model object using `Model.from_dir`:

.. code-block:: python

    from lightning_pose.api.model import Model

    model = Model.from_dir("outputs/doc_model")

Then, to predict on new data:

.. code-block:: python

    model.predict_on_video_file("path/to/video.mp4")

or:

.. code-block:: python

    model.predict_on_label_csv("path/to/csv_file.csv")

To predict on a single numpy frame (no file I/O):

.. code-block:: python

    import numpy as np

    frame = np.array(...)  # (H, W, 3) uint8 RGB
    result = model.predict_frame(frame)
    keypoints = result["keypoints"]   # (num_kp, 2) float32
    confidence = result["confidence"] # (num_kp,) float32

API Reference
=============

.. autoclass:: lightning_pose.api.model.Model
    :members:
    :exclude-members: __init__, from_dir2

Return types
------------

.. autoclass:: lightning_pose.data.datatypes.PredictionResult
    :members:
    :undoc-members:
    :exclude-members: __init__

.. autoclass:: lightning_pose.data.datatypes.MultiviewPredictionResult
    :members:
    :undoc-members:
    :exclude-members: __init__

.. autoclass:: lightning_pose.data.datatypes.ComputeMetricsSingleResult
    :members:
    :undoc-members:
    :exclude-members: __init__


Lightning Pose Internal API
===========================

* :ref:`metrics and callbacks modules <lp_modules>`
* :ref:`data package <lp_modules_data>`
* :ref:`losses package <lp_modules_losses>`
* :ref:`models package <lp_modules_models>`
* :ref:`utils package <lp_modules_utils>`

..
    Comment: this is the standard way of listing the API components, but I don't like the
    way it adds extra elements to the landing page, so doing the simpler version above now.

    .. toctree::
       :maxdepth: 1

       ../modules/lightning_pose
       ../modules/lightning_pose.data
       ../modules/lightning_pose.losses
       ../modules/lightning_pose.models
       ../modules/lightning_pose.utils
