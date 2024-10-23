.. _multi_gpu_training:

###################
Multi-GPU Training
###################

Multi-GPU training allows you to distribute the load of model training across GPUs.
This helps overcome OOMs in addition to accelerating training.

To use this feature, set :ref:`num_gpus <config_num_gpus>` in your config file.

How to choose batch_size
========================

Multi-GPU training distributes batches across multiple GPUs in a way that maintains the same
effective batch size as if you ran on 1 GPU. **Thus, if you reduced batch size in order to make
your model fit in one GPU, you should increase it back to your desired effective batch size.**

The batch size configuration parameters that this applies to are ``training.train_batch_size`` and
``training.val_batch_size`` for the labeled frames, and ``dali.train.base.sequence_length`` and
``dali.train.context.batch_size`` for unlabeled video frames. Test batch sizes are not relevant
to this document as testing only occurs on one GPU.

Calculate of per-GPU batch size
-------------------------------

Given the above, you need not worry about how lightning-pose calculates per-GPU batch size,
but it is documented here for transparency.

For the regular ``heatmap`` model, the per-GPU batch size will be:

.. code-block:: python

    ceil(batch_size / num_gpus)

For the context ``heatmap_mhcrnn`` model, the per-GPU batch size will be: 

.. code-block:: python

    ceil((batch_size - 4) / num_gpus) + 4

The calculation for ``heatmap_mhcrnn`` maintains the same effective batch size by accounting for
the 2 context frames that are loaded with each training frame. For example, if you specify a batch
size of 16, then your effective batch size was 16 - 4 = 12. To maintain 12 with 2 GPUs, 
each GPU will load 6 frames + 4 context frames, for a per-GPU batch size of 10. This is larger than
simply dividing the original batch size of 16 across 2 GPUs.

.. _execution_model:

Execution model
===============

.. warning::
    The implementation spawns ``num_gpus - 1`` processes of the same command originally executed,
    repeating all of the command's execution per process.
    Thus it is advised to only run multi-GPU training in a dedicated training script
    (``scripts/train_hydra.py``). If you use lightning-pose as part of a custom script and don't
    want your entire script to run once per GPU, your script should run ``scripts/train_hydra.py``
    rather than directly calling the ``train`` method.

Tensorboard metric calculation
==============================

All metrics can be interpreted the same way as with a single-GPU.
The metrics are the average value across the GPUs. 

Specifying the GPUs to run on
=============================

Use the environment variable ``CUDA_VISIBLE_DEVICES`` if you want lightning pose to run on certain
GPUs. For example, if you want to train on only the first two GPUs on your machine,

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1 python scripts/train_hydra.py