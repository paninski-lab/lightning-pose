.. _mixed_precision:

#####################################
Mixed Precision Training & Inference
#####################################

Lightning Pose supports running inference at reduced numerical precision (FP16 or BF16
"mixed precision") in addition to the default FP32. This can speed up the model's forward
pass on compatible GPUs. Model weights are always stored and loaded as FP32 on disk --
precision only affects how the forward pass is computed at inference time, not the checkpoint
itself.

This page will grow over time as we benchmark more architectures, datasets, and hardware.

Usage
=====

From the CLI, pass ``--precision`` to ``litpose predict``:

.. code-block:: bash

    litpose predict <model_dir> <input_path> --precision fp16

Valid choices are ``fp32`` (default), ``fp16``, and ``bf16``.

From the Python API, pass ``precision`` to ``Model.from_dir``:

.. code-block:: python

    from lightning_pose.api import Model

    model = Model.from_dir("outputs/2024-01-01/12-00-00", precision="16-mixed")

The Python API uses PyTorch Lightning's native precision strings directly:
``"32-true"``, ``"16-mixed"``, or ``"bf16-mixed"``.

Results: accuracy
==================

Two separate questions matter here: does the precision used *during training* affect
accuracy, and does the precision used *during inference* (independent of training precision)
affect accuracy?

**Training precision.** Training a model at reduced precision does change accuracy slightly,
on ``mirror-mouse-fused`` (resnet50_animal_ap10k, 100 train frames, mean over 3 seeds):

.. list-table:: Effect of training precision (same precision used for eval)
   :widths: 40 30
   :header-rows: 1

   * - Configuration
     - Mean pixel error
   * - FP32 train / FP32 eval
     - 7.11 px
   * - FP16 train / FP16 eval
     - 6.40 px
   * - BF16 train / BF16 eval
     - 6.84 px

FP16 training gave the lowest error in this experiment, BF16 in the middle, and FP32 the
highest -- though the gap is small relative to seed-to-seed variation, so this should not be
read as "FP16 training is definitively better."

**Inference precision.** Independent of training precision, changing the precision used
*only at inference time* (model trained at FP32, evaluated at FP16/BF16) has essentially no
effect on accuracy, across every dataset tested:

.. list-table:: Effect of inference precision (model trained at FP32)
   :widths: 25 20 20 20
   :header-rows: 1

   * - Dataset
     - FP32 eval
     - FP16 eval
     - BF16 eval
   * - mirror-mouse-fused
     - 7.11 px
     - 7.11 px
     - 7.11 px
   * - mirror-fish
     - 11.52 px
     - 11.52 px
     - 11.51 px
   * - crim13
     - 17.52 px
     - 17.52 px
     - 17.51 px

Deltas are well under 0.01px in every case -- far below seed-to-seed variation. In short:
**it is safe to run inference at reduced precision regardless of what precision the model
was trained at.**

Results: speed
================

Reduced precision can speed up the model's forward pass, but the effect depends heavily on
batch size and architecture. The numbers below isolate the forward pass only (synthetic
inputs already on the GPU, no data loading) on an A100-SXM4-80GB, measured relative to a true
FP32 baseline (TF32 disabled for both matmul and cuDNN):

.. list-table:: Forward-pass speedup vs. FP32, by batch size
   :widths: 20 15 15 15
   :header-rows: 1

   * - Architecture / batch size
     - FP16
     - BF16
     - Notes
   * - ResNet50, batch 1
     - 0.75x
     - 0.67x
     - slower -- autocast overhead dominates
   * - ResNet50, batch 8
     - 1.27x
     - 1.10x
     -
   * - ResNet50, batch 32
     - 3.14x
     - 2.93x
     -
   * - ResNet50, batch 64
     - 3.31x
     - 3.14x
     -
   * - ViT-S (DINOv2), batch 1
     - 0.69x
     - 0.69x
     - slower -- autocast overhead dominates
   * - ViT-S (DINOv2), batch 8
     - 1.29x
     - 1.43x
     -
   * - ViT-S (DINOv2), batch 32
     - 4.43x
     - 4.40x
     -
   * - ViT-S (DINOv2), batch 64
     - 4.73x
     - 4.73x
     -

At batch size 1, reduced precision is actually *slower* than FP32 due to autocast overhead
with little compute to amortize it against. The crossover point is around batch size 8, and
by batch size 32 both architectures see a substantial speedup (roughly 3x for ResNet50, 4.4x
for ViT-S). FP16 and BF16 perform similarly to each other on A100.

**Important caveat:** these numbers measure the GPU forward pass in isolation. End-to-end
video inference (via ``litpose predict``) did not show a measurable speedup from reduced
precision on a T4 GPU -- video preprocessing (DALI) dominates total runtime, not the model's
forward pass. Whether reduced precision speeds up your end-to-end workflow depends on your
bottleneck: batch size, GPU, and video decoding overhead all matter.
