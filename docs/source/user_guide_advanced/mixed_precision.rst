.. _mixed_precision:

#####################################
Mixed Precision Training & Inference
#####################################

Lightning Pose supports running inference at reduced numerical precision (FP16 or BF16
"mixed precision") in addition to the default FP32. This can speed up the model's forward
pass on compatible GPUs. Model weights are always stored and loaded as FP32 on disk --
precision only affects how the forward pass is computed at inference time, not the checkpoint
itself.

**TL;DR**

- **Accuracy:** across the 5 datasets tested, training at reduced precision does not cause a
  systematic accuracy loss -- differences between FP32/FP16/BF16 training are small and mostly
  within seed-to-seed noise (see the SEM columns below). Separately, and more definitively:
  **it is safe to run inference at a different precision than the model was trained at** --
  deltas are under 0.01px in every case tested.
- **Speed:** reduced precision speeds up the model's forward pass substantially at larger batch
  sizes (up to ~3x for ResNet50, ~4.7x for ViT-S at batch 64 on an A100). It does **not**
  measurably speed up end-to-end ``litpose predict`` on a T4 GPU -- video preprocessing (DALI)
  dominates total runtime there, not the forward pass.

This page will grow over time as we benchmark more architectures, datasets, and hardware.

Usage
=====

From the CLI, pass ``--precision`` to ``litpose predict``:

.. code-block:: bash

    litpose predict <model_dir> <input_path> --precision fp16

Valid choices are ``fp32`` (default), ``fp16``, and ``bf16``.

From the Python API, pass the same string to ``Model.from_dir``:

.. code-block:: python

    from lightning_pose.api import Model

    model = Model.from_dir("outputs/2024-01-01/12-00-00", precision="fp16")

The CLI and Python API use the same three precision strings -- ``"fp32"`` (default),
``"fp16"``, or ``"bf16"``. Internally these map to PyTorch Lightning's own precision strings
(``"32-true"``, ``"16-mixed"``, ``"bf16-mixed"``), but you never need to spell those out.

Results: accuracy
==================

Two separate questions matter here: does the precision used *during training* affect
accuracy, and does the precision used *during inference* (independent of training precision)
affect accuracy? Both experiments below use ``resnet50_animal_ap10k``, 100 train frames, and
report mean pixel error ± SEM (standard deviation across 3 seeds, divided by :math:`\sqrt{3}`).

**Training precision.** Training a model at reduced precision and evaluating at that same
precision:

.. list-table:: Effect of training precision (same precision used for eval)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Dataset
     - FP32 train / eval
     - FP16 train / eval
     - BF16 train / eval
   * - `mirror-mouse-fused <https://huggingface.co/datasets/paninski-lab/mirror-mouse-fused>`_
     - 7.11 ± 0.11 px
     - 6.40 ± 0.23 px
     - 6.84 ± 0.30 px
   * - `mirror-fish <https://huggingface.co/datasets/paninski-lab/mirror-fish>`_
     - 11.52 ± 0.69 px
     - 10.68 ± 0.91 px
     - 10.84 ± 0.17 px
   * - `crim13 <https://huggingface.co/datasets/paninski-lab/crim13>`_
     - 17.52 ± 0.74 px
     - 17.68 ± 0.24 px
     - 17.49 ± 0.92 px
   * - `facemap <https://huggingface.co/datasets/paninski-lab/facemap>`_
     - 12.95 ± 1.67 px
     - 13.57 ± 2.37 px
     - 11.85 ± 1.31 px
   * - `marmoset3k <https://huggingface.co/datasets/paninski-lab/marmoset3k>`_
     - 37.41 ± 0.61 px
     - 37.27 ± 0.12 px
     - 37.25 ± 0.75 px

Differences between precisions are small relative to the SEM in most cases -- e.g. facemap's
FP16 mean is nominally higher than FP32, but both are within ~1 SEM of each other. The
clearest gap is mirror-mouse-fused, where FP16 training is about 0.7px lower than FP32; even
there the effect is modest. Overall: mixed-precision training does not cause a systematic
accuracy loss across the datasets tested, though the direction and size of any effect is
dataset-dependent.

**Inference precision.** Independent of training precision, changing the precision used
*only at inference time* (model trained at FP32, evaluated at FP16/BF16) has essentially no
effect on accuracy, across every dataset tested:

.. list-table:: Effect of inference precision (model trained at FP32)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Dataset
     - FP32 eval
     - FP16 eval
     - BF16 eval
   * - `mirror-mouse-fused <https://huggingface.co/datasets/paninski-lab/mirror-mouse-fused>`_
     - 7.11 ± 0.11 px
     - 7.11 ± 0.11 px
     - 7.11 ± 0.10 px
   * - `mirror-fish <https://huggingface.co/datasets/paninski-lab/mirror-fish>`_
     - 11.52 ± 0.69 px
     - 11.52 ± 0.69 px
     - 11.51 ± 0.70 px
   * - `crim13 <https://huggingface.co/datasets/paninski-lab/crim13>`_
     - 17.52 ± 0.74 px
     - 17.52 ± 0.74 px
     - 17.51 ± 0.74 px
   * - `facemap <https://huggingface.co/datasets/paninski-lab/facemap>`_
     - 12.95 ± 1.67 px
     - 12.95 ± 1.67 px
     - 12.96 ± 1.68 px
   * - `marmoset3k <https://huggingface.co/datasets/paninski-lab/marmoset3k>`_
     - 37.41 ± 0.61 px
     - 37.41 ± 0.61 px
     - 37.44 ± 0.60 px

Deltas are well under 0.01px in every case -- far below seed-to-seed variation (SEM). In
short: **it is safe to run inference at reduced precision regardless of what precision the
model was trained at.**

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
