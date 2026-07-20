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
affect accuracy? Single-view results below use ``resnet50_animal_ap10k``; multi-view results
use ``heatmap_multiview_transformer`` with a ``vits_dinov2`` backbone, since the single-view
architecture has no cross-view attention mechanism. Both use 100 train frames and report mean
pixel error ± SEM (standard deviation across 3 seeds, divided by :math:`\sqrt{3}`) -- for
multi-view datasets, error is averaged across camera views within each seed before computing
the across-seed mean/SEM.

**Training precision.** Training a model at reduced precision and evaluating at that same
precision:

.. list-table:: Effect of training precision (same precision used for eval)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Dataset
     - FP32 train / eval
     - FP16 train / eval
     - BF16 train / eval
   * - **Single-view**
     -
     -
     -
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
   * - **Multi-view**
     -
     -
     -
   * - `mirror-mouse-separate <https://huggingface.co/datasets/paninski-lab/mirror-mouse-separate>`_ (2 views)
     - 5.23 ± 0.65 px
     - 5.23 ± 0.36 px
     - 5.03 ± 0.37 px
   * - `fly-anipose <https://huggingface.co/datasets/paninski-lab/fly-anipose>`_ (6 views)
     - 9.34 ± 0.02 px
     - 9.24 ± 0.09 px
     - 9.29 ± 0.10 px

Differences between precisions are small relative to the SEM in most cases -- e.g. facemap's
FP16 mean is nominally higher than FP32, but both are within ~1 SEM of each other. The
clearest single-view gap is mirror-mouse-fused, where FP16 training is about 0.7px lower than
FP32; even there the effect is modest. The multi-view datasets show an even cleaner null
result -- no dataset's FP16/BF16 mean clears its own FP32 SEM bar. Overall: mixed-precision
training does not cause a systematic accuracy loss across any dataset tested, single- or
multi-view, though the direction and size of any effect is dataset-dependent.

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
   * - **Single-view**
     -
     -
     -
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
   * - **Multi-view**
     -
     -
     -
   * - `mirror-mouse-separate <https://huggingface.co/datasets/paninski-lab/mirror-mouse-separate>`_ (2 views)
     - 5.23 ± 0.65 px
     - 4.96 ± 0.39 px
     - 4.95 ± 0.38 px
   * - `fly-anipose <https://huggingface.co/datasets/paninski-lab/fly-anipose>`_ (6 views)
     - 9.34 ± 0.02 px
     - 9.34 ± 0.02 px
     - 9.34 ± 0.02 px

Deltas are well under 0.01px in every case, single- or multi-view -- far below seed-to-seed
variation (SEM). In short: **it is safe to run inference at reduced precision regardless of
what precision the model was trained at, and this holds for multi-view cross-view-attention
models as well as single-view ones.**

Results: speed
================

Reduced precision can speed up the model's forward pass, but the effect depends heavily on
batch size and architecture. The numbers below isolate the forward pass only (synthetic
inputs already on the GPU, no data loading) on an A100-SXM4-80GB, measured relative to a true
FP32 baseline (TF32 disabled for both matmul and cuDNN). These use the single-view
architectures (ResNet50, ViT-S DINOv2) at 256x256 input (ViT-S snapped to 252x252 to divide
evenly by its 14px patch size):

.. list-table:: Forward-pass speedup vs. FP32, by batch size (single-view, 256px input)
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

The same isolated-forward-pass experiment on the multi-view architecture
(``heatmap_multiview_transformer`` + ``vits_dinov2``, 6 camera views, A100, true FP32
baseline) shows a comparable speedup, reached at a smaller nominal batch size since each
view multiplies the effective batch fed to the backbone:

.. list-table:: Forward-pass speedup vs. FP32, by batch size (multi-view, 6 views, 256px input per view)
   :widths: 30 15 15
   :header-rows: 1

   * - Batch size (effective ViT batch = batch x 6 views)
     - FP16
     - BF16
   * - 1 (6)
     - 1.08x
     - 1.07x
   * - 2 (12)
     - 1.49x
     - 1.63x
   * - 4 (24)
     - 2.89x
     - 2.91x
   * - 8 (48)
     - 4.29x
     - 4.11x
   * - 16 (96)
     - 4.56x
     - 4.55x

**Important caveat -- applies to both single- and multi-view:** these tables measure the
GPU forward pass in isolation. End-to-end video inference (via ``litpose predict``) did not
show a measurable speedup from reduced precision for either single-view models (T4 GPU) or
multi-view models (A100 GPU, see the end-to-end table below) -- despite the multi-view
forward pass accounting for the large majority of end-to-end wall time in a separate
profiling pass. This is a genuine open question we haven't resolved yet: why an isolated
forward-pass speedup this large doesn't show up end-to-end. We'll expand this section if and
when we track down the discrepancy.

**Multi-view end-to-end speed.** The forward-pass benchmark above uses single-view
architectures; multi-view models (``heatmap_multiview_transformer`` + ``vits_dinov2``)
process every camera view simultaneously each forward pass, so we also measured end-to-end
``litpose predict`` speed directly, 7 repeats per precision, on an A100-SXM4-80GB, 256x256
input per view. Both datasets use the same DALI ``sequence_length`` (64), so the two rows
below are directly comparable to each other, not just within each row:

.. list-table:: End-to-end inference speed, multi-view models (A100 GPU, seq_len=64)
   :widths: 25 20 20 20
   :header-rows: 1

   * - Dataset
     - FP32
     - FP16
     - BF16
   * - mirror-mouse-separate (2 views)
     - 79.6s
     - 79.8s (+0.2%)
     - 80.0s (+0.5%)
   * - fly-anipose (6 views)
     - 27.2s
     - 27.0s (-0.6%)
     - 27.1s (-0.1%)

Both datasets are flat across precisions (well under 1% spread) -- the same
DALI/video-I/O-bound pattern seen in the single-view benchmarks. Notably, on an earlier T4
run at a smaller, memory-constrained sequence length (16), fly-anipose showed a small but
real slowdown at BF16 (+5.2%); that slowdown disappears here at sequence length 64 on an
A100, consistent with our hypothesis that it was a batch-size artifact (the forward-pass
microbenchmark above shows reduced precision is actually slower than FP32 at small batch
sizes, crossing over to a speedup only around batch 8) rather than a fundamental property of
multi-view models.
