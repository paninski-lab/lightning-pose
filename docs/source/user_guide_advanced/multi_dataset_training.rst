.. _multi_dataset_training:

#######################################
Multi-dataset training and fine-tuning
#######################################

Options for training one model on a corpus merged from several labeled datasets (one canonical
keypoint list, each source labeling a subset of it) and for fine-tuning such a model on a new
dataset. Every option is off by default; a config without them trains exactly as stock
Lightning Pose.

Multi-dataset corpus
====================

The merged CSV lists images as ``labeled-data/<dataset>/<session>/<frame>``.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - key
     - default
     - what it does
   * - ``data.dataset_names``
     - ``null``
     - Ordered list of source datasets. Every image path must name one of them; each example
       gets its dataset id (index in this list). Enables the stratified split: train/val/test
       proportions are applied within every source dataset, and the split is saved to
       ``split_manifest.json`` in the model directory.
   * - ``training.sampling_temperature``
     - ``null``
     - With ``> 1``, a sampler sets each dataset's share of supervision proportional to
       ``(frames x mean labeled keypoints per frame)^(1/T)``; ``'inf'`` gives every dataset
       the same share. ``null`` or ``1`` keeps the stock shuffled loader. Needs
       ``data.dataset_names``; supervised models only. Epoch length then varies, so use
       step-based schedules (below).
   * - ``training.imgaug_per_dataset_zoom``
     - ``null``
     - ``{dataset: upper}`` or ``{dataset: [lower, upper]}``: that dataset gets its own copy of
       the augmentation pipeline with its own ``CropAndPad`` percent range (negative = zoom
       in, positive = zoom out); other datasets use the global pipeline. Needs
       ``data.dataset_names``; single-view models. Not applied to val/test or prediction.
   * - ``training.imgaug_seed``
     - ``null``
     - Seed for imgaug's own random number generator.

Model and optimization
======================

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - key
     - default
     - what it does
   * - ``model.head_hidden_channels``
     - ``null``
     - An int ``H`` makes the heatmap head nonlinear: the last transposed convolution outputs
       ``H`` channels, then LayerNorm -> ReLU -> 1x1 convolution to one channel per keypoint.
   * - ``training.lr_scheduler_params.multisteplr.milestone_steps``
     - (unset)
     - Learning-rate milestones in optimizer steps, applied every step (no rounding to
       epochs). Explicit epoch ``milestones`` keep their epoch behaviour.
   * - ``training.unfreeze_warmup_steps``
     - ``null``
     - With ``training.unfreezing_step``: after unfreezing, the backbone learning rate rises
       linearly from 10 % to 100 % over this many steps. The stock ramp (x1.5 per step)
       reaches the full rate about 6 steps after unfreezing, which can make ViT-B runs
       collapse at that step.
   * - ``training.gradient_clip_val``
     - ``null``
     - Clip the global gradient norm to this value.

Fine-tuning a trained model
===========================

Both options assume ``model.checkpoint`` points at the model being fine-tuned.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - key
     - default
     - what it does
   * - ``model.lora``
     - ``null``
     - ``{rank: 16, alpha: 2*rank, targets: [q_proj, k_proj, v_proj, o_proj, up_proj,
       down_proj], lr: null}``: freeze the backbone and train low-rank adapters on the named
       linear layers (zero-initialised, so training starts from the loaded model). The
       adapters are trained from the first step at ``lr`` (default: the global rate). Saved
       checkpoints contain the adapters and reload with the same block.
   * - ``model.anchor``
     - ``null``
     - ``{weight: 1.0, keypoints: null, mode: unlabeled, conf_power: 0}``: keep the keypoints
       the new dataset does not label close to the original model. Adds an MSE between the
       heatmaps of the model being trained and those of a frozen copy of ``model.checkpoint``,
       on channels without a label in the frame (``mode: unlabeled``) or on every channel
       (``all``), restricted to ``keypoints`` if given. ``conf_power > 0`` weights each term
       by the frozen model's heatmap peak. Labeled frames; single-view heatmap models.

Example
=======

The settings of the head-fixed mouse model (ViT-B DINOv3 at 384 px; the per-dataset zoom ranges
bring every source to the corpus's smallest and largest apparent scales):

.. code-block:: yaml

    data:
      image_resize_dims: {height: 384, width: 384}
      dataset_names: [facemap, ibl, kondo, ...]
    model:
      backbone: vitb_dinov3
      head_hidden_channels: 256
    training:
      sampling_temperature: 2
      imgaug_hflip: true
      imgaug_per_dataset_zoom: {facemap: [-0.15, 2.0], ibl: [-0.4, 0.15], ...}
      unfreezing_step: 3000
      unfreeze_warmup_steps: 1000
      gradient_clip_val: 1.0
      min_steps: 14000
      max_steps: 14000
      lr_scheduler_params:
        multisteplr:
          milestone_steps: [6000, 8000, 10000]
