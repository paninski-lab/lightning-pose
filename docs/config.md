# Important arguments

Below is a list of some commonly modified arguments related to model architecture/training. See
the file `scripts/configs/config_default.yaml` for a complete list of arguments and their defaults.
When training a model on a new dataset, you must copy/paste this default config and update the
arguments to match your data.

- training.train_batch_size (default: `16`) - batch size for labeled data
- training.min_epochs (default: `300`)
- training.max_epochs (default: `750`)
- model.model_type (default: `heatmap`)
  - regression: model directly outputs an (x, y) prediction for each keypoint; not recommended
  - heatmap: model outputs a 2D heatmap for each keypoint
  - heatmap_mhcrnn: the "multi-head convolutional RNN", this model takes a temporal window of
    frames as input, and outputs two heatmaps: one "context-aware" and one "static". The prediction
    with the highest confidence is automatically chosen. Must also set `model.do_context=True`.
- model.do_context (default: `False`) - set to `True` when using `model.model_type=heatmap_mhcrnn`.
- model.losses_to_use (default: `[]`) - this argument relates to the unsupervised losses. An empty
  list indicates a fully supervised model. Each element of the list corresponds to an unsupervised
  loss. For example,
  `model.losses_to_use=[pca_multiview,temporal]` will fit both a pca_multiview loss and a temporal
  loss. Options include:
  - pca_multiview: penalize inconsistencies between multiple camera views
  - pca_singleview: penalize implausible body configurations
  - temporal: penalize large temporal jumps

See the `losses` section of `scripts/configs/config_default.yaml` for more details on the various
losses and their associated hyperparameters. The default values in the config file are reasonable
for a range of datasets.

Some arguments related to video loading, both for semi-supervised models and when predicting new
videos with any of the models:

- dali.base.train.sequence_length (default: `32`) - number of unlabeled frames per batch in
  `regression` and `heatmap` models (i.e. "base" models that do not use temporal context frames)
- dali.base.predict.sequence_length (default: `96`) - batch size when predicting on a new video with
  a "base" model
- dali.context.train.batch_size (default: `16`) - number of unlabeled frames per batch in
  `heatmap_mhcrnn` model (i.e. "context" models that utilize temporal context frames); each frame in
  this batch will be accompanied by context frames, so the true batch size will actually be larger
  than this number
- dali.context.predict.sequence_length (default: `96`) - batch size when predicting on a ndew video
  with a "context" model
