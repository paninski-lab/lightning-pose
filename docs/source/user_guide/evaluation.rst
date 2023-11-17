###################
Evaluation
###################

Beyond providing access to loss values throughout training with Tensorboard, the Lightning Pose
package also offers several diagnostic tools to compare the performance of trained models on
labeled frames and unlabeled videos.

##### Fiftyone

This component provides tools for visualizing the predictions of one or more trained
models on labeled frames or on test videos.

See the documentation [here](docs/fiftyone.md).

##### Streamlit

This component provides tools for quantifying model performance across a range of
metrics for both labeled frames and unlabeled videos:

- Pixel error (labeled data only)
- Temporal norm (unlabeled data only)
- Pose PCA error (if `data.columns_for_singleview_pca` is not `null` in the config file)
- Multi-view consistency error (if `data.mirrored_column_matches` is not `null` in the config
  file)

See the documentation [here](docs/apps.md).

* streamlit
* fiftyone
