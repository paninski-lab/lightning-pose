##########
Evaluation
##########

Beyond providing access to loss values throughout training with Tensorboard, the Lightning Pose
package also offers several diagnostic tools to compare the performance of trained models on
labeled frames and unlabeled videos.

**FiftyOne** provides tools for visualizing the predictions of one or more trained models on
labeled frames.

**Streamlit** provides tools for quantifying and plotting model performance across a range of metrics for both labeled frames and unlabeled videos.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   evaluation.fiftyone
   evaluation.streamlit
