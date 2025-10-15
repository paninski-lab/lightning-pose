.. _user_guide_multiview:

#####################
User guide: multiview
#####################

In addition to the single camera setups discussed in the previous user guide, Lightning Pose also
supports multi-camera setups, where the same scene is captured from different angles with
different cameras.

We offer a multi-view transformer solution that processes all views simultaneously, learning
cross-view correlations to improve performance.

.. note::

    As of October 2025, multi-view Lightning Pose does not yet support context
    frames or unsupervised losses.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   organizing_data
   patch_masking_3d_loss
   training_inference