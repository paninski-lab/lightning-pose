HeatmapTracker
==============

.. currentmodule:: lightning_pose.models.heatmap_tracker

.. autoclass:: HeatmapTracker
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~HeatmapTracker.num_filters_for_upsampling

   .. rubric:: Methods Summary

   .. autosummary::

      ~HeatmapTracker.create_double_upsampling_layer
      ~HeatmapTracker.forward
      ~HeatmapTracker.get_loss_inputs_labeled
      ~HeatmapTracker.heatmaps_from_representations
      ~HeatmapTracker.initialize_upsampling_layers
      ~HeatmapTracker.make_upsampling_layers
      ~HeatmapTracker.predict_step
      ~HeatmapTracker.run_hard_argmax
      ~HeatmapTracker.run_subpixelmaxima

   .. rubric:: Attributes Documentation

   .. autoattribute:: num_filters_for_upsampling

   .. rubric:: Methods Documentation

   .. automethod:: create_double_upsampling_layer
   .. automethod:: forward
   .. automethod:: get_loss_inputs_labeled
   .. automethod:: heatmaps_from_representations
   .. automethod:: initialize_upsampling_layers
   .. automethod:: make_upsampling_layers
   .. automethod:: predict_step
   .. automethod:: run_hard_argmax
   .. automethod:: run_subpixelmaxima
