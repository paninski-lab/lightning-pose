"""Test the initialization, training, and inference of multiview, multihead heatmap models."""

import copy


# def test_multiview_multihead_heatmap_cnn(
#     cfg_multiview,
#     multiview_heatmap_data_module,
#     video_dataloader_multiview,
#     trainer,
#     run_model_test,
# ):
#     """Test initialization and training of a multiview multihead model with heatmap_cnn head."""

#     cfg_tmp = copy.deepcopy(cfg_multiview)
#     cfg_tmp.model.model_type = "heatmap_multiview_multihead"
#     cfg_tmp.model.head = "heatmap_cnn"
#     cfg_tmp.model.losses_to_use = []

#     run_model_test(
#         cfg=cfg_tmp,
#         data_module=multiview_heatmap_data_module,
#         video_dataloader=video_dataloader_multiview,
#         trainer=trainer,
#     )
