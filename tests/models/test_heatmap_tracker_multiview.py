"""Test the initialization, training, and inference of multiview heatmap models."""

import copy


def test_multiview_heatmap_cnn(
    cfg_multiview,
    multiview_heatmap_data_module,
    video_dataloader_multiview,
    trainer,
    run_model_test,
):
    """Test initialization and training of a multiview model with heatmap_cnn head."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_multiview"
    cfg_tmp.model.head = "heatmap_cnn"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module,
        video_dataloader=video_dataloader_multiview,
        trainer=trainer,
    )


def test_multiview_feature_transformer(
    cfg_multiview,
    multiview_heatmap_data_module,
    video_dataloader_multiview,
    trainer,
    run_model_test,
):
    """Test initialization and training of a multiview model with feature_transformer head."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_multiview"
    cfg_tmp.model.head = "feature_transformer"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module,
        video_dataloader=video_dataloader_multiview,
        trainer=trainer,
    )
