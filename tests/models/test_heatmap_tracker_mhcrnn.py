"""Test the initialization and training of context heatmap multi-head crnn models."""

import copy


def test_supervised_heatmap_mhcrnn(
    cfg,
    heatmap_data_module_context,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap mhcrnn model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_supervised_multiview_heatmap_mhcrnn(
    cfg_multiview,
    multiview_heatmap_data_module_context,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap mhcrnn model."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_semisupervised_heatmap_mhcrnn_pcasingleview(
    cfg,
    heatmap_data_module_combined_context,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a semi-supervised heatmap mhcrnn model.

    NOTE: the toy dataset is not a proper context dataset

    """

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    cfg_tmp.model.losses_to_use = ["pca_singleview"]

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_combined_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_semisupervised_heatmap_mhcrnn_pcasingleview_vit(
    cfg,
    heatmap_data_module_combined_context,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a semi-supervised heatmap mhcrnn model ViT backbone.

    NOTE: the toy dataset is not a proper context dataset

    """

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.backbone = "vit_b_sam"
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    cfg_tmp.model.losses_to_use = ["pca_singleview"]

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_combined_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_semisupervised_multiview_heatmap_mhcrnn_multiview(
    cfg_multiview,
    multiview_heatmap_data_module_combined_context,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a semi-supervised multiview heatmap model."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    cfg_tmp.model.losses_to_use = ["pca_multiview"]

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module_combined_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )
