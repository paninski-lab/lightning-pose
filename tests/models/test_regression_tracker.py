"""Test the initialization and training of regression models."""

import copy


def test_supervised_regression(
    cfg,
    base_data_module,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a supervised regression model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = []
    run_model_test(
        cfg=cfg_tmp,
        data_module=base_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_semisupervised_regression_temporal_pcasingleview(
    cfg,
    base_data_module_combined,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
):
    """Test the initialization and training of a semi-supervised regression model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = ["temporal", "pca_singleview"]
    run_model_test(
        cfg=cfg_tmp,
        data_module=base_data_module_combined,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )
