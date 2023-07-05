"""Test the initialization and training of regression models."""

import copy

from lightning_pose.utils.tests import run_model_test


def test_supervised_regression(
    cfg, base_data_module, video_dataloader, trainer, remove_logs
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


def test_supervised_regression_context(
    cfg_context, base_data_module_context, video_dataloader, trainer, remove_logs
):
    """Test the initialization and training of a supervised regression context model.

    NOTE: the toy dataset is not a proper context dataset

    """

    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = []
    run_model_test(
        cfg=cfg_tmp,
        data_module=base_data_module_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_semisupervised_regression_temporal(
    cfg,
    base_data_module_combined,
    video_dataloader,
    trainer,
    remove_logs,
):
    """Test the initialization and training of a semi-supervised regression model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = ["temporal"]
    run_model_test(
        cfg=cfg_tmp,
        data_module=base_data_module_combined,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


def test_semisupervised_regression_pcasingleview_context(
    cfg_context,
    base_data_module_combined_context,
    video_dataloader,
    trainer,
    remove_logs,
):
    """Test the initialization and training of a semi-supervised regression context model.

    NOTE: the toy dataset is not a proper context dataset

    """

    cfg_tmp = copy.deepcopy(cfg_context)
    cfg_tmp.model.model_type = "regression"
    cfg_tmp.model.losses_to_use = ["pca_singleview"]
    run_model_test(
        cfg=cfg_tmp,
        data_module=base_data_module_combined_context,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )
