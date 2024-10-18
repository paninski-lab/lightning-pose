"""Test the initialization and training of heatmap models."""

import copy
import pytest
import torch


@pytest.mark.parametrize("num_gpus", [1, 2] if torch.cuda.device_count() > 1 else [1])
def test_supervised_heatmap(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
    num_gpus,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []
    cfg_tmp.training.num_gpus = num_gpus

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


@pytest.mark.parametrize("num_gpus", [1, 2] if torch.cuda.device_count() > 1 else [1])
def test_supervised_multiview_heatmap(
    cfg_multiview,
    multiview_heatmap_data_module,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
    num_gpus,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []
    cfg_tmp.training.num_gpus = num_gpus

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )


@pytest.mark.parametrize("num_gpus", [1, 2] if torch.cuda.device_count() > 1 else [1])
def test_semisupervised_heatmap_temporal_pcasingleview(
    cfg,
    heatmap_data_module_combined,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
    num_gpus,
):
    """Test the initialization and training of a semi-supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = ["temporal", "pca_singleview"]
    cfg_tmp.training.num_gpus = num_gpus

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_combined,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )

@pytest.mark.parametrize("num_gpus", [1, 2] if torch.cuda.device_count() > 1 else [1])
def test_semisupervised_multiview_heatmap_multiview(
    cfg_multiview,
    multiview_heatmap_data_module_combined,
    video_dataloader,
    trainer,
    remove_logs,
    run_model_test,
    num_gpus,
):
    """Test the initialization and training of a semi-supervised multiview heatmap model."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = ["pca_multiview"]
    cfg_tmp.training.num_gpus = num_gpus

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module_combined,
        video_dataloader=video_dataloader,
        trainer=trainer,
        remove_logs_fn=remove_logs,
    )