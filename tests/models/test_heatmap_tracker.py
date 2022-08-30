"""Test the initialization and training of heatmap models."""

import copy
import pytest
import torch

from lightning_pose.utils.scripts import get_loss_factories, get_model


def test_supervised_heatmap(cfg, heatmap_data_module, video_dataloader, trainer, remove_logs):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.do_context = False
    cfg_tmp.model.losses_to_use = []
    
    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=heatmap_data_module)

    # build model
    model = get_model(cfg=cfg_tmp, data_module=heatmap_data_module, loss_factories=loss_factories)

    # train model for a couple epochs
    trainer.fit(model=model, datamodule=heatmap_data_module)

    # predict on labeled frames
    trainer.predict(
        model=model,
        dataloaders=heatmap_data_module.full_labeled_dataloader(),
        return_predictions=True,
    )

    # predict on unlabeled video
    trainer.predict(
        model=model,
        dataloaders=video_dataloader,
        return_predictions=True,
    )

    # remove tensors from gpu
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()


def test_unsupervised_heatmap_temporal(
    cfg, heatmap_data_module_combined, video_dataloader, trainer, remove_logs,
):
    """Test the initialization and training of an unsupervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.do_context = False
    cfg_tmp.model.losses_to_use = ["temporal"]

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=heatmap_data_module_combined)

    # build model
    model = get_model(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_combined,
        loss_factories=loss_factories,
    )

    # train model for a couple epochs
    trainer.fit(model=model, datamodule=heatmap_data_module_combined)

    # predict on labeled frames
    trainer.predict(
        model=model,
        dataloaders=heatmap_data_module_combined.full_labeled_dataloader(),
        return_predictions=True,
    )

    # predict on unlabeled video
    trainer.predict(
        model=model,
        dataloaders=video_dataloader,
        return_predictions=True,
    )

    # remove tensors from gpu
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()
