"""Helper function for unit-testing models."""

import gc

import torch

from lightning_pose.utils.predictions import PredictionHandler
from lightning_pose.utils.scripts import get_loss_factories, get_model


def run_model_test(cfg, data_module, video_dataloader, trainer, remove_logs_fn, video_list=None):
    """Helper function to simplify unit tests which run different models."""

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # build model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

    try:

        print("====")
        print("model: ", type(model))
        print(type(model).__bases__)
        print("backbone: ", type(model.backbone))
        print("====")
        # train model for a couple epochs
        trainer.fit(model=model, datamodule=data_module)

        # predict on labeled frames
        labeled_preds = trainer.predict(
            model=model,
            dataloaders=data_module.full_labeled_dataloader(),
            return_predictions=True,
        )
        pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=None)
        pred_handler(preds=labeled_preds)

        # predict on unlabeled video
        if video_dataloader is not None:
            unlabeled_preds = trainer.predict(
                model=model, dataloaders=video_dataloader, return_predictions=True)
            if video_list is not None:
                pred_handler = PredictionHandler(
                    cfg=cfg, data_module=data_module, video_file=video_list[0])
                pred_handler(preds=unlabeled_preds)

    finally:

        # remove tensors from gpu
        del loss_factories
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # clean up logging
        remove_logs_fn()
