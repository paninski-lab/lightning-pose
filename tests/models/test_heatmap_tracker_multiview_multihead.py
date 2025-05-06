"""Test the initialization and training of heatmap models."""

import copy


def test_multiview_multihead_heatmap_cnn(
    cfg_multiview,
    multiview_heatmap_data_module,
    video_dataloader,
    trainer,
    # run_model_test,
):
    """Test initialization and training of a multiview multihead model with heatmap_cnn head."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_multiview_multihead"
    cfg_tmp.model.head = "heatmap_cnn"
    cfg_tmp.model.losses_to_use = []

    # TODO: need to update Model API for predictions (images and vids)
    # run_model_test(
    #     cfg=cfg_tmp,
    #     data_module=multiview_heatmap_data_module,
    #     video_dataloader=video_dataloader,
    #     trainer=trainer,
    # )

    _run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


def _run_model_test(cfg, data_module, video_dataloader, trainer):
    """Helper function to simplify unit tests which run different models."""

    import gc
    import torch
    from lightning_pose.utils.predictions import PredictionHandler
    from lightning_pose.utils.scripts import (
        get_loss_factories,
        get_model,
    )

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

        # # predict on unlabeled video
        # if video_dataloader is not None:
        #     trainer.predict(model=model, dataloaders=video_dataloader, return_predictions=True)

    finally:

        # remove tensors from gpu
        del loss_factories
        del model
        gc.collect()
        torch.cuda.empty_cache()
