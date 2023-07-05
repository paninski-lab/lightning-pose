import gc

import torch

from lightning_pose.utils.scripts import get_loss_factories, get_model


def run_model_test(cfg, data_module, video_dataloader, trainer, remove_logs_fn):
    """Helper function to simplify unit tests which run different models."""

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # build model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

    print("====")
    print("model: ", type(model))
    print(type(model).__bases__)
    print("backbone: ", type(model.backbone))
    print("====")
    # train model for a couple epochs
    trainer.fit(model=model, datamodule=data_module)

    # predict on labeled frames
    trainer.predict(
        model=model,
        dataloaders=data_module.full_labeled_dataloader(),
        return_predictions=True,
    )

    # predict on unlabeled video
    trainer.predict(model=model, dataloaders=video_dataloader, return_predictions=True)

    # remove tensors from gpu
    del loss_factories
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs_fn()
