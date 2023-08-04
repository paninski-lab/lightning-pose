"""Test the predictions module."""

import copy
import gc

import lightning.pytorch as pl
import torch

from lightning_pose.utils.scripts import get_loss_factories, get_model


def test_predict_dataset(cfg, heatmap_data_module, remove_logs, tmpdir):
    """Test the prediction of a dataset after model training.

    NOTE: this only tests a heatmap tracker

    """

    from lightning_pose.utils.predictions import predict_dataset

    # make a basic heatmap tracker
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=heatmap_data_module)

    # build model
    model = get_model(cfg=cfg_tmp, data_module=heatmap_data_module, loss_factories=loss_factories)

    # make a checkpoint callback so we know where model is saved
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=str(tmpdir))

    # train model for a couple epochs
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=2,
        min_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[ckpt_callback],
        limit_train_batches=2,
    )
    trainer.fit(model=model, datamodule=heatmap_data_module)

    # test 1: all available inputs
    predict_dataset(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test1.csv")),
        trainer=trainer,
        model=model,
    )

    # test 2: no trainer
    predict_dataset(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test2.csv")),
        trainer=None,
        model=model,
    )

    # test 3: no trainer, no model
    predict_dataset(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test3.csv")),
        ckpt_file=ckpt_callback.best_model_path,
        trainer=None,
        model=None,
    )

    # remove tensors from gpu
    del loss_factories
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()


def test_predict_single_video(cfg, heatmap_data_module, video_list, remove_logs, tmpdir):
    """Test the prediction of a video after model training.

    NOTE: this only tests a heatmap tracker

    """

    from lightning_pose.utils.predictions import predict_single_video

    # make a basic heatmap tracker
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=heatmap_data_module)

    # build model
    model = get_model(cfg=cfg_tmp, data_module=heatmap_data_module, loss_factories=loss_factories)

    # make a checkpoint callback so we know where model is saved
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=str(tmpdir))

    # train model for a couple epochs
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=2,
        min_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[ckpt_callback],
        limit_train_batches=2,
    )
    trainer.fit(model=model, datamodule=heatmap_data_module)

    # test 1: all available inputs
    predict_single_video(
        cfg_file=cfg_tmp,
        video_file=video_list[0],
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test1.csv")),
        trainer=trainer,
        model=model,
        save_heatmaps=False,
    )

    # test 2: no trainer
    predict_single_video(
        cfg_file=cfg_tmp,
        video_file=video_list[0],
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test2.csv")),
        trainer=None,
        model=model,
        save_heatmaps=False,
    )

    # test 3: no trainer, no model
    predict_single_video(
        cfg_file=cfg_tmp,
        video_file=video_list[0],
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test3.csv")),
        ckpt_file=ckpt_callback.best_model_path,
        trainer=None,
        model=None,
        save_heatmaps=False,
    )

    # test 4: all available inputs, return heatmaps
    predict_single_video(
        cfg_file=cfg_tmp,
        video_file=video_list[0],
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test4.csv")),
        trainer=trainer,
        model=model,
        save_heatmaps=True,
    )

    # remove tensors from gpu
    del loss_factories
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()
