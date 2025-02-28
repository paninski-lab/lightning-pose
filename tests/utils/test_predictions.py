"""Test the predictions module."""

import copy
import gc
import os

import lightning.pytorch as pl
import pytest
import torch

from lightning_pose.utils.predictions import (
    export_predictions_and_labeled_video,
    predict_dataset,
    predict_single_video,
)
from lightning_pose.utils.scripts import get_loss_factories, get_model


def test_predict_dataset(cfg, heatmap_data_module, tmpdir):
    """Test the prediction of a dataset after model training.

    NOTE: this only tests a heatmap tracker

    """
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
        logger=False,
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


def test_predict_single_video(cfg, heatmap_data_module, video_list, tmpdir):
    """Test the prediction of a video after model training.

    NOTE: this only tests a heatmap tracker

    """
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
        logger=False,
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
    )

    # test 2: no trainer
    predict_single_video(
        cfg_file=cfg_tmp,
        video_file=video_list[0],
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test2.csv")),
        trainer=None,
        model=model,
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
    )

    # test 4: all available inputs, return heatmaps
    predict_single_video(
        cfg_file=cfg_tmp,
        video_file=video_list[0],
        data_module=heatmap_data_module,
        preds_file=str(tmpdir.join("test4.csv")),
        trainer=trainer,
        model=model,
    )

    # remove tensors from gpu
    del loss_factories
    del model
    gc.collect()
    torch.cuda.empty_cache()


def test_export_predictions_and_labeled_video(
    cfg, heatmap_data_module, video_list, tmpdir
):
    """Test helper function that predicts videos then makes a labeled movie."""
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
        logger=False,
        limit_train_batches=2,
    )
    trainer.fit(model=model, datamodule=heatmap_data_module)

    # test 1: all available inputs
    csv_file = str(tmpdir.join("test1.csv"))
    mp4_file = str(tmpdir.join("test1.mp4"))
    npy_file = csv_file.replace(".csv", "_heatmaps.npy")
    export_predictions_and_labeled_video(
        video_file=video_list[0],
        cfg=cfg_tmp,
        prediction_csv_file=csv_file,
        ckpt_file=None,
        trainer=trainer,
        model=model,
        data_module=heatmap_data_module,
        labeled_mp4_file=mp4_file,
    )
    assert os.path.exists(csv_file)
    assert os.path.exists(mp4_file)
    assert not os.path.exists(npy_file)

    # test 2: no trainer
    csv_file = str(tmpdir.join("test2.csv"))
    mp4_file = str(tmpdir.join("test2.mp4"))
    npy_file = csv_file.replace(".csv", "_heatmaps.npy")
    export_predictions_and_labeled_video(
        video_file=video_list[0],
        cfg=cfg_tmp,
        prediction_csv_file=csv_file,
        ckpt_file=None,
        trainer=None,
        model=model,
        data_module=heatmap_data_module,
        labeled_mp4_file=mp4_file,
    )
    assert os.path.exists(csv_file)
    assert os.path.exists(mp4_file)
    assert not os.path.exists(npy_file)

    # test 3: raise proper error
    with pytest.raises(ValueError):
        export_predictions_and_labeled_video(
            video_file=video_list[0],
            cfg=cfg_tmp,
            prediction_csv_file=str(tmpdir.join("test4.csv")),
            ckpt_file=None,
            trainer=trainer,
            model=None,
            data_module=heatmap_data_module,
            labeled_mp4_file=str(tmpdir.join("test3.mp4")),
        )

    # remove tensors from gpu
    del loss_factories
    del model
    gc.collect()
    torch.cuda.empty_cache()
