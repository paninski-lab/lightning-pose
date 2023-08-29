"""Test the scripts module.

Note that many of the functions in the scripts module are explicitly used (and therefore implicitly
tested) in conftest.py

"""

import copy
import gc
import os

import lightning.pytorch as pl
import pytest
import torch


def test_calculate_train_batches(cfg, base_dataset):
    """Test the computation of train batches, which is a function of labeled and unlabeled info."""

    from lightning_pose.utils.scripts import calculate_train_batches

    cfg_tmp = copy.deepcopy(cfg)

    # return value if set in config
    for n in [2, 12, 22]:
        cfg_tmp.training.limit_train_batches = n
        n_batches = calculate_train_batches(cfg_tmp, base_dataset)
        assert n_batches == n

    # None with small number of train frames - return minimum of 10
    cfg_tmp.training.limit_train_batches = None
    cfg_tmp.training.train_frames = 2
    n_batches = calculate_train_batches(cfg_tmp, base_dataset)
    assert n_batches == 10

    # None with large number of frames
    n = 50
    cfg_tmp.training.limit_train_batches = None
    cfg_tmp.training.train_frames = n
    cfg_tmp.training.train_batch_size = 1
    n_batches = calculate_train_batches(cfg_tmp, base_dataset)
    assert n_batches == n


def test_export_predictions_and_labeled_video(
        cfg, heatmap_data_module, video_list, remove_logs, tmpdir):
    """Test helper function that predicts videos then makes a labeled movie."""

    from lightning_pose.utils.scripts import (
        export_predictions_and_labeled_video,
        get_loss_factories,
        get_model,
    )

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
        save_heatmaps=False,
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
        save_heatmaps=False,
    )
    assert os.path.exists(csv_file)
    assert os.path.exists(mp4_file)
    assert not os.path.exists(npy_file)

    # test 3: no trainer, no model, save heatmaps
    csv_file = str(tmpdir.join("test3.csv"))
    mp4_file = str(tmpdir.join("test3.mp4"))
    npy_file = csv_file.replace(".csv", "_heatmaps.npy")
    export_predictions_and_labeled_video(
        video_file=video_list[0],
        cfg=cfg_tmp,
        prediction_csv_file=csv_file,
        ckpt_file=ckpt_callback.best_model_path,
        trainer=None,
        model=None,
        data_module=heatmap_data_module,
        labeled_mp4_file=mp4_file,
        save_heatmaps=True,
    )
    assert os.path.exists(csv_file)
    assert os.path.exists(mp4_file)
    assert os.path.exists(npy_file)

    # test 4: raise proper error
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
            save_heatmaps=False,
        )

    # remove tensors from gpu
    del loss_factories
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()
