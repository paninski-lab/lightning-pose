"""Test the predictions module."""

import copy
import gc
import os

import lightning.pytorch as pl
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from lightning_pose.utils.predictions import (
    _get_cfg_file,
    export_predictions_and_labeled_video,
    get_model_class,
    predict_dataset,
    predict_single_video,
)
from lightning_pose.utils.scripts import get_loss_factories, get_model


class TestGetCfgFile:
    """Test the _get_cfg_file function."""

    def test_get_cfg_file_from_string(self, tmp_path):
        """Loads a YAML file from a string path and returns a DictConfig."""
        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('model:\n  backbone: resnet18\n')
        result = _get_cfg_file(str(cfg_file))
        assert isinstance(result, DictConfig)
        assert result.model.backbone == 'resnet18'

    def test_get_cfg_file_from_dictconfig(self):
        """Returns the DictConfig unchanged when one is passed directly."""
        cfg = OmegaConf.create({'model': {'backbone': 'resnet18'}})
        result = _get_cfg_file(cfg)
        assert result is cfg

    def test_get_cfg_file_raises_for_invalid_type(self):
        """Raises ValueError when cfg_file is neither str nor DictConfig."""
        with pytest.raises(ValueError, match='cfg_file must be str or DictConfig'):
            _get_cfg_file(42)


class TestGetModelClass:
    """Test the get_model_class function."""

    def test_get_model_class_supervised_regression(self):
        """Returns RegressionTracker for supervised regression."""
        from lightning_pose.models import RegressionTracker
        assert get_model_class('regression', semi_supervised=False) is RegressionTracker

    def test_get_model_class_supervised_heatmap(self):
        """Returns HeatmapTracker for supervised heatmap."""
        from lightning_pose.models import HeatmapTracker
        assert get_model_class('heatmap', semi_supervised=False) is HeatmapTracker

    def test_get_model_class_supervised_heatmap_mhcrnn(self):
        """Returns HeatmapTrackerMHCRNN for supervised heatmap_mhcrnn."""
        from lightning_pose.models import HeatmapTrackerMHCRNN
        assert get_model_class('heatmap_mhcrnn', semi_supervised=False) is HeatmapTrackerMHCRNN

    def test_get_model_class_supervised_heatmap_multiview_transformer(self):
        """Returns HeatmapTrackerMultiviewTransformer for supervised multiview transformer."""
        from lightning_pose.models import HeatmapTrackerMultiviewTransformer
        assert (
            get_model_class('heatmap_multiview_transformer', semi_supervised=False)
            is HeatmapTrackerMultiviewTransformer
        )

    def test_get_model_class_supervised_raises_for_unknown(self):
        """Raises NotImplementedError for an unrecognised supervised model_type."""
        with pytest.raises(NotImplementedError, match='invalid model_type for a fully supervised'):
            get_model_class('unknown_type', semi_supervised=False)

    def test_get_model_class_semi_supervised_regression(self):
        """Returns SemiSupervisedRegressionTracker for semi-supervised regression."""
        from lightning_pose.models import SemiSupervisedRegressionTracker
        assert (
            get_model_class('regression', semi_supervised=True) is SemiSupervisedRegressionTracker
        )

    def test_get_model_class_semi_supervised_heatmap(self):
        """Returns SemiSupervisedHeatmapTracker for semi-supervised heatmap."""
        from lightning_pose.models import SemiSupervisedHeatmapTracker
        assert get_model_class('heatmap', semi_supervised=True) is SemiSupervisedHeatmapTracker

    def test_get_model_class_semi_supervised_heatmap_mhcrnn(self):
        """Returns SemiSupervisedHeatmapTrackerMHCRNN for semi-supervised heatmap_mhcrnn."""
        from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN
        assert (
            get_model_class('heatmap_mhcrnn', semi_supervised=True)
            is SemiSupervisedHeatmapTrackerMHCRNN
        )

    def test_get_model_class_semi_supervised_heatmap_multiview_transformer(self):
        """Returns SemiSupervisedHeatmapTrackerMultiviewTransformer for semi-supervised variant."""
        from lightning_pose.models import SemiSupervisedHeatmapTrackerMultiviewTransformer
        assert (
            get_model_class('heatmap_multiview_transformer', semi_supervised=True)
            is SemiSupervisedHeatmapTrackerMultiviewTransformer
        )

    def test_get_model_class_semi_supervised_raises_for_unknown(self):
        """Raises NotImplementedError for an unrecognised semi-supervised model_type."""
        with pytest.raises(
            NotImplementedError, match='invalid model_type for a semi-supervised',
        ):
            get_model_class('unknown_type', semi_supervised=True)


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
