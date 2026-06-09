import contextlib
import copy
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from lightning_pose.data import get_data_module
from lightning_pose.train import _evaluate_on_training_dataset, calculate_steps_per_epoch, train


# TODO: Replace with contextlib.chdir in python 3.11.
@contextlib.contextmanager
def chdir(dir):
    pwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(pwd)


def _test_cfg(cfg):

    # copy config and update paths
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.eval.test_videos_directory = cfg_tmp.data.video_dir

    # don't train for long
    cfg_tmp.training.min_epochs = 2
    cfg_tmp.training.max_epochs = 2
    cfg_tmp.training.check_val_every_n_epoch = 1
    cfg_tmp.training.lr_scheduler_params.multisteplr.milestones = [1, 2]
    cfg_tmp.training.log_every_n_steps = 1
    cfg_tmp.training.limit_train_batches = 2

    # train simple model
    cfg_tmp.model.losses_to_use = []

    # predict on vid
    cfg_tmp.eval.predict_vids_after_training = True
    cfg_tmp.eval.save_vids_after_training = True

    return cfg_tmp


def test_train_singleview(cfg, tmp_path):
    cfg = _test_cfg(cfg)
    cfg.model.model_type = "heatmap"

    # temporarily change working directory to temp output directory
    with chdir(tmp_path):
        # train model
        train(cfg)

    # ensure labeled data was properly processed
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "CollectedData.csv").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "predictions_pca_multiview_error.csv").is_file()
    assert (tmp_path / "predictions_pca_singleview_error.csv").is_file()
    assert (tmp_path / "predictions_pixel_error.csv").is_file()

    # ensure video data was properly processed
    assert (tmp_path / "video_preds" / "test_vid.csv").is_file()
    assert (tmp_path / "video_preds" / "test_vid_pca_multiview_error.csv").is_file()
    assert (tmp_path / "video_preds" / "test_vid_pca_singleview_error.csv").is_file()
    assert (tmp_path / "video_preds" / "test_vid_temporal_norm.csv").is_file()
    assert (
        tmp_path / "video_preds" / "labeled_videos" / "test_vid_labeled.mp4"
    ).is_file()


@pytest.mark.skip(reason="Not yet implemented in Model class.")
def test_train_singleview_detector_outputs(cfg, tmp_path):
    cfg = _test_cfg(cfg)
    cfg.eval.predict_vids_after_training = False
    with open_dict(cfg):
        cfg.detector = OmegaConf.create(
            {"crop_ratio": 1.5, "anchor_keypoints": ["nose_top", "tailMid_bot"]}
        )

    # temporarily change working directory to temp output directory
    detector_model_dir = tmp_path / "detector_model"
    detector_model_dir.mkdir()

    with chdir(detector_model_dir):
        # train model
        detector_model = train(cfg)

    # ensure cropped images were properly processed
    assert (
        detector_model_dir / "cropped_images" / "labeled-data" / "img00.png"
    ).is_file()
    assert (
        detector_model_dir / "cropped_images" / "labeled-data" / "img92.png"
    ).is_file()
    image_pred_dir = detector_model_dir / "image_preds" / "CollectedData.csv"
    assert (image_pred_dir / "bbox.csv").is_file()
    assert (image_pred_dir / "predictions.csv").is_file()
    assert (image_pred_dir / "predictions_pca_multiview_error.csv").is_file()
    assert (image_pred_dir / "predictions_pca_singleview_error.csv").is_file()
    assert (image_pred_dir / "predictions_pixel_error.csv").is_file()
    assert (image_pred_dir / "cropped_CollectedData.csv").is_file()

    # ensure cropped videos were properly processed
    # assert (detector_model_dir / "cropped_videos" / "test_vid.mp4").is_file()
    # assert (detector_model_dir / "cropped_videos" / "test_vid_bbox.csv").is_file()

    del cfg.detector
    pose_model_dir = tmp_path / "pose_model"
    pose_model_dir.mkdir()
    with chdir(pose_model_dir):
        # train model
        train(cfg, detector_model)  # type: ignore[arg-type]

    # ensure labeled data was properly processed
    assert (pose_model_dir / "config.yaml").is_file()
    assert (pose_model_dir / "cropped_CollectedData.csv").is_file()
    image_pred_dir = pose_model_dir / "image_preds" / "cropped_CollectedData.csv"
    assert (image_pred_dir / "predictions.csv").is_file()
    assert (image_pred_dir / "predictions_pixel_error.csv").is_file()


def test_train_multiview(cfg_multiview, tmp_path):
    from lightning_pose.train import train

    cfg = _test_cfg(cfg_multiview)
    cfg.model.model_type = "heatmap_multiview_transformer"
    cfg.model.backbone = "vits_dino"

    # temporarily change working directory to temp output directory
    with chdir(tmp_path):
        # train model
        train(cfg)

    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "top.csv").is_file()
    assert (tmp_path / "bot.csv").is_file()

    for view in ["top", "bot"]:
        # ensure labeled data was properly processed
        assert (tmp_path / f"predictions_{view}.csv").is_file()
        assert (tmp_path / f"predictions_{view}_pixel_error.csv").is_file()
        # assert (tmp_path / f"predictions_{view}_pca_multiview_error.csv").is_file()
        # assert (tmp_path / f"predictions_{view}_pca_singleview_error.csv").is_file()

        # ensure video data was properly processed
        assert (tmp_path / "video_preds" / f"test_vid_{view}.csv").is_file()
        assert (
            tmp_path / "video_preds" / f"test_vid_{view}_temporal_norm.csv"
        ).is_file()
        # assert (tmp_path / "video_preds", f"test_vid_{view}_pca_multiview_error.csv").is_file()
        # assert (tmp_path / "video_preds", f"test_vid_{view}_pca_singleview_error.csv").is_file()
        assert (
            tmp_path / "video_preds" / "labeled_videos" / f"test_vid_{view}_labeled.mp4"
        ).is_file()


# Multi-GPU tests must be run as their own scripts due to DDP.
# https://github.com/Lightning-AI/pytorch-lightning/issues/4397#issuecomment-722743582
# Our multi-GPU tests currently just ensure the train script finishes with status code 0.
def _execute_multi_gpu_test(cfg, tmp_path, pytestconfig):
    # set output directory to {tmp_path}/output.
    cfg = OmegaConf.merge(
        cfg, OmegaConf.create({"hydra": {"run": {"dir": tmp_path / "output"}}})
    )

    # Saves config in {tmp_dir}/config.yaml so train_hydra can read from it.
    OmegaConf.save(cfg, tmp_path / "config.yaml")

    # Add git repo directory to PYTHONPATH.
    env = dict(os.environ)
    assert "PYTHONPATH" not in env
    # If PYTHONPATH exists in env, we'd need to append the following path using ":" as a delimiter
    env["PYTHONPATH"] = pytestconfig.rootpath

    # Run train_hydra script.
    process = subprocess.run(
        [
            "python",
            pytestconfig.rootpath / "scripts" / "train_hydra.py",
            f"--config-path={tmp_path}",
            "--config-name=config",
        ],
        env=env,
        check=True,
    )
    assert process.returncode == 0


@pytest.mark.multigpu  # allow running only multi_gpu tests with `pytest -m multigpu`
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs required.")
def test_train_multi_gpu_supervised(cfg, tmp_path, pytestconfig):
    cfg = _test_cfg(cfg)
    # Make the batches large enough to test 2 batches on each GPU.
    cfg.training.num_gpus = 2
    cfg.training.train_batch_size = 4
    cfg.training.val_batch_size = 8

    _execute_multi_gpu_test(cfg, tmp_path, pytestconfig)


@pytest.mark.multigpu  # allow running only multi_gpu tests with `pytest -m multigpu`
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs required.")
def test_train_multi_gpu_unsupervised(cfg, tmp_path, pytestconfig):
    cfg = _test_cfg(cfg)
    # Make the batches large enough to test 2 batches on each GPU.
    cfg.training.num_gpus = 2
    cfg.training.train_batch_size = 4
    cfg.training.val_batch_size = 8
    cfg.dali.context.train.batch_size = 16
    cfg.model.model_type = "heatmap_mhcrnn"
    cfg.model.losses_to_use = ["pca_singleview", "pca_multiview", "temporal"]

    _execute_multi_gpu_test(cfg, tmp_path, pytestconfig)


class TestEvaluateOnTrainingDataset:
    """Test the _evaluate_on_training_dataset function."""

    def _make_model(self, tmp_path: Path, csv_stem: str = 'CollectedData') -> MagicMock:
        """Build a minimal single-view model mock."""
        model = MagicMock()
        model.config.is_single_view.return_value = True
        model.config.is_multi_view.return_value = False
        model.config.cfg.data.csv_file = f'{csv_stem}.csv'
        model.config.cfg.data.data_dir = str(tmp_path)
        model.model_dir = tmp_path
        model.image_preds_dir.return_value = tmp_path / 'image_preds'
        return model

    def _make_model_multiview(self, tmp_path: Path, csv_stem: str = 'CollectedData') -> MagicMock:
        """Build a minimal multi-view model mock."""
        model = MagicMock()
        model.config.is_single_view.return_value = False
        model.config.is_multi_view.return_value = True
        model.config.cfg.data.csv_file = [
            f'{csv_stem}_top.csv',
            f'{csv_stem}_bot.csv',
        ]
        model.config.cfg.data.data_dir = str(tmp_path)
        model.config.cfg.data.view_names = ['top', 'bot']
        model.config.cfg.data.get.return_value = None
        model.model_dir = tmp_path
        model.image_preds_dir.return_value = tmp_path / 'image_preds'
        return model

    def test_evaluate_on_training_dataset_no_suffix(self, tmp_path: Path) -> None:
        """No suffix: calls predict on the base CSV and copies output files."""
        model = self._make_model(tmp_path)
        csv_file = tmp_path / 'CollectedData.csv'
        csv_file.touch()
        image_preds_dir = tmp_path / 'image_preds' / 'CollectedData.csv'
        image_preds_dir.mkdir(parents=True)
        pred_file = image_preds_dir / 'predictions.csv'
        pred_file.write_text('data')

        _evaluate_on_training_dataset(model)

        model.predict_on_label_csv.assert_called_once_with(
            csv_file=csv_file,
            data_dir=str(tmp_path),
            compute_metrics=True,
            add_train_val_test_set=True,
        )
        assert (tmp_path / 'predictions.csv').exists()

    def test_evaluate_on_training_dataset_suffix_new(self, tmp_path: Path) -> None:
        """suffix='_new': calls predict on the _new CSV and copies with _new suffix."""
        model = self._make_model(tmp_path)
        csv_file = tmp_path / 'CollectedData_new.csv'
        csv_file.touch()
        image_preds_dir = tmp_path / 'image_preds' / 'CollectedData_new.csv'
        image_preds_dir.mkdir(parents=True)
        pred_file = image_preds_dir / 'predictions.csv'
        pred_file.write_text('data')

        _evaluate_on_training_dataset(model, suffix='_new')

        model.predict_on_label_csv.assert_called_once_with(
            csv_file=csv_file,
            data_dir=str(tmp_path),
            compute_metrics=True,
            add_train_val_test_set=False,
        )
        assert (tmp_path / 'predictions_new.csv').exists()

    def test_evaluate_on_training_dataset_suffix_test(self, tmp_path: Path) -> None:
        """suffix='_test': calls predict on the _test CSV and copies with _test suffix."""
        model = self._make_model(tmp_path)
        csv_file = tmp_path / 'CollectedData_test.csv'
        csv_file.touch()
        image_preds_dir = tmp_path / 'image_preds' / 'CollectedData_test.csv'
        image_preds_dir.mkdir(parents=True)
        pred_file = image_preds_dir / 'predictions.csv'
        pred_file.write_text('data')

        _evaluate_on_training_dataset(model, suffix='_test')

        model.predict_on_label_csv.assert_called_once_with(
            csv_file=csv_file,
            data_dir=str(tmp_path),
            compute_metrics=True,
            add_train_val_test_set=False,
        )
        assert (tmp_path / 'predictions_test.csv').exists()

    def test_evaluate_on_training_dataset_suffix_missing_file(self, tmp_path: Path) -> None:
        """suffix given but file absent: returns early without calling predict."""
        model = self._make_model(tmp_path)
        # do NOT create CollectedData_new.csv or CollectedData_test.csv

        for suffix in ('_new', '_test'):
            model.predict_on_label_csv.reset_mock()
            _evaluate_on_training_dataset(model, suffix=suffix)
            model.predict_on_label_csv.assert_not_called()

    def test_evaluate_on_training_dataset_multiview_no_suffix(self, tmp_path: Path) -> None:
        """Multi-view, no suffix: calls predict_on_label_csv_multiview."""
        model = self._make_model_multiview(tmp_path)
        for stem in ('CollectedData_top', 'CollectedData_bot'):
            (tmp_path / f'{stem}.csv').touch()
            preds_dir = tmp_path / 'image_preds' / f'{stem}.csv'
            preds_dir.mkdir(parents=True)
            (preds_dir / 'predictions.csv').write_text('data')

        _evaluate_on_training_dataset(model)

        model.predict_on_label_csv_multiview.assert_called_once()
        call_kwargs = model.predict_on_label_csv_multiview.call_args.kwargs
        assert call_kwargs['add_train_val_test_set'] is True
        assert (tmp_path / 'predictions_top.csv').exists()
        assert (tmp_path / 'predictions_bot.csv').exists()

    def test_evaluate_on_training_dataset_multiview_suffix_test(self, tmp_path: Path) -> None:
        """Multi-view, suffix='_test': uses _test CSV files and copies with _test suffix."""
        model = self._make_model_multiview(tmp_path)
        for stem in ('CollectedData_top_test', 'CollectedData_bot_test'):
            (tmp_path / f'{stem}.csv').touch()
            preds_dir = tmp_path / 'image_preds' / f'{stem}.csv'
            preds_dir.mkdir(parents=True)
            (preds_dir / 'predictions.csv').write_text('data')

        _evaluate_on_training_dataset(model, suffix='_test')

        model.predict_on_label_csv_multiview.assert_called_once()
        call_kwargs = model.predict_on_label_csv_multiview.call_args.kwargs
        assert call_kwargs['add_train_val_test_set'] is False
        assert (tmp_path / 'predictions_top_test.csv').exists()
        assert (tmp_path / 'predictions_bot_test.csv').exists()


class TestCalculateStepsPerEpoch:
    """Test the calculate_steps_per_epoch function."""

    def test_calculate_steps_per_epoch_supervised(self, cfg, base_dataset):
        """Test the computation of steps per epoch."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []

        # Small number of train frames
        cfg_tmp.training.train_frames = 3
        cfg_tmp.training.train_batch_size = 2
        base_data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
        n_batches = calculate_steps_per_epoch(base_data_module)
        assert n_batches == 2

        # Large number of frames
        cfg_tmp.training.limit_train_batches = None
        cfg_tmp.training.train_frames = 49
        cfg_tmp.training.train_batch_size = 2
        base_data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
        n_batches = calculate_steps_per_epoch(base_data_module)
        assert n_batches == 25  # ceil (49 / 2)

    def test_calculate_steps_per_epoch_unsupervised(self, cfg, base_dataset, toy_data_dir):
        """Test the computation of steps per epoch."""
        video_dir = os.path.join(toy_data_dir, 'videos')
        cfg_tmp = copy.deepcopy(cfg)

        # Small number of train frames - return minimum of 10
        cfg_tmp.training.train_frames = 3
        cfg_tmp.training.train_batch_size = 2
        base_data_module_combined = get_data_module(
            cfg_tmp, dataset=base_dataset, video_dir=video_dir,
        )
        n_batches = calculate_steps_per_epoch(base_data_module_combined)
        assert n_batches == 10

        # Large number of frames
        cfg_tmp.training.limit_train_batches = None
        cfg_tmp.training.train_frames = 49
        cfg_tmp.training.train_batch_size = 2
        base_data_module_combined = get_data_module(
            cfg_tmp, dataset=base_dataset, video_dir=video_dir,
        )
        n_batches = calculate_steps_per_epoch(base_data_module_combined)
        assert n_batches == 25  # ceil (49 / 2)
