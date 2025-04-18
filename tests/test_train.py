import contextlib
import copy
import os
import subprocess

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from lightning_pose.train import train


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
    pwd = os.getcwd()
    # copy config and update paths
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.eval.test_videos_directory = cfg_tmp.data.video_dir

    # don't train for long
    cfg_tmp.training.min_epochs = 2
    cfg_tmp.training.max_epochs = 2
    cfg_tmp.training.check_val_every_n_epoch = 1
    cfg_tmp.training.log_every_n_steps = 1
    cfg_tmp.training.limit_train_batches = 2

    # train simple model
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    # predict on vid
    cfg_tmp.eval.predict_vids_after_training = True
    cfg_tmp.eval.save_vids_after_training = True
    return cfg_tmp


def test_train_singleview(cfg, tmp_path):
    cfg = _test_cfg(cfg)

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
        train(cfg, detector_model)

    # ensure labeled data was properly processed
    assert (pose_model_dir / "config.yaml").is_file()
    assert (pose_model_dir / "cropped_CollectedData.csv").is_file()
    image_pred_dir = pose_model_dir / "image_preds" / "cropped_CollectedData.csv"
    assert (image_pred_dir / "predictions.csv").is_file()
    assert (image_pred_dir / "predictions_pixel_error.csv").is_file()


def test_train_multiview(cfg_multiview, tmp_path):
    from lightning_pose.train import train

    cfg = _test_cfg(cfg_multiview)

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
            f"--config-name=config",
        ],
        env=env,
        check = True,
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
