"""Provide pytest fixtures for the entire test suite.

These fixtures create data and data modules that can be reused by other tests. Their
construction relies heavily on the utility functions provided in `utils/scripts.py`.

"""

import copy
import gc
import os
import shutil
import subprocess
import sys
from typing import Callable, List

import cv2
import imgaug.augmenters as iaa
import lightning.pytorch as pl
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from lightning_pose.data.dali import LitDaliWrapper, PrepareDALI
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import (
    BaseTrackingDataset,
    HeatmapDataset,
    MultiviewHeatmapDataset,
)
from lightning_pose.utils.io import get_videos_in_dir
from lightning_pose.utils.predictions import PredictionHandler
from lightning_pose.utils.scripts import (
    get_callbacks,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)

TOY_DATA_ROOT_DIR = "data/mirror-mouse-example"
TOY_MDATA_ROOT_DIR = "data/mirror-mouse-example_split"


@pytest.fixture
def video_list() -> List[str]:
    return get_videos_in_dir(os.path.join(TOY_DATA_ROOT_DIR, "videos"))


@pytest.fixture
def toy_data_dir() -> str:
    return TOY_DATA_ROOT_DIR


@pytest.fixture
def cfg() -> dict:
    """Load all toy data config file without hydra."""
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    config_file = os.path.join(base_dir, "scripts", "configs", "config_mirror-mouse-example.yaml")
    cfg = OmegaConf.load(config_file)
    # make small batches so that we can run on a gpu with limited memory
    cfg.training.train_batch_size = 2
    cfg.training.val_batch_size = 4
    cfg.training.test_batch_size = 4
    cfg.training.imgaug = "dlc"
    cfg.dali.base.train.sequence_length = 6
    cfg.dali.base.predict.sequence_length = 16
    return cfg


@pytest.fixture
def cfg_multiview() -> dict:
    """Load all toy data config file without hydra."""
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    config_file = os.path.join(base_dir, "scripts", "configs", "config_mirror-mouse-example.yaml")
    cfg = OmegaConf.load(config_file)
    # make small batches so that we can run on a gpu with limited memory
    cfg.training.train_batch_size = 2
    cfg.training.val_batch_size = 4
    cfg.training.test_batch_size = 4
    cfg.training.imgaug = "dlc"
    cfg.dali.base.train.sequence_length = 6
    cfg.dali.base.predict.sequence_length = 16
    cfg.data.data_dir = "${LP_ROOT_PATH:}/data/mirror-mouse-example_split"
    cfg.data.csv_file = ["top.csv", "bot.csv"]
    cfg.data.view_names = ["bot", "top"]
    cfg.data.num_keypoints = 7
    cfg.data.keypoint_names = [
        "paw1LH", "paw2LF", "paw3RF", "paw4RH", "tailBase", "tailMid", "nose",
    ]
    cfg.data.columns_for_singleview_pca = [0, 1, 2, 3, 4, 5, 6]
    cfg.data.mirrored_column_matches = [0, 1, 2, 3, 4, 5, 6]
    cfg.model.backbone_pretrained = False

    return cfg


def make_multiview_dataset() -> None:

    # create multiview dataset
    repo_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    base_dir = os.path.join(repo_dir, TOY_DATA_ROOT_DIR)
    split_dir = os.path.join(repo_dir, TOY_MDATA_ROOT_DIR)

    try:
        os.makedirs(split_dir, exist_ok=False)
    except FileExistsError:
        print("Directory Exists! ", split_dir)
        return None

    y_split = 168  # empirically found for the example video

    # copy and split labeled data
    src_dir_ld = os.path.join(base_dir, "labeled-data")

    dst_dir_ld = os.path.join(split_dir, "labeled-data")
    for view in ["top", "bot"]:
        view_dir = os.path.join(dst_dir_ld, f"example_{view}")
        os.makedirs(view_dir, exist_ok=True)

    frames = os.listdir(src_dir_ld)
    for frame in frames:
        frame_file = os.path.join(src_dir_ld, frame)
        img = cv2.cvtColor(cv2.imread(frame_file), cv2.COLOR_BGR2RGB)

        # split views and save
        save_file = os.path.join(dst_dir_ld, "example_top", frame)
        cv2.imwrite(save_file, img[:y_split])
        save_file = os.path.join(dst_dir_ld, "example_bot", frame)
        cv2.imwrite(save_file, img[y_split:])

    # copy and split videos
    src_dir_vids = os.path.join(base_dir, "videos")
    dst_dir_vids = os.path.join(split_dir, "videos")
    os.makedirs(dst_dir_vids, exist_ok=True)
    videos = os.listdir(src_dir_vids)
    for video in videos:
        src_vid = os.path.join(src_dir_vids, video)
        dst_vid_top = os.path.join(dst_dir_vids, video.replace(".mp4", "_top.mp4"))
        dst_vid_bot = os.path.join(dst_dir_vids, video.replace(".mp4", "_bot.mp4"))
        ffmpeg_cmd = f"ffmpeg -i {src_vid} -filter_complex '[0]crop=iw:{y_split}:0:0[top];[0]crop=iw:ih-{y_split}:0:{y_split}[bot]' -map '[top]' {dst_vid_top} -map '[bot]' {dst_vid_bot}"  # noqa: E501
        subprocess.run(ffmpeg_cmd, shell=True)

    # copy and split CollectedData.csv
    src_file = os.path.join(base_dir, "CollectedData.csv")
    dst_file_top = os.path.join(split_dir, "top.csv")
    dst_file_bot = os.path.join(split_dir, "bot.csv")

    df_og = pd.read_csv(src_file, header=[0, 1, 2], index_col=0)
    # just take top view columns
    df_top = df_og.filter(regex="_top")
    # just take bottom view columns
    df_bot = df_og.filter(regex="_bot")
    for col in list(df_bot.columns):
        df_bot.rename({col[1]: col[1].replace("_bot", "")})
    # subtract off split
    df_bot.loc[:, df_bot.columns.get_level_values("coords") == "y"] -= y_split
    # rename indices
    index_top = [
        "/".join([d.split("/")[0], "example_top", d.split("/")[1]]) for d in df_top.index]
    df_top.index = index_top
    index_bot = [
        "/".join([d.split("/")[0], "example_bot", d.split("/")[1]]) for d in df_bot.index]
    df_bot.index = index_bot
    # save

    df_top.to_csv(dst_file_top)
    df_bot.to_csv(dst_file_bot)

    df_top = pd.read_csv(dst_file_top, header=[0], index_col=0)
    df_bot = pd.read_csv(dst_file_bot, header=[0], index_col=0)

    for col in df_top.columns:
        if any(df_top[col] == "obs_top"):
            df_top.drop(columns=[col], inplace=True)

    for col in df_bot.columns:
        if any(df_bot[col] == "obsHigh_bot"):
            df_bot.drop(columns=[col], inplace=True)

    for col in df_bot.columns:
        if any(df_bot[col] == "obsLow_bot"):
            df_bot.drop(columns=[col], inplace=True)

    df_top.replace("_top", "", inplace=True, regex=True)
    df_bot.replace("_bot", "", inplace=True, regex=True)

    df_top.to_csv(dst_file_top)
    df_bot.to_csv(dst_file_bot)


@pytest.fixture
def imgaug_transform(cfg) -> iaa.Sequential:
    """Create basic resizing transform."""
    return get_imgaug_transform(cfg)


@pytest.fixture
def base_dataset(cfg, imgaug_transform) -> BaseTrackingDataset:
    """Create a dataset for regression models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "regression"
    base_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield base_dataset

    # cleanup after all tests have run (no more calls to yield)
    del base_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_dataset(cfg, imgaug_transform) -> HeatmapDataset:
    """Create a dataset for heatmap models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    heatmap_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield heatmap_dataset

    # cleanup after all tests have run (no more calls to yield)
    del heatmap_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def multiview_heatmap_dataset(cfg_multiview, imgaug_transform) -> MultiviewHeatmapDataset:
    """Create a dataset for heatmap models from toy data."""
    make_multiview_dataset()
    # setup
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap"
    multiview_heatmap_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_MDATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield multiview_heatmap_dataset

    # cleanup after all tests have run (no more calls to yield)
    del multiview_heatmap_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_dataset_context(cfg, imgaug_transform) -> HeatmapDataset:
    """Create a dataset for heatmap models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    heatmap_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_DATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield heatmap_dataset

    # cleanup after all tests have run (no more calls to yield)
    del heatmap_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def multiview_heatmap_dataset_context(cfg_multiview, imgaug_transform) -> HeatmapDataset:
    """Create a dataset for heatmap models from toy data."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_mhcrnn"
    heatmap_dataset = get_dataset(
        cfg_tmp, data_dir=TOY_MDATA_ROOT_DIR, imgaug_transform=imgaug_transform
    )

    # return to tests
    yield heatmap_dataset

    # cleanup after all tests have run (no more calls to yield)
    del heatmap_dataset
    torch.cuda.empty_cache()


@pytest.fixture
def base_data_module(cfg, base_dataset) -> BaseDataModule:
    """Create a labeled data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = []
    data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module(cfg, heatmap_dataset) -> BaseDataModule:
    """Create a labeled data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = []
    data_module = get_data_module(cfg_tmp, dataset=heatmap_dataset, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def multiview_heatmap_data_module(cfg_multiview, multiview_heatmap_dataset) -> BaseDataModule:
    """Create a labeled data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.losses_to_use = []
    data_module = get_data_module(cfg_tmp, dataset=multiview_heatmap_dataset, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module_context(cfg, heatmap_dataset_context) -> BaseDataModule:
    """Create a labeled data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = []
    data_module = get_data_module(cfg_tmp, dataset=heatmap_dataset_context, video_dir=None)
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def multiview_heatmap_data_module_context(
    cfg_multiview,
    multiview_heatmap_dataset_context,
) -> BaseDataModule:
    """Create a labeled data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.losses_to_use = []
    data_module = get_data_module(
        cfg_tmp, dataset=multiview_heatmap_dataset_context, video_dir=None,
    )
    data_module.setup()

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def base_data_module_combined(cfg, base_dataset) -> UnlabeledDataModule:
    """Create a combined data module for regression models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = ["temporal"]
    data_module = get_data_module(
        cfg_tmp,
        dataset=base_dataset,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module_combined(cfg, heatmap_dataset) -> UnlabeledDataModule:
    """Create a combined data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = ["temporal"]  # trigger semi-supervised data module
    data_module = get_data_module(
        cfg_tmp,
        dataset=heatmap_dataset,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def multiview_heatmap_data_module_combined(
    cfg_multiview,
    multiview_heatmap_dataset
) -> UnlabeledDataModule:
    """Create a combined data module for multiview heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.losses_to_use = ["pca_multiview"]  # trigger semi-supervised data module
    data_module = get_data_module(
        cfg_tmp,
        dataset=multiview_heatmap_dataset,
        video_dir=os.path.join(cfg_multiview.data.data_dir, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def heatmap_data_module_combined_context(cfg, heatmap_dataset_context) -> UnlabeledDataModule:
    """Create a combined data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = ["temporal"]  # trigger semi-supervised data module
    data_module = get_data_module(
        cfg_tmp,
        dataset=heatmap_dataset_context,
        video_dir=os.path.join(TOY_DATA_ROOT_DIR, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def multiview_heatmap_data_module_combined_context(
    cfg_multiview,
    multiview_heatmap_dataset_context,
) -> UnlabeledDataModule:
    """Create a combined data module for heatmap models."""

    # setup
    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.losses_to_use = ["pca_multiview"]  # trigger semi-supervised data module
    data_module = get_data_module(
        cfg_tmp,
        dataset=multiview_heatmap_dataset_context,
        video_dir=os.path.join(cfg_multiview.data.data_dir, "videos"),
    )
    # data_module.setup()  # already done in UnlabeledDataModule constructor

    # return to tests
    yield data_module

    # cleanup after all tests have run (no more calls to yield)
    del data_module
    torch.cuda.empty_cache()


@pytest.fixture
def video_dataloader(cfg, base_dataset, video_list) -> LitDaliWrapper:
    """Create a prediction dataloader for a new video."""

    # setup
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="base",
        dali_config=cfg.dali,
        filenames=video_list,
        resize_dims=[base_dataset.height, base_dataset.width],
    )
    video_dataloader = vid_pred_class()

    # return to tests
    yield video_dataloader

    # cleanup after all tests have run (no more calls to yield)
    del video_dataloader
    torch.cuda.empty_cache()


@pytest.fixture
def trainer(cfg) -> pl.Trainer:
    """Create a basic pytorch lightning trainer for testing models."""

    cfg.training.unfreezing_epoch = 1 # exercise unfreezing
    callbacks = get_callbacks(cfg, early_stopping=False, lr_monitor=False, backbone_unfreeze=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=2,
        min_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        limit_train_batches=2,
        num_sanity_val_steps=0,
    )

    return trainer


@pytest.fixture
def remove_logs() -> Callable:
    def _remove_logs():
        base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
        logging_dir = os.path.join(base_dir, "lightning_logs")
        shutil.rmtree(logging_dir)

    return _remove_logs


@pytest.fixture
def run_model_test() -> Callable:

    def _run_model_test(cfg, data_module, video_dataloader, trainer, remove_logs_fn):
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
                trainer.predict(model=model, dataloaders=video_dataloader, return_predictions=True)

        finally:

            # remove tensors from gpu
            del loss_factories
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # clean up logging
            remove_logs_fn()

    return _run_model_test
