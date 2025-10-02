"""Functions for predicting keypoints on labeled datasets and unlabeled videos."""

from __future__ import annotations

import datetime
import gc
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Type

import cv2
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.dali import PrepareDALI
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.utils import count_frames
from lightning_pose.models import ALLOWED_MODELS

if TYPE_CHECKING:
    from lightning_pose.api.model import Model

# to ignore imports for sphix-autoapidoc
__all__ = [
    "PredictionHandler",
    "predict_dataset",
    "predict_single_video",
    "make_dlc_pandas_index",
    "get_model_class",
    "load_model_from_checkpoint",
    "create_labeled_video",
    "export_predictions_and_labeled_video",
]


@typechecked
def _get_cfg_file(cfg_file: str | DictConfig):
    """Load yaml configuration files."""
    if isinstance(cfg_file, str):
        # load configuration file
        with open(cfg_file, "r") as f:
            cfg = OmegaConf.load(f)
    elif isinstance(cfg_file, DictConfig):
        cfg = cfg_file
    else:
        raise ValueError("cfg_file must be str or DictConfig, not %s!" % type(cfg_file))
    return cfg


class PredictionHandler:
    """Convert batches of model outputs into a prediction dataframe."""

    def __init__(
        self,
        cfg: DictConfig,
        data_module: pl.LightningDataModule | None = None,
        video_file: str | None = None,
    ) -> None:
        """

        Args
            cfg
            data_module: Only required for prediction of CSV files.
            video_file: For prediction on video, path to the video file.
                Used to get frame_count.
        """
        if data_module is None and video_file is None:
            raise ValueError("must pass either data_module or video_file")

        if cfg.data.get("keypoint_names", None) is None:
            raise ValueError("must include `keypoint_names` field in cfg.data")

        self.cfg = cfg
        self.data_module = data_module
        self.video_file = video_file

    @property
    def frame_count(self) -> int:
        """Returns the number of frames in the video or the labeled dataset"""
        if self.video_file is not None:
            return count_frames(self.video_file)
        else:
            return len(self.data_module.dataset)

    @property
    def keypoint_names(self):
        return list(self.cfg.data.keypoint_names)

    @property
    def do_context(self):
        if self.data_module:
            return self.data_module.dataset.do_context
        else:
            return self.cfg.model.model_type == "heatmap_mhcrnn"

    def unpack_preds(
        self,
        preds: list[
            Tuple[
                TensorType["batch", "two_times_num_keypoints"],
                TensorType["batch", "num_keypoints"],
            ]
        ],
    ) -> Tuple[
        TensorType["num_frames", "two_times_num_keypoints"],
        TensorType["num_frames", "num_keypoints"],
    ]:
        """unpack list of preds coming out from pl.trainer.predict, confs tuples into tensors.
        It still returns unnecessary final rows, which should be discarded at the dataframe stage.
        This works for the output of predict_loader, suitable for
        batch_size=1, sequence_length=16, step=16
        """
        # stack the predictions into rows.
        # loop over the batches, and stack
        stacked_preds = torch.vstack([pred[0] for pred in preds])
        stacked_confs = torch.vstack([pred[1] for pred in preds])

        if self.video_file is not None:  # dealing with dali loaders

            # DB: this used to be an else but I think it should apply to all dataloaders now
            # first we chop off the last few rows that are not part of the video
            # next:
            # for baseline: chop extra empty frames from last sequence.
            num_rows_to_discard = stacked_preds.shape[0] - self.frame_count
            if num_rows_to_discard > 0:
                stacked_preds = stacked_preds[:-num_rows_to_discard]
                stacked_confs = stacked_confs[:-num_rows_to_discard]
            # for context: missing first two frames, have to handle with the last two frames still

            if self.do_context:
                # fix shifts in the context model
                stacked_preds = self.fix_context_preds_confs(stacked_preds)
                if self.cfg.model.model_type == "heatmap_mhcrnn":
                    stacked_confs = self.fix_context_preds_confs(
                        stacked_confs, zero_pad_confidence=False
                    )
                else:
                    stacked_confs = self.fix_context_preds_confs(
                        stacked_confs, zero_pad_confidence=True
                    )
            # else:
            # in this dataloader, the last sequence has a few extra frames.
        return stacked_preds, stacked_confs

    def fix_context_preds_confs(
        self, stacked_preds: TensorType, zero_pad_confidence: bool = False
    ):
        """
        In the context model, ind=0 is associated with image[2], and ind=1 is associated with
        image[3], so we need to shift the predictions and confidences by two and eliminate the
        edges.
        NOTE: confidences are not zero in the first and last two images, they are instead replicas
        of images[-2] and images[-3]
        """
        # first pad the first two rows for which we have no valid preds.
        preds_1 = torch.tile(stacked_preds[0], (2, 1))  # copying twice the prediction for image[2]
        preds_2 = stacked_preds[0:-2]  # throw out the last two rows.
        preds_combined = torch.vstack([preds_1, preds_2])
        # repat the last one twice
        if preds_combined.shape[0] == self.frame_count:
            # i.e., after concat this has the length of the video.
            # we don't have valid predictions for the last two elements, so we pad with element -3
            preds_combined[-2:, :] = preds_combined[-3, :]
        else:
            # we don't have as many predictions as frames; pad with final entry which is valid.
            n_pad = self.frame_count - preds_combined.shape[0]
            preds_combined = torch.vstack(
                [preds_combined, torch.tile(preds_combined[0], (n_pad, 1))]
            )

        if zero_pad_confidence:
            # zeroing out those first and last two rows (after we've shifted everything above)
            preds_combined[:2, :] = 0.0
            preds_combined[-2:, :] = 0.0

        return preds_combined

    @staticmethod
    def make_pred_arr_undo_resize(
        keypoints_np: np.array,
        confidence_np: np.array,
    ) -> np.array:
        """Resize keypoints and add confidences into one numpy array.

        Args:
            keypoints_np: shape (n_frames, n_keypoints * 2)
            confidence_np: shape (n_frames, n_keypoints)

        Returns:
            np.ndarray: cols are (bp0_x, bp0_y, bp0_likelihood, bp1_x, bp1_y, ...)

        """
        # check num frames in the dataset
        assert keypoints_np.shape[0] == confidence_np.shape[0]
        # check we have two (x,y) coordinates and a single likelihood value
        assert keypoints_np.shape[1] == confidence_np.shape[1] * 2

        num_joints = confidence_np.shape[-1]  # model.num_keypoints
        predictions = np.zeros((keypoints_np.shape[0], num_joints * 3))
        predictions[:, 0] = np.arange(keypoints_np.shape[0])
        predictions[:, 0::3] = keypoints_np[:, 0::2]
        predictions[:, 1::3] = keypoints_np[:, 1::2]
        predictions[:, 2::3] = confidence_np

        return predictions

    def make_dlc_pandas_index(self, keypoint_names: list | None = None) -> pd.MultiIndex:
        return make_dlc_pandas_index(
            cfg=self.cfg, keypoint_names=keypoint_names or self.keypoint_names
        )

    def add_split_indices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add split indices to the dataframe."""
        df["set"] = np.array(["unused"] * df.shape[0])

        dataset_split_indices = {
            "train": self.data_module.train_dataset.indices,
            "validation": self.data_module.val_dataset.indices,
            "test": self.data_module.test_dataset.indices,
        }

        for key, val in dataset_split_indices.items():
            df.loc[val, ("set", "", "")] = np.repeat(key, len(val))
        return df

    def __call__(
        self,
        preds: list[
            Tuple[
                TensorType["batch", "two_times_num_keypoints"],
                TensorType["batch", "num_keypoints"],
            ]
        ],
        is_multiview_video: bool = False,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Call this function to get a pandas dataframe of the predictions for a single video.
        Assuming you've already run trainer.predict(), and have a list of Tuple predictions.

        Args:
            preds: list of tuples of (predictions, confidences)
            is_multiview_video: specify True when you are using multiview video prediction
                dataloader, i.e. for heatmap_multiview.

        Returns:
            pd.DataFrame: index is (frame, bodypart, x, y, likelihood)

        """
        stacked_preds, stacked_confs = self.unpack_preds(preds=preds)
        if (
            self.cfg.data.get("view_names", None)
            and len(self.cfg.data.view_names) > 1
            and (self.video_file is None or is_multiview_video)
        ):
            # NOTE: if self.video_file is not None assume we are processing one view at a time, and
            # move to the `else` block below.
            # UPDATE: No longer true, added is_multiview_video mode.
            num_keypoints = len(self.keypoint_names)
            view_to_df = {}
            for view_idx, view_name in enumerate(self.cfg.data.view_names):
                idx_beg = view_idx * num_keypoints
                idx_end = idx_beg + num_keypoints
                stacked_preds_single = stacked_preds[:, idx_beg * 2:idx_end * 2]
                stacked_confs_single = stacked_confs[:, idx_beg:idx_end]
                pred_arr = self.make_pred_arr_undo_resize(
                    stacked_preds_single.cpu().numpy(), stacked_confs_single.cpu().numpy()
                )
                pdindex = self.make_dlc_pandas_index(self.keypoint_names)
                df = pd.DataFrame(pred_arr, columns=pdindex)
                view_to_df[view_name] = df
                if self.video_file is None:
                    # specify which image is train/test/val/unused
                    df = self.add_split_indices_to_df(df)
                    df.index = self.data_module.dataset.dataset[view_name].image_names
            retval = view_to_df
        else:
            pred_arr = self.make_pred_arr_undo_resize(
                stacked_preds.cpu().numpy(), stacked_confs.cpu().numpy()
            )
            pdindex = self.make_dlc_pandas_index()
            df = pd.DataFrame(pred_arr, columns=pdindex)
            if self.video_file is None:
                # specify which image is train/test/val/unused
                df = self.add_split_indices_to_df(df)
                df.index = self.data_module.dataset.image_names
            retval = df

        return retval


@typechecked
def predict_dataset(
    cfg: DictConfig,
    data_module: BaseDataModule,
    preds_file: str | list[str],
    ckpt_file: str | None = None,
    trainer: pl.Trainer | None = None,
    model: ALLOWED_MODELS | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Save predicted keypoints for a labeled dataset.

    Args:
        cfg: hydra config
        data_module: data module that contains dataloaders for train, val, test splits
        preds_file: path for the predictions .csv file
        ckpt_file: absolute path to the checkpoint of your trained model; requires .ckpt suffix
        trainer: pl.Trainer object
        model: Lightning Module

    Returns:
        pandas dataframe with predictions or dict with dataframe of predictions for each view

    """

    delete_model = False
    if model is None:
        model = load_model_from_checkpoint(
            cfg=cfg, ckpt_file=ckpt_file, eval=True, data_module=data_module,
        )
        delete_model = True

    delete_trainer = False
    if trainer is None:
        trainer = pl.Trainer(devices=1, accelerator="auto", logger=False)
        delete_trainer = True

    labeled_preds = trainer.predict(
        model=model,
        dataloaders=data_module.full_labeled_dataloader(),
        return_predictions=True,
    )

    pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=None)
    labeled_preds_df = pred_handler(preds=labeled_preds)
    if isinstance(labeled_preds_df, dict):
        if isinstance(preds_file, str):
            # old logic used to save to <predictions>_<view_name>.csv
            for view_name, df in labeled_preds_df.items():
                df.to_csv(preds_file.replace(".csv", f"_{view_name}.csv"))
        elif isinstance(preds_file, list):
            # preds_file is a list of views corresponding to cfg.data.view_names.
            # this allows the caller to specify the output locations more flexibly.

            # Check the order of labeled_preds_df keys matches the order of the views in the cfg.
            assert list(labeled_preds_df.keys()) == list(cfg.data.view_names)

            for (view_name, df), _pred_file in zip(labeled_preds_df.items(), preds_file):
                df.to_csv(_pred_file)

    else:
        labeled_preds_df.to_csv(preds_file)

    # clear up memory
    if delete_model:
        del model
    if delete_trainer:
        del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return labeled_preds_df


@typechecked
def predict_single_video(
    cfg_file: str | DictConfig,
    video_file: str,
    preds_file: str,
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
    ckpt_file: str | None = None,
    trainer: pl.Trainer | None = None,
    model: ALLOWED_MODELS | None = None,
) -> pd.DataFrame:
    """This function is deprecated. Use `predict_video` instead.

    Make predictions for a single video, loading frame sequences using DALI.

    This function initializes a DALI pipeline, prepares a dataloader, and passes it on
    to _make_predictions().

    Args:
        cfg_file: either a hydra config or a path pointing to one, with all the model specs.
            needed for loading the model.
        video_file: absolute path to a single video you want to get predictions for, .mp4 file.
        preds_file: absolute filename for the predictions .csv file
        data_module: contains keypoint names for prediction file
        ckpt_file: absolute path to the checkpoint of your trained model; requires .ckpt suffix
        trainer: pl.Trainer object
        model: Lightning Module

    Returns:
        pandas dataframe with predictions

    """
    warnings.warn(
        "predict_single_video is deprecated. Use `predict_video` instead.",
        DeprecationWarning,
    )

    cfg = _get_cfg_file(cfg_file=cfg_file).copy()  # copy because we update imgaug field below

    delete_model = False
    if model is None:
        skip_data_module = True if data_module is None else False
        model = load_model_from_checkpoint(
            cfg=cfg, ckpt_file=ckpt_file, eval=True, data_module=data_module,
            skip_data_module=skip_data_module,
        )
        delete_model = True

    delete_trainer = False
    if trainer is None:
        trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        delete_trainer = True

    # ----------------------------------------------------------------------------------
    # set up
    # ----------------------------------------------------------------------------------
    # initialize
    model_type = "context" if cfg.model.model_type == "heatmap_mhcrnn" else "base"
    cfg.training.imgaug = "default"
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type=model_type,
        dali_config=cfg.dali,
        filenames=[video_file],
        resize_dims=[
            cfg.data.image_resize_dims.height,
            cfg.data.image_resize_dims.width,
        ],
    )
    # get loader
    predict_loader = vid_pred_class()

    # initialize prediction handler class
    pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=video_file)

    # ----------------------------------------------------------------------------------
    # compute predictions
    # ----------------------------------------------------------------------------------
    preds = trainer.predict(
        model=model,
        dataloaders=predict_loader,
        return_predictions=True,
    )

    # call this instance on a single vid's preds
    preds_df = pred_handler(preds=preds)
    # save the predictions to a csv; create directory if it doesn't exist
    os.makedirs(os.path.dirname(preds_file), exist_ok=True)
    preds_df.to_csv(preds_file)

    # clear up memory
    if delete_model:
        del model
    if delete_trainer:
        del trainer
    del predict_loader
    gc.collect()
    torch.cuda.empty_cache()

    return preds_df


@typechecked
def make_dlc_pandas_index(cfg: DictConfig, keypoint_names: list[str]) -> pd.MultiIndex:
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % cfg.model.model_type], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex


@typechecked
def get_model_class(map_type: str, semi_supervised: bool) -> Type[ALLOWED_MODELS]:
    """[summary]

    Args:
        map_type (str): "regression" | "heatmap"
        semi_supervised (bool): True if you want to use unlabeled videos

    Returns:
        a ptl model class to be initialized outside of this function.

    """
    if not semi_supervised:
        if map_type == "regression":
            from lightning_pose.models import RegressionTracker as Model
        elif map_type == "heatmap":
            from lightning_pose.models import HeatmapTracker as Model
        elif map_type == "heatmap_mhcrnn":
            from lightning_pose.models import HeatmapTrackerMHCRNN as Model
        elif map_type == "heatmap_multiview":
            from lightning_pose.models import HeatmapTrackerMultiview as Model
        elif map_type == "heatmap_multiview_multihead":
            from lightning_pose.models import HeatmapTrackerMultiviewMultihead as Model
        elif map_type == "heatmap_multiview_transformer":
            from lightning_pose.models import HeatmapTrackerMultiviewTransformer as Model
        else:
            raise NotImplementedError(
                f"{map_type} is an invalid model_type for a fully supervised model"
            )
    else:
        if map_type == "regression":
            from lightning_pose.models import SemiSupervisedRegressionTracker as Model
        elif map_type == "heatmap":
            from lightning_pose.models import SemiSupervisedHeatmapTracker as Model
        elif map_type == "heatmap_mhcrnn":
            from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN as Model
        elif map_type == "heatmap_multiview_transformer":
            from lightning_pose.models import (
                SemiSupervisedHeatmapTrackerMultiviewTransformer as Model,
            )
        else:
            raise NotImplementedError(
                f"{map_type} is an invalid model_type for a semi-supervised model"
            )

    return Model


@typechecked
def load_model_from_checkpoint(
    cfg: DictConfig,
    ckpt_file: str,
    eval: bool = False,
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
    skip_data_module: bool = False,
) -> ALLOWED_MODELS:
    """Load Lightning Pose model from checkpoint file.

    Args:
        cfg: model config
        ckpt_file: absolute path to model checkpoint
        eval: True for eval mode, False for train mode
        data_module: used to initialize unsupervised losses
        skip_data_module: if `data_module` is not None this is ignored.
            If False and `data_module=None`, a data module is created from the config file and
            unsupervised losses are accessible in the model.
            If True and `data_module=None`, the unsupervised losses are not accessible in the
            model; this is recommended for running inference on new videos

    Returns:
        model as a Lightning Module

    """
    from lightning_pose.utils.io import (
        check_if_semi_supervised,
        return_absolute_data_paths,
    )
    from lightning_pose.utils.scripts import (
        get_data_module,
        get_dataset,
        get_imgaug_transform,
        get_loss_factories,
    )

    # get loss factories
    delete_extras = False
    if not data_module and not skip_data_module:
        # create data module if not provided as input
        delete_extras = True
        data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
        imgaug_transform = get_imgaug_transform(cfg=cfg)
        dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
        data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
    if not data_module:
        loss_factories = {"supervised": None, "unsupervised": None}
    else:
        loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # pick the right model class
    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    ModelClass = get_model_class(
        map_type=cfg.model.model_type,
        semi_supervised=semi_supervised,
    )

    # initialize a model instance, load weights from .ckpt file (fix state_dict keys if needed)
    try:
        checkpoint = torch.load(ckpt_file)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with default settings: {e}")
        print("Attempting to load with weights_only=False...")
        checkpoint = torch.load(ckpt_file, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # fix state dict key mismatch for upsampling layers
    # old checkpoints may have 'upsampling_layers' without 'head.' prefix
    keys_remapped = False
    for key in list(state_dict.keys()):
        if key.startswith("upsampling_layers."):
            # Add 'head.' prefix if missing
            new_key = "head." + key
            state_dict[new_key] = state_dict.pop(key)
            keys_remapped = True

    if keys_remapped:
        # save the fixed state dict back to checkpoint
        checkpoint["state_dict"] = state_dict
        # create a temporary file with the fixed checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            fixed_ckpt_file = tmp_file.name
    else:
        fixed_ckpt_file = ckpt_file

    if semi_supervised:
        model = ModelClass.load_from_checkpoint(
            fixed_ckpt_file,
            loss_factory=loss_factories["supervised"],
            loss_factory_unsupervised=loss_factories["unsupervised"],
            strict=False,
        )
    else:
        model = ModelClass.load_from_checkpoint(
            fixed_ckpt_file,
            loss_factory=loss_factories["supervised"],
            strict=False,
        )

    # clean up temporary file if created
    if keys_remapped:
        import os
        os.unlink(fixed_ckpt_file)

    if eval:
        model.eval()

    # clear up memory
    if delete_extras:
        del imgaug_transform
        del dataset
        del data_module
    del loss_factories
    torch.cuda.empty_cache()

    return model


@typechecked
def _make_cmap(number_colors: int, cmap: str):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


@typechecked
def create_labeled_video(
    clip: VideoFileClip,
    xs_arr: np.ndarray,
    ys_arr: np.ndarray,
    mask_array: np.ndarray | None = None,
    dotsize: int = 4,
    colormap: str | None = "cool",
    fps: float | None = None,
    output_video_path: str = "movie.mp4",
    start_time: float = 0.0,
) -> None:
    """Helper function for creating annotated videos.
    Args
        clip
        xs_arr: shape T x n_joints
        ys_arr: shape T x n_joints
        mask_array: shape T x n_joints; timepoints/joints with a False entry will not be plotted
        dotsize: size of marker dot on labeled video
        colormap: matplotlib color map for markers
        fps: None to default to fps of original video
        output_video_path: video file name
        start_time: time (in seconds) of video start
    """

    if mask_array is None:
        mask_array = ~np.isnan(xs_arr)

    n_frames, n_keypoints = xs_arr.shape

    # set colormap for each color
    colors = _make_cmap(n_keypoints, cmap=colormap)

    # extract info from clip
    nx, ny = clip.size
    dur = int(clip.duration - clip.start)
    fps_og = clip.fps

    # upsample clip if low resolution; need to do this for dots and text to look nice
    if nx <= 100 or ny <= 100:
        upsample_factor = 2.5
    elif nx <= 192 or ny <= 192:
        upsample_factor = 2
    else:
        upsample_factor = 1

    if upsample_factor > 1:
        clip = clip.resize((upsample_factor * nx, upsample_factor * ny))
        nx, ny = clip.size

    print(f"Duration of video [s]: {np.round(dur, 2)}, recorded at {np.round(fps_og, 2)} fps!")

    def seconds_to_hms(seconds):
        # Convert seconds to a timedelta object
        td = datetime.timedelta(seconds=seconds)

        # Extract hours, minutes, and seconds from the timedelta object
        hours = td // datetime.timedelta(hours=1)
        minutes = (td // datetime.timedelta(minutes=1)) % 60
        seconds = td % datetime.timedelta(minutes=1)

        # Format the hours, minutes, and seconds into a string
        hms_str = f"{hours:02}:{minutes:02}:{seconds.seconds:02}"

        return hms_str

    # add marker to each frame t, where t is in sec
    def add_marker_and_timestamps(get_frame, t):
        image = get_frame(t)
        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * fps_og))
        # ----------------
        # markers
        # ----------------
        if index >= n_frames:
            print(f"add_marker_and_timestamps: Skipped frame {index}")
        else:
            for bpindex in range(n_keypoints):
                if mask_array[index, bpindex]:
                    xc = min(int(upsample_factor * xs_arr[index, bpindex]), nx - 1)
                    yc = min(int(upsample_factor * ys_arr[index, bpindex]), ny - 1)
                    frame = cv2.circle(
                        frame,
                        center=(xc, yc),
                        radius=dotsize,
                        color=colors[bpindex].tolist(),
                        thickness=-1,
                    )
        # ----------------
        # timestamps
        # ----------------
        seconds_from_start = t + start_time
        time_from_start = seconds_to_hms(seconds_from_start)
        idx_from_start = int(np.round(seconds_from_start * fps_og))
        text = f"t={time_from_start}, frame={idx_from_start}"
        # define text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        # calculate the size of the text
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        # calculate the position of the text in the upper-left corner
        offset = 6
        text_x = offset  # offset from the left
        text_y = text_size[1] + offset  # offset from the bottom
        # make black rectangle with a small padding of offset / 2 pixels
        cv2.rectangle(
            frame,
            (text_x - int(offset / 2), text_y + int(offset / 2)),
            (text_x + text_size[0] + int(offset / 2), text_y - text_size[1] - int(offset / 2)),
            (0, 0, 0),  # rectangle color
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # font color
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        return frame

    clip_marked = clip.fl(add_marker_and_timestamps)
    clip_marked.write_videofile(
        output_video_path, codec="libx264", fps=fps or fps_og or 20.0
    )
    clip_marked.close()


@typechecked
def export_predictions_and_labeled_video(
    video_file: str,
    cfg: DictConfig,
    prediction_csv_file: str,
    ckpt_file: str | None = None,
    trainer: pl.Trainer | None = None,
    model: ALLOWED_MODELS | None = None,
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
    labeled_mp4_file: str | None = None,
) -> pd.DataFrame:
    """Deprecated, use `predict_video` and `generate_labeled_video`.

    Export predictions csv and a labeled video for a single video file."""
    warnings.warn(
        "export_predictions_and_labeled_video is deprecated. "
        "Use `predict_video` and `generate_labeled_video` instead.",
        DeprecationWarning,
    )
    if ckpt_file is None and model is None:
        raise ValueError("either 'ckpt_file' or 'model' must be passed")

    # compute predictions
    preds_df = predict_single_video(
        video_file=video_file,
        ckpt_file=ckpt_file,
        cfg_file=cfg,
        preds_file=prediction_csv_file,
        trainer=trainer,
        model=model,
        data_module=data_module,
    )

    # create labeled video
    if labeled_mp4_file is not None:
        generate_labeled_video(
            video_file=video_file,
            preds_df=preds_df,
            output_mp4_file=labeled_mp4_file,
            confidence_thresh_for_vid=cfg.eval.confidence_thresh_for_vid,
            colormap=cfg.eval.get("colormap", "cool")
        )
    return preds_df


def generate_labeled_video(
    video_file: str,
    preds_df: pd.DataFrame,
    output_mp4_file: str,
    confidence_thresh_for_vid: float,
    colormap: str,
):
    os.makedirs(os.path.dirname(output_mp4_file), exist_ok=True)
    # transform df to numpy array
    keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
    xs_arr = keypoints_arr[:, :, 0]
    ys_arr = keypoints_arr[:, :, 1]
    mask_array = keypoints_arr[:, :, 2] > confidence_thresh_for_vid
    # video generation
    video_clip = VideoFileClip(video_file)
    create_labeled_video(
        clip=video_clip,
        xs_arr=xs_arr,
        ys_arr=ys_arr,
        mask_array=mask_array,
        output_video_path=output_mp4_file,
        colormap=colormap,
    )


def predict_video(
    video_file: str | list[str],
    model: Model,
    output_pred_file: str | list[str] | None = None,
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Args:
        video_file: Predict on a video, or for true multiview models, a list of videos
            (order: 1-1 correspondence with cfg.data.view_names).
        model: The model to predict with.
        output_pred_file: (optional) File to save predictions in.
            For multiview, a list of files (1-1 correspondance to cfg.data.view_names).
    """

    is_multiview = not isinstance(video_file, str)

    if is_multiview:
        # Validate output_pred_file is a list
        if output_pred_file is not None and not isinstance(output_pred_file, list):
            raise ValueError(
                "for multiview prediction, 'output_pred_file' should be a list corresponding to "
                "view_names"
            )

        # sanity check 1-1 correspondence of video_file to cfg.data.view_names
        # important since PredictionHandler relies on correspondence to organize the outputted dict
        for single_video_file, view_name in zip(
            video_file, model.config.cfg.data.view_names
        ):
            assert (
                view_name in Path(single_video_file).stem
            ), "expected video_file to correspond 1-1 with cfg.data.view_name"

    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    model_type = "context" if model.config.cfg.model.model_type == "heatmap_mhcrnn" else "base"

    filenames = [video_file] if not is_multiview else [[f] for f in video_file]
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type=model_type,
        dali_config=model.config.cfg.dali,
        # Important: This will be a list of lists for multiview.
        # This will trigger dali to return multiview batches to predict_step.
        filenames=filenames,
        resize_dims=[
            model.config.cfg.data.image_resize_dims.height,
            model.config.cfg.data.image_resize_dims.width,
        ],
    )
    # get loader
    predict_loader = vid_pred_class()

    # initialize prediction handler class
    pred_handler = PredictionHandler(
        cfg=model.config.cfg,
        video_file=video_file[0] if is_multiview else video_file,
    )

    # compute predictions
    preds = trainer.predict(
        model=model.model,
        dataloaders=predict_loader,
        return_predictions=True,
    )

    preds_df = pred_handler(preds=preds, is_multiview_video=is_multiview)

    # Convert to a 1-1 correspondence list similar to video_files, for multiview.
    if isinstance(preds_df, dict):
        preds_df = [
            preds_df[view_name] for view_name in model.config.cfg.data.view_names
        ]

    if output_pred_file is not None:
        # save the predictions to a csv; create directory if it doesn't exist

        if is_multiview:
            for df, output_file in zip(preds_df, output_pred_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file)
        else:
            preds_df.to_csv(output_pred_file)

    # clear up memory
    del model
    del trainer
    del predict_loader
    gc.collect()
    torch.cuda.empty_cache()

    return preds_df
