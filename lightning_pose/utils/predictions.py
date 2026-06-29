"""Functions for predicting keypoints on labeled datasets and unlabeled videos."""

from __future__ import annotations

import datetime
import gc
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import cv2
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from moviepy import VideoFileClip
from omegaconf import DictConfig, ListConfig

from lightning_pose.callbacks import JSONInferenceProgressTracker
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.utils import count_frames

if TYPE_CHECKING:
    from lightning_pose.api import Model

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__ = [
    "predict_dataset",
    "predict_video",
    "generate_labeled_video",
]


class PredictionHandler:
    """Convert batches of model outputs into a prediction dataframe."""

    def __init__(
        self,
        cfg: DictConfig | ListConfig,
        data_module: BaseDataModule | UnlabeledDataModule | None = None,
        video_file: str | None = None,
    ) -> None:
        """Initialize a PredictionHandler.

        Args:
            cfg: hydra config object.
            data_module: only required for prediction of CSV files.
            video_file: for prediction on video, path to the video file; used to get frame_count.
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
            assert self.data_module is not None
            return len(self.data_module.dataset)  # type: ignore[arg-type]

    @property
    def keypoint_names(self) -> list[str]:
        """List of keypoint name strings from the config.

        Returns:
            List of keypoint names.
        """
        return list(self.cfg.data.keypoint_names)

    @property
    def do_context(self) -> bool:
        """Whether the model/loader uses 5-frame context.

        Returns:
            True if context frames are used, otherwise False.
        """
        if self.data_module:
            return self.data_module.dataset.do_context  # type: ignore[union-attr]
        else:
            return self.cfg.model.model_type == "heatmap_mhcrnn"

    def unpack_preds(
        self,
        preds: list[
            tuple[
                Float[torch.Tensor, "batch two_times_num_keypoints"],
                Float[torch.Tensor, "batch num_keypoints"],
            ]
        ],
    ) -> tuple[
        Float[torch.Tensor, "num_frames two_times_num_keypoints"],
        Float[torch.Tensor, "num_frames num_keypoints"],
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
        self, stacked_preds: torch.Tensor, zero_pad_confidence: bool = False
    ) -> torch.Tensor:
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
        keypoints_np: np.ndarray,
        confidence_np: np.ndarray,
    ) -> np.ndarray:
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
        """Build a DLC-style pandas MultiIndex for labelling prediction DataFrames.

        Args:
            keypoint_names: optional override for the list of keypoint names; defaults to
                ``self.keypoint_names``.

        Returns:
            ``pd.MultiIndex`` with levels ``["scorer", "bodyparts", "coords"]``.
        """
        return make_dlc_pandas_index(
            cfg=self.cfg, keypoint_names=keypoint_names or self.keypoint_names
        )

    def add_split_indices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add split indices to the dataframe."""
        assert self.data_module is not None
        df["set"] = np.array(["unused"] * df.shape[0])

        assert self.data_module.train_dataset is not None
        assert self.data_module.val_dataset is not None
        assert self.data_module.test_dataset is not None
        dataset_split_indices = {
            "train": self.data_module.train_dataset.indices,
            "validation": self.data_module.val_dataset.indices,
            "test": self.data_module.test_dataset.indices,
        }

        for key, val in dataset_split_indices.items():
            df.loc[val, ("set", "", "")] = np.repeat(key, len(val))
        return df

    @overload
    def __call__(
        self,
        preds: list[
            tuple[
                Float[torch.Tensor, "batch two_times_num_keypoints"],
                Float[torch.Tensor, "batch num_keypoints"],
            ]
        ],
        is_multiview_video: Literal[False] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def __call__(
        self,
        preds: list[
            tuple[
                Float[torch.Tensor, "batch two_times_num_keypoints"],
                Float[torch.Tensor, "batch num_keypoints"],
            ]
        ],
        is_multiview_video: Literal[True],
    ) -> dict[str, pd.DataFrame]: ...

    def __call__(
        self,
        preds: list[
            tuple[
                Float[torch.Tensor, "batch two_times_num_keypoints"],
                Float[torch.Tensor, "batch num_keypoints"],
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
                    assert self.data_module is not None
                    view_dataset = self.data_module.dataset.dataset  # type: ignore[index]
                    df.index = view_dataset[view_name].image_names
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
                assert self.data_module is not None
                df.index = self.data_module.dataset.image_names  # type: ignore[union-attr]
            retval = df

        return retval


def predict_dataset(
    model: Model,
    data_module: BaseDataModule,
    preds_file: str | list[str],
    cfg: DictConfig | ListConfig | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Save predicted keypoints for a labeled dataset.

    Args:
        model: API model wrapper; its underlying lightning module is used for inference.
        data_module: data module that contains dataloaders for train, val, test splits.
        preds_file: path for the predictions .csv file.
        cfg: hydra config; if None, falls back to ``model.config.cfg``.

    Returns:
        pandas dataframe with predictions or dict with dataframe of predictions for each view

    """
    cfg_eff = cfg if cfg is not None else model.config.cfg

    trainer = pl.Trainer(devices=1, accelerator='gpu', logger=False)

    labeled_preds = trainer.predict(
        model=model.model,
        dataloaders=data_module.full_labeled_dataloader(),
        return_predictions=True,
    )
    assert labeled_preds is not None

    pred_handler = PredictionHandler(cfg=cfg_eff, data_module=data_module, video_file=None)
    labeled_preds_typed = cast(
        list[tuple[torch.Tensor, torch.Tensor]], labeled_preds
    )
    labeled_preds_df = pred_handler(preds=labeled_preds_typed)
    if isinstance(labeled_preds_df, dict):
        if isinstance(preds_file, str):
            # old logic used to save to <predictions>_<view_name>.csv
            for view_name, df in labeled_preds_df.items():
                df.to_csv(preds_file.replace(".csv", f"_{view_name}.csv"))
        elif isinstance(preds_file, list):
            # preds_file is a list of views corresponding to cfg.data.view_names.
            # this allows the caller to specify the output locations more flexibly.

            # Check the order of labeled_preds_df keys matches the order of the views in the cfg.
            assert list(labeled_preds_df.keys()) == list(cfg_eff.data.view_names)

            for (_view_name, df), _pred_file in zip(
                labeled_preds_df.items(), preds_file, strict=True
            ):
                df.to_csv(_pred_file)

    else:
        assert isinstance(preds_file, str), 'preds_file must be a str for single-view predictions'
        labeled_preds_df.to_csv(preds_file)

    # clear up memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return labeled_preds_df


@overload
def predict_video(
    video_file: str,
    model: Model,
    output_pred_file: str | None = None,
    progress_file: Path | None = None,
    bbox_file: str | Path | None = None,
) -> pd.DataFrame: ...


@overload
def predict_video(
    video_file: list[str],
    model: Model,
    output_pred_file: list[str] | None = None,
    progress_file: Path | None = None,
) -> list[pd.DataFrame]: ...


def predict_video(
    video_file: str | list[str],
    model: Model,
    output_pred_file: str | list[str] | None = None,
    progress_file: Path | None = None,
    bbox_file: str | Path | None = None,
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Args:
        video_file: Predict on a video, or for true multiview models, a list of videos
            (order: 1-1 correspondence with cfg.data.view_names).
        model: The model to predict with.
        output_pred_file: (optional) File to save predictions in.
            For multiview, a list of files (1-1 correspondance to cfg.data.view_names).
        bbox_file: (optional) path to a bbox CSV (columns x, y, h, w; one row per frame).
            when provided, DALI delivers full-resolution frames and the wrapper crops each
            frame to the bbox before resizing to the model's input dims. single-view only.
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
            video_file, model.config.cfg.data.view_names, strict=True
        ):
            assert (
                view_name in Path(single_video_file).stem
            ), "expected video_file to correspond 1-1 with cfg.data.view_name"

    bbox_df: pd.DataFrame | None = None
    if bbox_file is not None:
        if is_multiview:
            raise ValueError('bbox_file is not supported for multiview prediction')
        assert isinstance(video_file, str)
        bbox_df = pd.read_csv(bbox_file, index_col=0)
        frame_count = count_frames(video_file)
        if len(bbox_df) != frame_count:
            raise ValueError(
                f'bbox_file has {len(bbox_df)} rows but video has {frame_count} frames'
            )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        callbacks=(
            [JSONInferenceProgressTracker(progress_file)] if progress_file is not None else None
        ),
    )
    model_type: Literal["base", "context"] = (
        "context" if model.config.cfg.model.model_type == "heatmap_mhcrnn" else "base"
    )

    filenames = [video_file] if not is_multiview else [[f] for f in video_file]
    from lightning_pose.data.dali import PrepareDALI  # avoids ImportError on cpu-only installs
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
        bbox_df=bbox_df,
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
    assert preds is not None

    preds_typed = cast(list[tuple[torch.Tensor, torch.Tensor]], preds)
    preds_df = pred_handler(preds=preds_typed, is_multiview_video=is_multiview)

    # Convert to a 1-1 correspondence list similar to video_files, for multiview.
    if isinstance(preds_df, dict):
        preds_df = [
            preds_df[view_name] for view_name in model.config.cfg.data.view_names
        ]

    if output_pred_file is not None:
        # save the predictions to a csv; create directory if it doesn't exist

        if is_multiview:
            for df, output_file in zip(preds_df, output_pred_file, strict=True):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file)
        else:
            assert isinstance(preds_df, pd.DataFrame)
            assert isinstance(output_pred_file, str)
            preds_df.to_csv(output_pred_file)

    # clear up memory
    del model
    del trainer
    del predict_loader
    gc.collect()
    torch.cuda.empty_cache()

    return preds_df


def make_dlc_pandas_index(
    cfg: DictConfig | ListConfig,
    keypoint_names: list[str],
) -> pd.MultiIndex:
    """Create a DLC-style three-level pandas MultiIndex for prediction DataFrames.

    Args:
        cfg: hydra config used to obtain the model type for the scorer level.
        keypoint_names: list of body-part names.

    Returns:
        ``pd.MultiIndex`` with levels ``["scorer", "bodyparts", "coords"]`` where coords are
        ``["x", "y", "likelihood"]``.
    """
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [[f"{cfg.model.model_type}_tracker"], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex


def _make_cmap(number_colors: int, cmap: str) -> np.ndarray:
    """Sample ``number_colors`` evenly spaced RGB colours from a matplotlib colormap.

    Args:
        number_colors: number of discrete colours to sample.
        cmap: matplotlib colormap name (e.g., ``"cool"``).

    Returns:
        Uint8 array of shape ``(number_colors, 3)`` with RGB values in ``[0, 255]``.
    """
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


def _create_labeled_video(
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

    Args:
        clip: moviepy VideoFileClip to annotate.
        xs_arr: x coordinates of keypoints, shape (T, n_joints)
        ys_arr: y coordinates of keypoints, shape (T, n_joints)
        mask_array: shape (T, n_joints); timepoints/joints with False will not be plotted
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
    colors = _make_cmap(n_keypoints, cmap=colormap or "cool")

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
        clip = cast(VideoFileClip, clip.resized((upsample_factor * nx, upsample_factor * ny)))
        nx, ny = clip.size

    logger.info(
        f'duration of video [s]: {np.round(dur, 2)}, recorded at {np.round(fps_og, 2)} fps'
    )

    def seconds_to_hms(seconds: float) -> str:
        """Format a duration in seconds as an ``HH:MM:SS`` string."""
        # Convert seconds to a timedelta object
        td = datetime.timedelta(seconds=seconds)

        # Extract hours, minutes, and seconds from the timedelta object
        hours = td // datetime.timedelta(hours=1)
        minutes = (td // datetime.timedelta(minutes=1)) % 60
        remainder = td % datetime.timedelta(minutes=1)

        # Format the hours, minutes, and seconds into a string
        hms_str = f"{hours:02}:{minutes:02}:{remainder.seconds:02}"

        return hms_str

    # add marker to each frame t, where t is in sec
    def add_marker_and_timestamps(get_frame: Any, t: float) -> np.ndarray:
        """Overlay keypoint markers and a timestamp on the frame at time ``t``."""
        image = get_frame(t)
        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * fps_og))
        # ----------------
        # markers
        # ----------------
        if index >= n_frames:
            logger.debug(f'add_marker_and_timestamps: skipped frame {index}')
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

    clip_marked = clip.transform(add_marker_and_timestamps)
    clip_marked.write_videofile(
        output_video_path, codec="libx264", fps=fps or fps_og or 20.0
    )
    clip_marked.close()


def generate_labeled_video(
    video_file: str,
    preds_df: pd.DataFrame,
    output_mp4_file: str,
    confidence_thresh_for_vid: float,
    colormap: str,
) -> None:
    """Overlay keypoint markers on a video and write the result to disk.

    Args:
        video_file: path to the source video file.
        preds_df: predictions DataFrame with columns indexed as
            ``(scorer, bodypart, coord)`` where coord is x, y, or likelihood.
        output_mp4_file: path where the labeled video will be saved.
        confidence_thresh_for_vid: keypoints with confidence below this value are not plotted.
        colormap: matplotlib colormap name used to colour each keypoint.
    """
    os.makedirs(os.path.dirname(output_mp4_file), exist_ok=True)
    # transform df to numpy array
    keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
    xs_arr = keypoints_arr[:, :, 0]
    ys_arr = keypoints_arr[:, :, 1]
    mask_array = keypoints_arr[:, :, 2] > confidence_thresh_for_vid
    # video generation
    video_clip = VideoFileClip(video_file)
    _create_labeled_video(
        clip=video_clip,
        xs_arr=xs_arr,
        ys_arr=ys_arr,
        mask_array=mask_array,
        output_video_path=output_mp4_file,
        colormap=colormap,
    )
