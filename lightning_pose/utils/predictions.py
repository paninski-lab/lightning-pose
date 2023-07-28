"""Functions for predicting keypoints on labeled datasets and unlabeled videos."""

import os
import time
from typing import List, Optional, Tuple, Union

import cv2
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torchtyping import TensorType
from tqdm import tqdm
from typeguard import typechecked

from lightning_pose.data.dali import LitDaliWrapper, PrepareDALI
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.utils import count_frames
from lightning_pose.models.heatmap_tracker import HeatmapTracker, SemiSupervisedHeatmapTracker
from lightning_pose.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
from lightning_pose.utils import pretty_print_str

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@typechecked
def get_cfg_file(cfg_file: Union[str, DictConfig]):
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
    def __init__(
        self,
        cfg: DictConfig,
        data_module: Optional[pl.LightningDataModule] = None,
        video_file: Optional[str] = None,
    ) -> None:
        """

        Args
            cfg
            data_module
            video_file

        """

        # check args: data_module is optional under certain conditions
        if data_module is None:
            if video_file is None:
                raise ValueError("must pass data_module to constructor if predicting on a dataset")
            if cfg.data.get("keypoint_names", None) is None \
                    and cfg.data.get("keypoints", None) is None:
                raise ValueError(
                    "must include `keypoint_names` or `keypoints` field in cfg.data if not "
                    "passing data_module as an argument to PredictionHandler")

        self.cfg = cfg
        self.data_module = data_module
        self.video_file = video_file
        if video_file is not None:
            assert os.path.isfile(video_file)

    @property
    def frame_count(self) -> int:
        """Returns the number of frames in the video or the labeled dataset"""
        if self.video_file is not None:
            return count_frames(self.video_file)
        else:
            return len(self.data_module.dataset)

    @property
    def keypoint_names(self):
        if self.cfg.data.get("keypoint_names", None) is not None:
            return list(self.cfg.data.keypoint_names)
        elif self.cfg.data.get("keypoints", None) is not None:
            return list(self.cfg.data.keypoints)
        else:
            return self.data_module.dataset.keypoint_names

    @property
    def do_context(self):
        if self.data_module:
            return self.data_module.dataset.do_context
        else:
            return self.cfg.model.do_context

    def unpack_preds(
        self,
        preds: List[
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

    def make_pred_arr_undo_resize(
        self,
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
        assert keypoints_np.shape[0] == confidence_np.shape[0]  # num frames in the dataset
        assert keypoints_np.shape[1] == (
            confidence_np.shape[1] * 2
        )  # we have two (x,y) coordinates and a single likelihood value

        num_joints = confidence_np.shape[-1]  # model.num_keypoints
        predictions = np.zeros((keypoints_np.shape[0], num_joints * 3))
        predictions[:, 0] = np.arange(keypoints_np.shape[0])
        # put x vals back in original pixel space
        x_resize = self.cfg.data.image_resize_dims.width
        x_og = self.cfg.data.image_orig_dims.width
        predictions[:, 0::3] = keypoints_np[:, 0::2] / x_resize * x_og
        # put y vals back in original pixel space
        y_resize = self.cfg.data.image_resize_dims.height
        y_og = self.cfg.data.image_orig_dims.height
        predictions[:, 1::3] = keypoints_np[:, 1::2] / y_resize * y_og
        predictions[:, 2::3] = confidence_np

        return predictions

    def make_dlc_pandas_index(self) -> pd.MultiIndex:
        return make_dlc_pandas_index(cfg=self.cfg, keypoint_names=self.keypoint_names)

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
        preds: List[
            Tuple[
                TensorType["batch", "two_times_num_keypoints"],
                TensorType["batch", "num_keypoints"],
            ]
        ],
    ) -> pd.DataFrame:
        """
        Call this function to get a pandas dataframe of the predictions for a single video.
        Assuming you've already run trainer.predict(), and have a list of Tuple predictions.
        Args:
            preds: list of tuples of (predictions, confidences)
            video_file: path to video file
        Returns:
            pd.DataFrame: index is (frame, bodypart, x, y, likelihood)
        """
        stacked_preds, stacked_confs = self.unpack_preds(preds=preds)
        pred_arr = self.make_pred_arr_undo_resize(
            stacked_preds.cpu().numpy(), stacked_confs.cpu().numpy()
        )
        pdindex = self.make_dlc_pandas_index()
        df = pd.DataFrame(pred_arr, columns=pdindex)
        if self.video_file is None:
            # specify which image is train/test/val/unused
            df = self.add_split_indices_to_df(df)
            df.index = self.data_module.dataset.image_names

        return df


@typechecked
def predict_dataset(
    cfg: DictConfig,
    data_module: BaseDataModule,
    ckpt_file: str,
    preds_file: str,
    trainer: Optional[pl.Trainer] = None,
    model: Optional[
        Union[
            RegressionTracker,
            HeatmapTracker,
            SemiSupervisedRegressionTracker,
            SemiSupervisedHeatmapTracker,
        ]
    ] = None,
) -> pd.DataFrame:
    """Save predicted keypoints for a labeled dataset.

    Args:
        cfg: hydra config
        data_module: data module that contains dataloaders for train, val, test splits
        ckpt_file: absolute path to the checkpoint of your trained model; requires .ckpt
            suffix
        preds_file: absolute filename for the predictions .csv file
        trainer
        model

    Returns:
        pd.DataFrame: pandas dataframe with predictions

    """

    if model is None:
        model = load_model_from_checkpoint(
            cfg=cfg, ckpt_file=ckpt_file, eval=True, data_module=data_module,
        )

    if trainer is None:
        trainer = pl.Trainer(devices=1, accelerator="auto")

    labeled_preds = trainer.predict(
        model=model,
        dataloaders=data_module.full_labeled_dataloader(),
        return_predictions=True,
    )

    pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=None)
    labeled_preds_df = pred_handler(preds=labeled_preds)
    labeled_preds_df.to_csv(preds_file)

    return labeled_preds_df


@typechecked
def predict_single_video(
    video_file: str,
    ckpt_file: str,
    cfg_file: Union[str, DictConfig],
    preds_file: str,
    data_module: Union[BaseDataModule, UnlabeledDataModule],
    trainer: Optional[pl.Trainer] = None,
    model: Optional[
        Union[
            RegressionTracker,
            HeatmapTracker,
            SemiSupervisedRegressionTracker,
            SemiSupervisedHeatmapTracker,
        ]
    ] = None,
    save_heatmaps: Optional[bool] = False,
) -> pd.DataFrame:
    """Make predictions for a single video, loading frame sequences using DALI.

    This function initializes a DALI pipeline, prepares a dataloader, and passes it on
    to _make_predictions().

    Args:
        video_file (str): absolute path to a single video you want to get predictions
            for, typically .mp4 file.
        ckpt_file (str): absolute path to the checkpoint of your trained model. assumed
            .ckpt format.
        cfg_file (Union[str, DictConfig]): either a hydra config or a path pointing to
            one, with all the model specs. needed for loading the model.
        preds_file (str): absolute filename for the predictions .csv file
        data_module:
        trainer:
        model:
        save_heatmaps:

    Returns:
        pd.DataFrame: pandas dataframe with predictions

    """

    cfg = get_cfg_file(cfg_file=cfg_file)

    delete_model = False
    if model is None:
        model = load_model_from_checkpoint(
            cfg=cfg, ckpt_file=ckpt_file, eval=True, data_module=data_module
        )
        delete_model = True
    model.to("cuda")

    delete_trainer = False
    if trainer is None:
        trainer = pl.Trainer(accelerator="gpu", devices=1)
        delete_trainer = True

    # ----------------------------------------------------------------------------------
    # set up
    # ----------------------------------------------------------------------------------
    # initialize
    model_type = "context" if cfg.model.do_context else "base"
    cfg.training.imgaug = "default"
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type=model_type,
        dali_config=cfg.dali,
        filenames=[video_file],
        resize_dims=[cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width]
    )
    # get loader
    predict_loader = vid_pred_class()

    # initialize prediction handler class
    pred_handler = PredictionHandler(cfg=cfg, data_module=data_module, video_file=video_file)

    # ----------------------------------------------------------------------------------
    # compute predictions
    # ----------------------------------------------------------------------------------

    # use a different function for now to return heatmaps
    if save_heatmaps:
        if predict_loader.do_context:
            batch_size = cfg.dali.context.predict.sequence_length
        else:
            batch_size = cfg.dali.base.predict.sequence_length
        keypoints, confidences, heatmaps = _predict_frames(
            cfg=cfg,
            model=model,
            dataloader=predict_loader,
            n_frames_=pred_handler.frame_count,
            batch_size=batch_size,
            return_heatmaps=True,
        )
        preds = [(torch.tensor(keypoints), torch.tensor(confidences))]
        if heatmaps is not None:
            heatmaps_file = preds_file.replace(".csv", "_heatmaps.npy")
            os.makedirs(os.path.dirname(heatmaps_file), exist_ok=True)
            np.save(heatmaps_file, heatmaps)

    else:
        preds = trainer.predict(
            model=model,
            ckpt_path=ckpt_file,
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
    torch.cuda.empty_cache()

    return preds_df


@typechecked
def _predict_frames(
    cfg: DictConfig,
    model: LightningModule,
    dataloader: Union[torch.utils.data.DataLoader, LitDaliWrapper],
    n_frames_: int,
    batch_size: int,
    return_heatmaps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """Predict all frames in a data loader without undoing the resize/reshape; can return heatmaps.

    Args:
        cfg (DictConfig): hydra config.
        model (LightningModule): a loaded model ready to be evaluated.
        dataloader: dataloader ready to be iterated
        n_frames_ (int): total number of frames in the dataset or video
        batch_size (int): regular batch_size for images or sequence_length for videos
        return_heatmaps (str, optional): [description]. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]: keypoints, confidences,
            and potentially heatmaps.

    """

    if "heatmap" not in cfg.model.model_type:
        return_heatmaps = False

    keypoints_np = np.zeros((n_frames_, model.num_keypoints * 2))
    confidence_np = np.zeros((n_frames_, model.num_keypoints))

    if return_heatmaps:
        heatmaps_np = np.zeros((
            n_frames_,
            model.num_keypoints,
            model.output_shape[0],  # // (2 ** model.downsample_factor),
            model.output_shape[1],  # // (2 ** model.downsample_factor)
        ))
    else:
        heatmaps_np = None

    t_beg = time.time()
    n_frames_counter = 0  # total frames processed
    n_batches = int(np.ceil(n_frames_ / batch_size))
    n = -1
    with torch.inference_mode():
        for n, batch in enumerate(tqdm(dataloader, total=n_batches)):

            if cfg.model.model_type == "heatmap":
                # push batch through model
                pred_keypoints, confidence, pred_heatmaps = model.predict_step(
                    batch=batch, batch_idx=n, return_heatmaps=return_heatmaps
                )
                # send to numpy
                pred_keypoints = pred_keypoints.detach().cpu().numpy()
                confidence = confidence.detach().cpu().numpy()
                pred_heatmaps = pred_heatmaps.detach().cpu().numpy()

            elif cfg.model.model_type == "heatmap_mhcrnn":
                # push batch through model
                pred_keypoints, confidence, pred_heatmaps = model.predict_step(
                    batch=batch, batch_idx=n, return_heatmaps=return_heatmaps
                )
                # send to numpy
                pred_keypoints = pred_keypoints.detach().cpu().numpy()
                confidence = confidence.detach().cpu().numpy()
                pred_heatmaps = pred_heatmaps.detach().cpu().numpy()

            else:
                # push batch through model
                pred_keypoints, confidence = model.predict_step(
                    batch=batch, batch_idx=n
                )
                # send to numpy
                pred_keypoints = pred_keypoints.detach().cpu().numpy()
                confidence = confidence.detach().cpu().numpy()
                pred_heatmaps = None

            n_frames_curr = pred_keypoints.shape[0]
            if n_frames_counter + n_frames_curr > n_frames_:
                # final sequence
                final_batch_size = n_frames_ - n_frames_counter
                keypoints_np[n_frames_counter:] = pred_keypoints[:final_batch_size]
                confidence_np[n_frames_counter:] = confidence[:final_batch_size]
                if return_heatmaps:
                    heatmaps_np[n_frames_counter:] = pred_heatmaps[:final_batch_size]
                n_frames_curr = final_batch_size
            else:  # at every sequence except the final
                keypoints_np[n_frames_counter:n_frames_counter + n_frames_curr] = pred_keypoints
                confidence_np[n_frames_counter:n_frames_counter + n_frames_curr] = confidence
                if return_heatmaps:
                    heatmaps_np[n_frames_counter:n_frames_counter + n_frames_curr] = pred_heatmaps

            n_frames_counter += n_frames_curr

        t_end = time.time()
        pretty_print_str("inference speed: %1.2f fr/sec" % ((n * batch_size) / (t_end - t_beg)))
        # for regression networks, confidence_np will be all zeros, heatmaps_np will be None
        return keypoints_np, confidence_np, heatmaps_np


@typechecked
def make_dlc_pandas_index(cfg: DictConfig, keypoint_names: List[str]) -> pd.MultiIndex:
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % cfg.model.model_type], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex


# @typechecked
def get_model_class(map_type: str, semi_supervised: bool) -> LightningModule:
    """[summary]

    Args:
        map_type (str): "regression" | "heatmap"
        semi_supervised (bool): True if you want to use unlabeled videos

    Returns:
        a ptl model class to be initialized outside of this function.

    """
    if not semi_supervised:
        if map_type == "regression":
            from lightning_pose.models.regression_tracker import RegressionTracker as Model
        elif map_type == "heatmap":
            from lightning_pose.models.heatmap_tracker import HeatmapTracker as Model
        elif map_type == "heatmap_mhcrnn":
            from lightning_pose.models.heatmap_tracker_mhcrnn import HeatmapTrackerMHCRNN as Model
        else:
            raise NotImplementedError(
                "%s is an invalid model_type for a fully supervised model" % map_type
            )
    else:
        if map_type == "regression":
            from lightning_pose.models.regression_tracker import (
                SemiSupervisedRegressionTracker as Model,
            )
        elif map_type == "heatmap":
            from lightning_pose.models.heatmap_tracker import SemiSupervisedHeatmapTracker as Model
        elif map_type == "heatmap_mhcrnn":
            from lightning_pose.models.heatmap_tracker_mhcrnn import (
                SemiSupervisedHeatmapTrackerMHCRNN as Model,
            )
        else:
            raise NotImplementedError(
                "%s is an invalid model_type for a semi-supervised model" % map_type
            )

    return Model


# @typechecked
def load_model_from_checkpoint(
    cfg: DictConfig,
    ckpt_file: str,
    eval: bool = False,
    data_module: Optional[Union[BaseDataModule, UnlabeledDataModule]] = None,
    skip_data_module: bool = False,
) -> LightningModule:
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
    from lightning_pose.utils.io import check_if_semi_supervised, return_absolute_data_paths
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
    # initialize a model instance, with weights loaded from .ckpt file
    if semi_supervised:
        model = ModelClass.load_from_checkpoint(
            ckpt_file,
            loss_factory=loss_factories["supervised"],
            loss_factory_unsupervised=loss_factories["unsupervised"],
            strict=False,
        )
    else:
        model = ModelClass.load_from_checkpoint(
            ckpt_file,
            loss_factory=loss_factories["supervised"],
            strict=False,
        )

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
def make_cmap(number_colors: int, cmap: str = "cool"):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


def create_labeled_video(
    clip,
    xs_arr,
    ys_arr,
    mask_array=None,
    dotsize=4,
    colormap="cool",
    fps=None,
    filename="movie.mp4",
):
    """Helper function for creating annotated videos.

    Args
        clip (moviepy.editor.VideoFileClip)
        xs_arr (np.ndarray): shape T x n_joints
        ys_arr (np.ndarray): shape T x n_joints
        mask_array (np.ndarray, boolean): shape T x n_joints, same as df_x and df_y;
            any timepoints/joints with a False entry will not be plotted
        dotsize (int): size of marker dot on labeled video
        colormap (str): matplotlib color map for markers
        fps (float, optional): None to default to fps of original video
        filename (str, optional): video file name

    """

    if mask_array is None:
        mask_array = ~np.isnan(xs_arr)

    n_frames, n_keypoints = xs_arr.shape

    # Set colormap for each color
    colors = make_cmap(n_keypoints, cmap=colormap)

    nx, ny = clip.size
    dur = int(clip.duration - clip.start)
    fps_og = clip.fps

    print(f"Duration of video [s]: {np.round(dur, 2)}, recorded at {np.round(fps_og, 2)} fps!")

    # add marker to each frame t, where t is in sec
    def add_marker(get_frame, t):
        image = get_frame(t * 1.0)
        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * 1.0 * fps_og))
        for bpindex in range(n_keypoints):
            if index >= n_frames:
                print("Skipped frame {}, marker {}".format(index, bpindex))
                continue
            if mask_array[index, bpindex]:
                xc = min(int(xs_arr[index, bpindex]), nx - 1)
                yc = min(int(ys_arr[index, bpindex]), ny - 1)
                frame = cv2.circle(
                    frame,
                    center=(xc, yc),
                    radius=dotsize,
                    color=colors[bpindex].tolist(),
                    thickness=-1
                )
        return frame

    clip_marked = clip.fl(add_marker)
    clip_marked.write_videofile(
        filename, codec="libx264", fps=fps_og if fps is None else fps
    )
    clip_marked.close()

    return
