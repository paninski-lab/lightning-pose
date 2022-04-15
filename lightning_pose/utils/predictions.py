"""Functions for predicting keypoints on labeled datasets and unlabeled videos."""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
import time
import torch
from torch.utils.data import DataLoader
from torchtyping import patch_typeguard
from tqdm import tqdm
from typeguard import typechecked
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from lightning_pose.data.dali import LightningWrapper
from lightning_pose.utils.io import check_if_semi_supervised
from lightning_pose.utils.scripts import pretty_print_str


_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

patch_typeguard()  # use before @typechecked


def get_devices(device: Literal["gpu", "cuda", "cpu"]) -> Dict[str, str]:
    """Get pytorch and dali device strings."""
    if device == "gpu" or device == "cuda":
        device_pt = "cuda"
        device_dali = "gpu"
    elif device == "cpu":
        device_pt = "cpu"
        device_dali = "cpu"
    else:
        raise NotImplementedError("must choose 'gpu' or 'cpu' for `device` argument")
    return {"device_pt": device_pt, "device_dali": device_dali}


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


def check_prediction_file_format(save_file: str) -> None:
    """Make sure prediction file is a csv or hdf5 file."""
    if not (
        save_file.endswith(".csv")
        or save_file.endswith(".hdf5")
        or save_file.endswith(".hdf")
        or save_file.endswith(".h5")
        or save_file.endswith(".h")
    ):
        raise NotImplementedError("Currently only .csv and .h5 files are supported")


@typechecked
def predict_dataset(
    cfg: DictConfig,
    data_module: LightningDataModule,
    ckpt_file: str,
    preds_file: str,
    heatmap_file: Optional[str] = None,
) -> None:
    """Save predicted keypoints and heatmaps for a labeled dataset.

    Args:
        cfg: hydra config
        data_module: data module that contains dataloaders for train, val, test splits
        ckpt_file: absolute path to the checkpoint of your trained model; requires .ckpt
            suffix
        preds_file: absolute filename for the predictions .csv file
        heatmap_file: absolute filename for the heatmaps .h5 file; if None, no heatmaps
            are saved

    """

    model = load_model_from_checkpoint(cfg=cfg, ckpt_file=ckpt_file, eval=True)
    model.to(_TORCH_DEVICE)
    full_dataset = data_module.dataset
    num_datapoints = len(full_dataset)
    # recover the indices assuming we re-use the same random seed as in training
    dataset_split_indices = {
        "train": data_module.train_dataset.indices,
        "validation": data_module.val_dataset.indices,
        "test": data_module.test_dataset.indices,
    }

    full_dataloader = DataLoader(
        dataset=full_dataset, batch_size=data_module.test_batch_size
    )

    df, heatmaps_np = _make_predictions(
        cfg=cfg,
        model=model,
        dataloader=full_dataloader,
        n_frames_=num_datapoints,
        batch_size=data_module.test_batch_size,
        return_heatmaps=bool(heatmap_file),
    )

    # add train/test/val column
    df["set"] = np.array(["unused"] * df.shape[0])
    # iterate over conditions and their indices, and add strs to dataframe
    for key, val in dataset_split_indices.items():
        df.loc[val, "set"] = np.repeat(key, len(val))

    # save predictions
    save_dframe(df, preds_file)

    # save heatmaps
    if heatmaps_np is not None:
        save_folder = os.path.dirname(heatmap_file)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        save_heatmaps(heatmaps_np, heatmap_file)


@typechecked
def predict_single_video(
    video_file: str,
    ckpt_file: str,
    cfg_file: Union[str, DictConfig],
    preds_file: str,
    heatmap_file: Optional[str] = None,
    sequence_length: int = 16,
    device: Literal["gpu", "cuda", "cpu"] = "gpu",
    video_pipe_kwargs: dict = {},
) -> Tuple[pd.DataFrame, Union[np.ndarray, None]]:
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
        heatmap_file (str): absolute filename for the heatmaps .h5 file; if None, no
            heatmaps are saved
        sequence_length (int, optional): number of frames in a sequence of video frames
            drawn by DALI. Defaults to 16. can be controlled externally by
            cfg.eval.dali_parameters.sequence_length.
        device (Literal[, optional): device for DALI to use. Defaults to "gpu".
        video_pipe_kwargs (dict, optional): any additional kwargs for DALI. Defaults to
            {}.

    Returns:
        Tuple[pd.DataFrame, Union[np.ndarray, None]]: pandas dataframe with predictions,
            and a potential numpy array with predicted heatmaps.

    """
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
    from lightning_pose.data.dali import video_pipe
    from lightning_pose.data.utils import count_frames
    from lightning_pose.utils.scripts import pretty_print_str

    device_dict = get_devices(device)

    cfg = get_cfg_file(cfg_file=cfg_file)
    pretty_print_str(string="Loading trained model from %s... " % ckpt_file)

    model = load_model_from_checkpoint(cfg=cfg, ckpt_file=ckpt_file, eval=True)
    model.to(device_dict["device_pt"])

    # set some defaults
    batch_size = 1  # don't modify, change sequence length (exposed to user) instead
    video_pipe_kwargs_defaults = {"num_threads": 2, "device_id": 0}
    for key, val in video_pipe_kwargs_defaults.items():
        if key not in video_pipe_kwargs.keys():
            video_pipe_kwargs[key] = val

    check_prediction_file_format(save_file=preds_file)
    pretty_print_str(string="Building DALI video eval pipeline...")

    # build video loader/pipeline
    pipe = video_pipe(
        resize_dims=(
            cfg.data.image_resize_dims.height,
            cfg.data.image_resize_dims.width,
        ),
        batch_size=batch_size,
        sequence_length=sequence_length,
        filenames=[video_file],
        random_shuffle=False,
        device=device_dict["device_dali"],
        name="reader",
        pad_sequences=True,
        **video_pipe_kwargs
    )

    predict_loader = LightningWrapper(
        pipe,
        output_map=["x"],
        last_batch_policy=LastBatchPolicy.FILL,
        last_batch_padded=False,
        auto_reset=False,
        reader_name="reader",
    )
    # iterate through video
    n_frames_ = count_frames(video_file)  # total frames in video
    pretty_print_str(string="Predicting video at %s..." % video_file)
    df, heatmaps_np = _make_predictions(
        cfg=cfg,
        model=model,
        dataloader=predict_loader,
        n_frames_=n_frames_,
        batch_size=sequence_length,  # note: different from the batch_size defined above
        data_name=video_file,
        return_heatmaps=bool(heatmap_file),
    )

    try:
        save_dframe(df, preds_file)
    except PermissionError:
        new_save_file = os.path.join(os.getcwd(), preds_file.split("/")[-1])
        save_dframe(df, new_save_file)
        print(
            f"Couldn't save file to the desired location due to a PermissionError. "
            f"Instead saved it in %s" % new_save_file
        )

    if heatmaps_np is not None:
        if not os.path.exists(os.path.dirname(heatmap_file)):
            os.makedirs(os.path.dirname(heatmap_file))
        save_heatmaps(heatmaps_np, heatmap_file)

    # if iterating over multiple models, outside this function, the below will reduce
    # memory
    del model, pipe, predict_loader
    torch.cuda.empty_cache()

    return df, heatmaps_np


@typechecked
def _make_predictions(
    cfg: DictConfig,
    model: LightningModule,
    dataloader: Union[torch.utils.data.DataLoader, LightningWrapper],
    n_frames_: int,
    batch_size: int,
    data_name: str = "dataset",
    return_heatmaps: bool = False,
) -> Tuple[pd.DataFrame, Union[np.ndarray, None]]:
    """Wrapper function that predicts, resizes, and puts results in a dataframe.

    Args:
        cfg (DictConfig): hydra config.
        model (LightningModule): a loaded model ready to be evaluated.
        dataloader (Union[torch.utils.data.DataLoader, LightningWrapper]): dataloader
            ready to be iterated.
        n_frames_ (int): total number of frames in the dataset or video
        batch_size (int): regular batch_size for images or sequence_length for videos
        data_name (str, optional): [description]. Defaults to "dataset".
        return_heatmaps (str, optional): [description]. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, Union[np.ndarray, None]]: keypoint dataframe and heatmaps

    """

    keypoints_np, confidence_np, heatmaps_np = _predict_frames(
        cfg, model, dataloader, n_frames_, batch_size, data_name, return_heatmaps,
    )
    # unify keypoints and confidences into one numpy array, scale (x,y) coords by
    # resizing factor
    predictions = make_pred_arr_undo_resize(cfg, keypoints_np, confidence_np)

    # get bodypart names from labeled data csv if possible
    csv_file = get_csv_file(cfg)
    keypoint_names = get_keypoint_names(cfg, csv_file)

    # make a hierarchical pandas index, dlc style, using keypoint_names
    pd_index = make_dlc_pandas_index(cfg, keypoint_names)

    # build dataframe from the hierarchal index and predictions array
    df = pd.DataFrame(predictions, columns=pd_index)

    return df, heatmaps_np


@typechecked
def _predict_frames(
    cfg: DictConfig,
    model: LightningModule,
    dataloader: Union[torch.utils.data.DataLoader, LightningWrapper],
    n_frames_: int,
    batch_size: int,
    data_name: str = "dataset",
    return_heatmaps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """Predict all frames from a data loader without undoing the resize or reshaping.

    Args:
        cfg (DictConfig): hydra config.
        model (LightningModule): a loaded model ready to be evaluated.
        dataloader (Union[torch.utils.data.DataLoader, LightningWrapper]): dataloader
            ready to be iterated.
        n_frames_ (int): total number of frames in the dataset or video
        batch_size (int): regular batch_size for images or sequence_length for videos
        data_name (str, optional): [description]. Defaults to "dataset".
        return_heatmaps (str, optional): [description]. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]: keypoints, confidences,
            and potentially heatmaps.

    """

    if cfg.model.model_type != "heatmap":
        return_heatmaps = False

    keypoints_np = np.zeros((n_frames_, model.num_keypoints * 2))
    confidence_np = np.zeros((n_frames_, model.num_keypoints))
    if return_heatmaps:
        heatmaps_np = np.zeros(
            (
                n_frames_,
                model.num_keypoints,
                model.output_shape[0],  # // (2 ** model.downsample_factor),
                model.output_shape[1],  # // (2 ** model.downsample_factor)
            )
        )
    else:
        heatmaps_np = None
    t_beg = time.time()
    n_frames_counter = 0  # total frames processed
    n_batches = int(np.ceil(n_frames_ / batch_size))
    n = -1
    with torch.inference_mode():
        for n, batch in enumerate(tqdm(dataloader, total=n_batches)):
            if type(batch) == dict:
                image = batch["images"].to(_TORCH_DEVICE)  # predicting from dataset
            else:
                image = batch  # predicting from video
            outputs = model.forward(image, mode='2d_context')
            if cfg.model.model_type == "heatmap":
                pred_keypoints, confidence = model.run_subpixelmaxima(outputs)
                # send to cpu
                pred_keypoints = pred_keypoints.detach().cpu().numpy()
                confidence = confidence.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
            else:
                pred_keypoints = outputs.detach().cpu().numpy()
                confidence = np.zeros((outputs.shape[0], outputs.shape[1] // 2))
            n_frames_curr = pred_keypoints.shape[0]
            if n_frames_counter + n_frames_curr > n_frames_:
                # final sequence
                final_batch_size = n_frames_ - n_frames_counter
                keypoints_np[n_frames_counter:] = pred_keypoints[:final_batch_size]
                confidence_np[n_frames_counter:] = confidence[:final_batch_size]
                if return_heatmaps:
                    heatmaps_np[n_frames_counter:] = outputs[:final_batch_size]
                n_frames_curr = final_batch_size
            else:  # at every sequence except the final
                keypoints_np[
                    n_frames_counter : n_frames_counter + n_frames_curr
                ] = pred_keypoints
                confidence_np[
                    n_frames_counter : n_frames_counter + n_frames_curr
                ] = confidence
                if return_heatmaps:
                    heatmaps_np[
                        n_frames_counter : n_frames_counter + n_frames_curr
                    ] = outputs

            n_frames_counter += n_frames_curr
        t_end = time.time()
        if n == -1:
            print(
                "WARNING: issue processing %s" % data_name
            )  # TODO: what can go wrong here?
            return None, None, None
        else:
            pretty_print_str(
                "inference speed: %1.2f fr/sec" % ((n * batch_size) / (t_end - t_beg))
            )
            # for regression networks, confidence_np will be all zeros,
            # heatmaps_np will be None
            return keypoints_np, confidence_np, heatmaps_np


@typechecked
def make_pred_arr_undo_resize(
    cfg: DictConfig,
    keypoints_np: np.array,
    confidence_np: np.array,
) -> np.array:
    """Resize keypoints and add confidences into one numpy array.

    Args:
        cfg: hydara config; contains resizing info
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
    x_resize = cfg.data.image_resize_dims.width
    x_og = cfg.data.image_orig_dims.width
    predictions[:, 0::3] = keypoints_np[:, 0::2] / x_resize * x_og
    # put y vals back in original pixel space
    y_resize = cfg.data.image_resize_dims.height
    y_og = cfg.data.image_orig_dims.height
    predictions[:, 1::3] = keypoints_np[:, 1::2] / y_resize * y_og
    predictions[:, 2::3] = confidence_np

    return predictions


@typechecked
def get_videos_in_dir(video_dir: str) -> List[str]:
    # gather videos to process
    # TODO: check if you're give a path to a single video?
    pretty_print_str(string="Looking inside %s..." % video_dir)
    assert os.path.isdir(video_dir)
    all_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]
    video_files = []
    for f in all_files:
        if f.endswith(".mp4"):
            video_files.append(f)
    if len(video_files) == 0:
        raise IOError("Did not find any video files (.mp4) in %s" % video_dir)
    return video_files


@typechecked
def get_csv_file(cfg: DictConfig) -> str:
    from lightning_pose.utils.io import return_absolute_data_paths

    if ("data_dir" in cfg.data) and ("csv_file" in cfg.data):
        # needed for getting bodypart names for toy_dataset
        data_dir, _ = return_absolute_data_paths(cfg.data)
        csv_file = os.path.join(data_dir, cfg.data.csv_file)
    else:
        csv_file = ""
    return csv_file


@typechecked
def get_keypoint_names(cfg: DictConfig, csv_file: Optional[str] = None) -> List[str]:
    if os.path.exists(csv_file):
        if "header_rows" in cfg.data:
            header_rows = list(cfg.data.header_rows)
        else:
            # assume dlc format
            header_rows = [0, 1, 2]
        df = pd.read_csv(csv_file, header=header_rows)
        # collect marker names from multiindex header
        keypoint_names = [c[0] for c in df.columns[1::2]]
    else:
        keypoint_names = ["bp_%i" % n for n in range(cfg.data.num_targets // 2)]
    return keypoint_names


@typechecked
def make_dlc_pandas_index(cfg: DictConfig, keypoint_names: List[str]) -> pd.MultiIndex:
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % cfg.model.model_type], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex


@typechecked
def get_model_class(map_type: str, semi_supervised: bool) -> Type[LightningModule]:
    """[summary]

    Args:
        map_type (str): "regression" | "heatmap"
        semi_supervised (bool): True if you want to use unlabeled videos

    Returns:
        a ptl model class to be initialized outside of this function.

    """
    if not semi_supervised:
        if map_type == "regression":
            from lightning_pose.models.regression_tracker import RegressionTracker

            return RegressionTracker
        elif map_type == "heatmap":
            from lightning_pose.models.heatmap_tracker import HeatmapTracker

            return HeatmapTracker
        else:
            raise NotImplementedError(
                "%s is an invalid model_type for a fully supervised model" % map_type
            )
    else:
        if map_type == "regression":
            from lightning_pose.models.regression_tracker import (
                SemiSupervisedRegressionTracker,
            )

            return SemiSupervisedRegressionTracker
        elif map_type == "heatmap":
            from lightning_pose.models.heatmap_tracker import (
                SemiSupervisedHeatmapTracker,
            )

            return SemiSupervisedHeatmapTracker
        else:
            raise NotImplementedError(
                "%s is an invalid model_type for a semi-supervised model" % map_type
            )


@typechecked
def load_model_from_checkpoint(
    cfg: DictConfig, ckpt_file: str, eval: bool = False
) -> LightningModule:
    """this will have: path to a specific .ckpt file which we extract using other funcs
    will also take the standard hydra config file"""
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
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)
    imgaug_transform = get_imgaug_transform(cfg=cfg)
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)
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

    return model


@typechecked
def save_dframe(df: pd.DataFrame, save_file: str) -> None:
    if save_file.endswith(".csv"):
        df.to_csv(save_file)
        pretty_print_str("Saved predictions to: %s" % save_file)
    elif save_file.find(".h") > -1:
        df.to_hdf(save_file)
        pretty_print_str("Saved predictions to: %s" % save_file)
    else:
        raise NotImplementedError("Currently only .csv and .h5 files are supported")


@typechecked
def save_heatmaps(heatmaps_np: np.ndarray, save_file: str)-> None:
    import h5py
    assert save_file.endswith(".h5") or save_file.endswith(".hdf5")
    with h5py.File(save_file, "w") as hf:
        hf.create_dataset("heatmaps", data=heatmaps_np)
