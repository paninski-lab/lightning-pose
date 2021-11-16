import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
from pose_est_nets.utils.io import (
    check_if_semi_supervised,
    set_or_open_folder,
    get_latest_version,
)
from pose_est_nets.models.heatmap_tracker import (
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
)
from pose_est_nets.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
import matplotlib.pyplot as plt
import os
import csv
import json
import time
from typing import Callable, Optional, Tuple, List, Union, Literal
from typeguard import typechecked
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_videos_in_dir(video_dir: str) -> List[str]:
    # gather videos to process
    # TODO: check if you're give a path to a single video?
    assert os.path.isdir(video_dir)
    all_files = [video_dir + "/" + f for f in os.listdir(video_dir)]
    video_files = []
    for f in all_files:
        if f.endswith(".mp4"):
            video_files.append(f)
    if len(video_files) == 0:
        raise IOError("Did not find any video files (.mp4) in %s" % video_dir)
    return video_files


def get_model_class(map_type: str, semi_supervised: bool):
    """[summary]

    Args:
        map_type (str): "regression" | "heatmap"
        semi_supervised (bool): True if you want to use unlabeled videos

    Returns:
        a ptl model class to be initialized outside of this function.
    """
    if not (semi_supervised):
        if map_type == "regression":
            return RegressionTracker
        elif map_type == "heatmap":
            return HeatmapTracker
        else:
            raise NotImplementedError(
                "%s is an invalid map_type for a fully supervised model" % map_type
            )
    else:
        if map_type == "regression":
            return SemiSupervisedRegressionTracker
        elif map_type == "heatmap":
            return SemiSupervisedHeatmapTracker
        else:
            raise NotImplementedError(
                "%s is an invalid map_type for a semi-supervised model" % map_type
            )


def load_model_from_checkpoint(cfg: DictConfig, ckpt_file: str, eval: bool = False):
    """this will have: path to a specific .ckpt file which we extract using other funcs
    will also take the standard hydra config file"""
    from pose_est_nets.utils.io import check_if_semi_supervised

    semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
    # pick the right model class
    ModelClass = get_model_class(
        map_type=cfg.model.model_type,
        semi_supervised=semi_supervised,
    )
    # initialize a model instance, with weights loaded from .ckpt file
    if semi_supervised:
        model = ModelClass.load_from_checkpoint(
            ckpt_file,
            semi_super_losses_to_use=OmegaConf.to_object(cfg.model.losses_to_use),
            loss_params=OmegaConf.to_object(cfg.losses),
        )
    else:
        model = ModelClass.load_from_checkpoint(ckpt_file)
    if eval:
        model.eval()
    return model


def predict_videos(
    video_path: str,
    ckpt_file: str,
    cfg_file: Union[str, DictConfig],
    save_file: Optional[str] = None,
    sequence_length: int = 16,
    device: Literal["gpu", "cuda", "cpu"] = "gpu",
    video_pipe_kwargs={},
):
    """Loop over a list of videos and process with tracker using DALI for fast inference.

    Args:
        video_path (str): process all videos located in this directory
        ckpt_file (str): .ckpt file for model
        cfg_file (str): yaml file saved by hydra; must contain
            - cfg_file.losses
            - cfg_file.data.image_orig_dims
            - cfg_file.data.image_resize_dims
            - cfg_file.model.losses_to_use
            - cfg_file.model.model_type
        save_file (str): full filename of tracked points; currently supports hdf5 and csv; if
            NoneType, the output will be saved in the video path
        sequence_length (int)
        device (str): "gpu" | "cpu"
        video_pipe_kwargs (dict): extra keyword-value argument pairs for
            `pose_est_nets.datasets.DALI.video_pipe` function

    TODO: support different video formats

    """

    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
    import nvidia.dali.types as types

    from pose_est_nets.datasets.dali import video_pipe, LightningWrapper, count_frames
    from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
    from pose_est_nets.models.regression_tracker import (
        RegressionTracker,
        SemiSupervisedRegressionTracker,
    )
    from pose_est_nets.models.heatmap_tracker import (
        HeatmapTracker,
        SemiSupervisedHeatmapTracker,
    )
    from pose_est_nets.utils.io import (
        set_or_open_folder,
        get_latest_version,
    )

    # check input
    # TODO: this is problematic, we may have multiple files and not a single file name.
    if save_file is not None:
        if not (
            save_file.endswith(".csv")
            or save_file.endswith(".hdf5")
            or save_file.endswith(".hdf")
            or save_file.endswith(".h5")
            or save_file.endswith(".h")
        ):
            raise NotImplementedError("Currently only .csv and .h5 files are supported")

    if device == "gpu" or device == "cuda":
        device_pt = "cuda"
        device_dali = "gpu"
    elif device == "cpu":
        device_pt = "cpu"
        device_dali = "cpu"
    else:
        raise NotImplementedError("must choose 'gpu' or 'cpu' for `device` argument")

    # gather videos to process
    assert os.path.exists(video_path)
    all_files = [video_path + "/" + f for f in os.listdir(video_path)]
    video_files = []
    for f in all_files:
        if f.endswith(".mp4"):
            video_files.append(f)
    if len(video_files) == 0:
        raise IOError("Did not find any video files (.mp4) in %s" % video_path)

    if isinstance(cfg_file, str):
        # load configuration file
        with open(cfg_file, "r") as f:
            cfg = OmegaConf.load(f)
    elif isinstance(cfg_file, DictConfig):
        cfg = cfg_file
    else:
        raise ValueError("cfg_file must be str or DictConfig, not %s!" % type(cfg_file))

    model = load_model_from_checkpoint(cfg=cfg, ckpt_file=ckpt_file, eval=True)

    model.to(device_pt)

    # set some defaults
    batch_size = (
        1  # don't change this, change sequence length (exposed to user) instead
    )
    video_pipe_kwargs_defaults = {"num_threads": 2, "device_id": 0}
    for key, val in video_pipe_kwargs_defaults.items():
        if key not in video_pipe_kwargs.keys():
            video_pipe_kwargs[key] = val

    # loop over videos
    for video_file in video_files:

        print("Processing video at %s" % video_file)

        if save_file is None:
            # create filename based on video name and model type
            video_file_name = os.path.basename(video_file).replace(".mp4", "")
            semi_supervised = check_if_semi_supervised(cfg.model.losses_to_use)
            if (
                semi_supervised
            ):  # only if any of the unsupervised `cfg.model.losses_to_use` is actually used
                loss_str = ""
                if len(cfg.model.losses_to_use) > 0:
                    for loss in list(cfg.model.losses_to_use):
                        loss_str = loss_str.join(
                            "_%s_%.6f" % (loss, cfg["losses"][loss]["weight"])
                        )
            else:
                loss_str = ""
            save_file = os.path.join(
                video_path,
                "%s_%s%s.csv" % (video_file_name, cfg.model.model_type, loss_str),
            )

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
            device=device_dali,
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
        make_predictions_and_create_csv(
            cfg=cfg,
            model=model,
            dataloader=predict_loader,
            n_frames_=n_frames_,
            batch_size=sequence_length,  # note this is different from the batch_size defined above
            save_file=save_file,
            data_name=video_file,
        )

    # if iterating over multiple models, outside this function, the below will reduce
    # memory
    del model, pipe, predict_loader
    torch.cuda.empty_cache()


def predict_frames(
    cfg: DictConfig,
    model: LightningModule,
    dataloader: torch.utils.data.DataLoader,
    n_frames_: int,  # total number of frames in the dataset or video
    batch_size: int,  # regular batch_size for images or sequence_length for videos
    data_name: str = "dataset",
    save_heatmaps: bool = False,  # TODO: save heatmaps to hdf5 file.
):
    keypoints_np = np.zeros((n_frames_, model.num_keypoints * 2))
    confidence_np = np.zeros((n_frames_, model.num_keypoints))
    t_beg = time.time()
    n_frames_counter = 0  # total frames processed
    n = -1
    with torch.no_grad():
        for n, batch in enumerate(tqdm(dataloader)):
            if type(batch) == dict:
                image = batch["images"].to(_TORCH_DEVICE)  # predicting from dataset
            else:
                image = batch  # predicting from video
            outputs = model.forward(image)
            if cfg.model.model_type == "heatmap":
                pred_keypoints, confidence = model.run_subpixelmaxima(outputs)
                # send to cpu
                pred_keypoints = pred_keypoints.detach().cpu().numpy()
                confidence = confidence.detach().cpu().numpy()
            else:
                pred_keypoints = outputs.detach().cpu().numpy()
                confidence = np.zeros((outputs.shape[0], outputs.shape[1] // 2))
            n_frames_curr = pred_keypoints.shape[0]
            if n_frames_counter + n_frames_curr > n_frames_:
                # final sequence
                final_batch_size = n_frames_ - n_frames_counter
                keypoints_np[n_frames_counter:] = pred_keypoints[:final_batch_size]
                confidence_np[n_frames_counter:] = confidence[:final_batch_size]
                n_frames_curr = final_batch_size
            else:  # at every sequence except the final
                keypoints_np[
                    n_frames_counter : n_frames_counter + n_frames_curr
                ] = pred_keypoints
                confidence_np[
                    n_frames_counter : n_frames_counter + n_frames_curr
                ] = confidence

            n_frames_counter += n_frames_curr
        t_end = time.time()
        if n == -1:
            print(
                "WARNING: issue processing %s" % data_name
            )  # TODO: what can go wrong here?
            return
        else:
            print(
                "inference speed: %1.2f fr/sec" % ((n * batch_size) / (t_end - t_beg))
            )
            # for a regression network, confidence_np will be all zeros
            return keypoints_np, confidence_np


def make_predictions_arr(
    cfg: DictConfig,
    keypoints_np: np.array,
    confidence_np: np.array,
) -> np.array:
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


def make_predictions_and_create_csv(
    cfg: DictConfig,
    model: LightningModule,
    dataloader: torch.utils.data.DataLoader,
    n_frames_: int,  # total number of frames in the dataset or video
    batch_size: int,  # regular batch_size for images or sequence_length for videos
    save_file: str,
    data_name: str = "dataset",
    save_heatmaps: bool = False,
):
    keypoints_np, confidence_np = predict_frames(
        cfg,
        model,
        dataloader,
        n_frames_,
        batch_size,
        data_name,
        save_heatmaps,
    )

    predictions = make_predictions_arr(cfg, keypoints_np, confidence_np)
    # get bodypart names from labeled data csv if possible
    from pose_est_nets.utils.io import verify_real_data_paths

    if ("data_dir" in cfg.data) and ("csv_file" in cfg.data):
        data_dir, _ = verify_real_data_paths(
            cfg.data
        )  # needed for getting bodypart names for toy_dataset
        csv_file = os.path.join(data_dir, cfg.data.csv_file)
    else:
        csv_file = ""
    if os.path.exists(csv_file):
        if "header_rows" in cfg.data:
            header_rows = list(cfg.data.header_rows)
        else:
            # assume dlc format
            header_rows = [0, 1, 2]
        df = pd.read_csv(csv_file, header=header_rows)
        # collect marker names from multiindex header
        joint_labels = [c[0] for c in df.columns[1::2]]
    else:
        joint_labels = ["bp_%i" % n for n in range(model.num_keypoints)]

    # build data frame
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % cfg.model.model_type], joint_labels, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(predictions, columns=pdindex)
    if save_file.endswith(".csv"):
        df.to_csv(save_file)
    elif save_file.find(".h") > -1:
        df.to_hdf(save_file)
    else:
        raise NotImplementedError("Currently only .csv and .h5 files are supported")


def predict_dataset(
    cfg: DictConfig,
    datamod: LightningDataModule,
    hydra_output_directory: str,
    ckpt_file: str,
    save_file: str = None,
    heatmap_idxs: List[int] = None,
):
    """
    Call this function with a path to ckpt file for a trained model
    heatmap_idxs: indexes of datapoints to save heatmaps for (NOT IMPLEMENTED YET)

    """
    model = load_model_from_checkpoint(cfg=cfg, ckpt_file=ckpt_file, eval=True)
    model.to(_TORCH_DEVICE)
    full_dataset = datamod.dataset
    num_datapoints = len(full_dataset)
    dataset_split_indices = {
        "train": datamod.train_dataset.indices,
        "validation": datamod.val_dataset.indices,
        "test": datamod.test_dataset.indices,
    }
    with open(hydra_output_directory + "/dataset_split_indices.json", "w") as f:
        json.dump(dataset_split_indices, f)

    full_dataloader = DataLoader(
        dataset=full_dataset, batch_size=datamod.test_batch_size
    )
    if save_file is None:
        save_file = (
            hydra_output_directory + "/predictions.csv"
        )  # default for now, should be saved to the model directory
    make_predictions_and_create_csv(
        cfg=cfg,
        model=model,
        dataloader=full_dataloader,
        n_frames_=num_datapoints,
        batch_size=datamod.test_batch_size,
        save_file=save_file,
    )
