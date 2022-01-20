import fiftyone as fo
import fiftyone.core.metadata as fom
import h5py
import hydra
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Union, Callable
from typeguard import typechecked

from lightning_pose.models.heatmap_tracker import HeatmapTracker
from lightning_pose.utils.io import return_absolute_path, return_absolute_data_paths


@typechecked
def check_lists_equal(list_1: list, list_2: list) -> bool:
    return len(list_1) == len(list_2) and sorted(list_1) == sorted(list_2)


@typechecked
def check_unique_tags(data_pt_tags: List[str]) -> bool:
    uniques = list(np.unique(data_pt_tags))
    cond_list = ["test", "train", "validation"]
    cond_list_with_unused_images = ["test", "train", "validation", "unused"]
    flag = check_lists_equal(uniques, cond_list) or check_lists_equal(
        uniques, cond_list_with_unused_images
    )
    return flag


@typechecked
def get_image_tags(pred_df: pd.DataFrame) -> pd.Series:
    # last column indicates if the image was used for training, testing, validation or unused at all
    data_pt_tags = pred_df.iloc[:, -1].replace("0.0", "unused")
    assert check_unique_tags(data_pt_tags=data_pt_tags)
    return data_pt_tags


def tensor_to_keypoint_list(keypoint_tensor, height, width):
    # TODO: standardize across video & image plotting. Dan's video util is more updated.
    img_kpts_list = []
    for i in range(len(keypoint_tensor)):
        img_kpts_list.append(
            tuple(
                (
                    float(keypoint_tensor[i][0] / width),
                    float(keypoint_tensor[i][1] / height),
                )
            )
        )
        # keypoints are normalized to the original image dims, either add these to data
        # config, or automatically detect by loading a sample image in dataset.py or
        # something
    return img_kpts_list


@typechecked
def make_keypoint_list(
    csv_with_preds: pd.DataFrame,
    keypoint_names: List[str],
    frame_idx: int,
    width: int,
    height: int,
) -> List[fo.Keypoint]:
    keypoints_list = []
    for kp_name in keypoint_names:  # loop over names
        print(kp_name)
        # "bodyparts" it appears in the csv as we read it right now, but should be ignored
        if kp_name == "bodyparts":
            continue
        # write a single keypoint's position, confidence, and name
        keypoints_list.append(
            fo.Keypoint(
                points=[
                    [
                        csv_with_preds[kp_name]["x"][frame_idx] / width,
                        csv_with_preds[kp_name]["y"][frame_idx] / height,
                    ]
                ],
                confidence=csv_with_preds[kp_name]["likelihood"][frame_idx],
                label=kp_name,  # sometimes plotted aggresively; can comment out if needed.
            )
        )
    return keypoints_list


def make_dataset_and_viz_from_csvs(cfg: DictConfig):

    # basic error checking
    assert len(cfg.eval.model_display_names) == len(cfg.eval.hydra_paths)

    df_header_rows = OmegaConf.to_object(cfg.data.header_rows)  # default is [1,2]
    data_dir, video_dir = return_absolute_data_paths(cfg.data)

    # load ground truth csv file from which we take image paths
    gt_csv_data = pd.read_csv(
        os.path.join(data_dir, cfg.data.csv_file), header=df_header_rows
    )
    image_paths = list(gt_csv_data.iloc[:, 0])
    num_kpts = cfg.data.num_keypoints

    # below doesn't seem needed, work with dataframe
    gt_keypoints = gt_csv_data.iloc[:, 1:].to_numpy()
    gt_keypoints = gt_keypoints.reshape(-1, num_kpts, 2)

    # load info from a single predictions csv file
    model_maybe_relative_paths = cfg.eval.hydra_paths
    model_abs_paths = [
        return_absolute_path(m, n_dirs_back=2) for m in model_maybe_relative_paths
    ]

    # could go to iteration zero of the loop below
    prediction_csv_file = os.path.join(model_abs_paths[0], "predictions.csv")
    pred_df = pd.read_csv(prediction_csv_file, header=df_header_rows)
    # data_pt_tags = list(pred_df.iloc[:, -1])
    # for images we ignore in training, replace a zero entry by the string "unused"
    data_pt_tags = pred_df.iloc[:, -1].replace("0.0", "unused")
    assert check_unique_tags(data_pt_tags=data_pt_tags)

    # store predictions from different models
    model_preds_np = np.empty(
        shape=(len(model_maybe_relative_paths), len(image_paths), num_kpts, 3)
    )
    heatmap_height = cfg.data.image_resize_dims.height // (
        2 ** cfg.data.downsample_factor
    )
    heatmap_width = cfg.data.image_resize_dims.width // (
        2 ** cfg.data.downsample_factor
    )
    model_heatmaps_np = np.empty(
        shape=(
            len(model_maybe_relative_paths),
            len(image_paths),
            num_kpts,
            heatmap_height,
            heatmap_width,
        )
    )

    # assuming these are absolute paths for now, might change this later
    for model_idx, model_dir in enumerate(model_abs_paths):
        pred_csv_path = os.path.join(model_dir, "predictions.csv")
        pred_heatmap_path = os.path.join(model_dir, "heatmaps_and_images/heatmaps.h5")
        model_pred_csv = pd.read_csv(
            pred_csv_path, header=df_header_rows
        )  # load ground-truth data csv
        keypoints_np = model_pred_csv.iloc[:, 1:-1].to_numpy()
        keypoints_np = keypoints_np.reshape(-1, num_kpts, 3)  # x, y, confidence
        model_h5 = h5py.File(pred_heatmap_path, "r")
        heatmaps = model_h5.get("heatmaps")
        heatmaps_np = np.array(heatmaps)
        model_preds_np[model_idx] = keypoints_np
        model_heatmaps_np[model_idx] = heatmaps_np

    samples = []
    keypoint_idx = 0  # index of keypoint to visualize heatmap for

    for img_idx, img_name in enumerate(tqdm(image_paths)):
        gt_kpts_list = tensor_to_keypoint_list(
            gt_keypoints[img_idx],
            cfg.data.image_orig_dims.height,
            cfg.data.image_orig_dims.width,
        )
        img_path = os.path.join(data_dir, img_name)
        assert os.path.isfile(img_path)
        tag = data_pt_tags[img_idx]
        if tag == 0.0:
            tag = "train-not_used"
        sample = fo.Sample(filepath=img_path, tags=[tag])
        sample["ground_truth"] = fo.Keypoints(
            keypoints=[fo.Keypoint(points=gt_kpts_list)]
        )
        for model_idx, model_name in enumerate(cfg.eval.model_display_names):
            model_kpts_list = tensor_to_keypoint_list(
                model_preds_np[model_idx][img_idx],
                cfg.data.image_orig_dims.height,
                cfg.data.image_orig_dims.width,
            )
            sample[model_name + "_prediction"] = fo.Keypoints(
                keypoints=[fo.Keypoint(points=model_kpts_list)]
            )
            # TODO: fo.Heatmap does not exist?
            # model_heatmap = model_heatmaps_np[model_idx][img_idx][keypoint_idx]
            # sample[model_name + "_heatmap_"] = fo.Heatmap(map=model_heatmap)

        samples.append(sample)
        keypoint_idx += 1
        if keypoint_idx == num_kpts:
            keypoint_idx = 0

    # create a dataset and add all samples to it
    full_dataset = fo.Dataset(cfg.eval.fifty_one_dataset_name)
    full_dataset.add_samples(samples)

    try:
        full_dataset.compute_metadata(skip_failures=False)
    except ValueError:
        print(full_dataset.exists("metadata", False))
        print(
            "the above print should indicate bad image samples, e.g., with bad paths."
        )

    session = fo.launch_app(full_dataset, remote=True)
    session.wait()

    return
