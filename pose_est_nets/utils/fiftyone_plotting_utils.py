import fiftyone as fo
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from pose_est_nets.models.heatmap_tracker import HeatmapTracker
import torch
from tqdm import tqdm
from typing import Union, Callable
import pandas as pd


# def make_keypoint_list(
#     keypoint_arr: Union[torch.tensor, pd.core.frame.dataframe],
#     img_width: int,
#     img_height: int,
# ) -> list:  # why not making anything into pandas arr?
#     """this function will have two modes. one is a pandas dframe, the other is a tensor.
#     or alternatively, convert everything to a tensor/numpy arr and index with ints"""
#     if type(keypoint_arr) is torch.tensor:
#         print("tensor")
#     elif type(keypoint_arr) is pd.core.frame.dataframe:
#         print("dataframe")
#     return []


def tensor_to_keypoint_list(keypoint_tensor, height, width):
    # TODO: standardize across video and image plotting. Dan's video util is more updated.
    img_kpts_list = []
    for i in range(len(keypoint_tensor)):
        img_kpts_list.append(
            tuple(
                (
                    float(keypoint_tensor[i][0] / width),  # height
                    float(keypoint_tensor[i][1] / height),  # width
                )
            )
        )
        # keypoints are normalized to the original image dims, either add these to data config, or automatically detect by
        # loading a sample image in dataset.py or something
    return img_kpts_list


def make_dataset_and_evaluate(cfg: DictConfig, datamod: Callable, best_models: dict):
    reverse_transform = []
    reverse_transform.append(
        iaa.Resize(
            {
                "height": cfg.data.image_orig_dims.height,
                "width": cfg.data.image_orig_dims.width,
            }
        )
    )
    reverse_transform = iaa.Sequential(reverse_transform)
    image_names = datamod.dataset.image_names
    ground_truth_keypoints = datamod.dataset.keypoints
    train_indices = datamod.train_dataset.indices
    valid_indices = datamod.val_dataset.indices
    test_indices = datamod.test_dataset.indices
    # send all the models to .eval() mode
    for model in best_models.values():
        model.eval()

    samples = []  # this is the list of predictions we append to
    for idx, img_name in enumerate(tqdm(image_names)):
        img_path = os.path.join(cfg.data.data_dir, img_name)
        # assert os.path.isfile(img_path)
        ground_truth_img_kpts = ground_truth_keypoints[idx]
        nan_bool = (
            torch.sum(torch.isnan(ground_truth_img_kpts), dim=1) > 0
        )  # e.g., when dim == 0, those columns (keypoints) that have more than zero nans
        ground_truth_img_kpts = ground_truth_img_kpts[~nan_bool]
        ground_truth_kpts_list = tensor_to_keypoint_list(
            ground_truth_img_kpts,
            cfg.data.image_orig_dims.height,
            cfg.data.image_orig_dims.width,
        )
        if idx in train_indices:
            tag = "train"
        elif idx in valid_indices:
            tag = "valid"
        elif idx in test_indices:
            tag = "test"
        else:
            tag = "error"
        sample = fo.Sample(filepath=img_path, tags=[tag])
        sample["ground_truth"] = fo.Keypoints(
            keypoints=[fo.Keypoint(points=ground_truth_kpts_list)]
        )
        img = datamod.dataset.__getitem__(idx)[0].unsqueeze(0)
        img_BHWC = img.permute(0, 2, 3, 1)  # Needs to be BHWC format
        with torch.no_grad():
            for name, model in best_models.items():
                pred = model.forward(img)
                if isinstance(model, HeatmapTracker) or issubclass(
                    type(model), HeatmapTracker
                ):  # check if model is in the heatmap family
                    pred, confidence = model.run_subpixelmaxima(pred)
                    print(pred)
                    print(confidence)
                resized_pred = reverse_transform(
                    images=img_BHWC.numpy(),
                    keypoints=(pred.detach().numpy().reshape((1, -1, 2))),
                )[1][0]
                pred_kpts_list = tensor_to_keypoint_list(
                    resized_pred[~nan_bool],
                    cfg.data.image_orig_dims.height,
                    cfg.data.image_orig_dims.width,
                )
                sample[name + "_prediction"] = fo.Keypoints(
                    keypoints=[fo.Keypoint(points=pred_kpts_list)]
                )
        samples.append(sample)

    full_dataset = fo.Dataset(cfg.eval.fifty_one_dataset_name)
    full_dataset.add_samples(samples)
    print("fiftyone_dataset:")
    print(full_dataset)
    print("fiftyone_dataset.first():")
    print(full_dataset.first())
    full_dataset.compute_metadata()
    print(full_dataset.exists("metadata", False))
    # session = fo.launch_app(full_dataset, remote=True)
    # session.wait()
    return
