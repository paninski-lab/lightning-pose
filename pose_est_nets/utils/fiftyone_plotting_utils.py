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

def tensor_to_keypoint_list(keypoint_tensor, height, width):  # TODO: move to utils file
    img_kpts_list = []
    for i in range(len(keypoint_tensor)):
        img_kpts_list.append(
            tuple(
                (float(keypoint_tensor[i][0] / height), float(keypoint_tensor[i][1] / width))
            )
        )
        # keypoints are normalized to the original image dims, either add these to data config, or automatically detect by
        # loading a sample image in dataset.py or something
    return img_kpts_list

def make_dataset_and_evaluate(cfg, datamod, best_models):
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
    image_names = datamod.fulldataset.image_names
    gt_keypoints = datamod.fulldataset.labels
    samples = []
    train_indices = datamod.train_set.indices
    valid_indices = datamod.val_set.indices
    test_indices = datamod.test_set.indices
    for model in best_models.values():
        model.eval()
    for idx, img_name in enumerate(image_names):
        print(idx)
        img_path = os.path.join(cfg.data.data_dir, img_name)
        # assert os.path.isfile(img_path)
        gt_img_kpts = gt_keypoints[idx]
        nan_bool = (
            torch.sum(torch.isnan(gt_img_kpts), dim=1) > 0
        )  # e.g., when dim == 0, those columns (keypoints) that have more than zero nans
        gt_img_kpts = gt_img_kpts[~nan_bool]
        gt_kpts_list = tensor_to_keypoint_list(gt_img_kpts, cfg.data.image_orig_dims.height, cfg.data.image_orig_dims.width)
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
            keypoints=[fo.Keypoint(points=gt_kpts_list)]
        )
        img = datamod.fulldataset.__getitem__(idx)[0].unsqueeze(0)
        img_BHWC = img.permute(0, 2, 3, 1)  # Needs to be BHWC format
        for name, model in best_models.items():
            pred = model.forward(img)
            if isinstance(model, HeatmapTracker) or issubclass(
                type(model), HeatmapTracker
            ):  # check if model is in the heatmap family
                pred, confidence = model.run_subpixelmaxima(pred)
            resized_pred = reverse_transform(
                images=img_BHWC.numpy(),
                keypoints=(pred.detach().numpy().reshape((1, -1, 2))),
            )[1][0]
            pred_kpts_list = tensor_to_keypoint_list(resized_pred[~nan_bool], cfg.data.image_orig_dims.height, cfg.data.image_orig_dims.width)
            sample[name + "_prediction"] = fo.Keypoints(
                keypoints=[fo.Keypoint(points=pred_kpts_list)]
            )
        samples.append(sample)

    full_dataset = fo.Dataset(cfg.eval.fifty_one_dataset_name)
    full_dataset.add_samples(samples)
    print(full_dataset)
    print("counts: {} ".format(full_dataset.count("ground_truth.keypoints.points")))
    print("metadata: {} ".format(full_dataset.compute_metadata()))
    print("first: {} ".format(full_dataset.first()))
    session = fo.launch_app(full_dataset, remote=True)
    session.wait()
    return
