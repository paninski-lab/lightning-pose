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
import fiftyone.core.metadata as fom
import h5py


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

def make_dataset_and_viz_from_csvs(cfg: DictConfig):
    assert(len(cfg.eval.model_display_names) == len(cfg.eval.hydra_paths))
    header_rows = OmegaConf.to_object(cfg.data.header_rows)
    csv_data = pd.read_csv(
        os.path.join(cfg.data.data_dir, cfg.data.csv_file), header=header_rows
    )
    image_names = list(csv_data.iloc[:, 0])
    num_kpts = cfg.data.num_targets//2
    gt_keypoints = csv_data.iloc[:, 1:].to_numpy()
    gt_keypoints = gt_keypoints.reshape(-1, num_kpts, 2)
    model_directory_paths = cfg.eval.hydra_paths
    data_pt_tags = list(pd.read_csv(model_directory_paths[0] + "predictions.csv", header=header_rows).iloc[:, -1])
    model_preds_np = np.empty(
        shape=(len(model_directory_paths), len(image_names), num_kpts, 3)
    )
    heatmap_height = cfg.data.image_resize_dims.height // (2 ** cfg.data.downsample_factor)
    heatmap_width = cfg.data.image_resize_dims.width // (2 ** cfg.data.downsample_factor)
    model_heatmaps_np = np.empty(
        shape=(len(model_directory_paths), len(image_names), num_kpts, heatmap_height, heatmap_width)
    )

    #assuming these are absolute paths for now, might change this later
    for model_idx, model_dir in enumerate(model_directory_paths):
        pred_csv_path = model_dir + "predictions.csv"
        pred_heatmap_path = model_dir + "heatmaps_and_images/heatmaps.h5"
        model_csv = pd.read_csv(
            pred_csv_path, header=header_rows
        )  # load ground-truth data csv
        keypoints_np = model_csv.iloc[:, 1:-1].to_numpy()
        keypoints_np = keypoints_np.reshape(-1, num_kpts, 3) #x, y, confidence
        model_h5 = h5py.File(pred_heatmap_path, 'r')
        heatmaps = model_h5.get("heatmaps")
        heatmaps_np = np.array(heatmaps)
        model_preds_np[model_idx] = keypoints_np
        model_heatmaps_np[model_idx] = heatmaps_np

    samples = []
    keypoint_idx = 0 #index of keypoint to visualize heatmap for
    for img_idx, img_name in enumerate(tqdm(image_names)):
        gt_kpts_list = tensor_to_keypoint_list(
            gt_keypoints[img_idx],
            cfg.data.image_orig_dims.height,
            cfg.data.image_orig_dims.width
        )
        img_path = os.path.join(cfg.data.data_dir, img_name)
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
                cfg.data.image_orig_dims.width
            )
            sample[model_name + "_prediction"] = fo.Keypoints(
                keypoints=[fo.Keypoint(points=model_kpts_list)]
            )
            model_heatmap = model_heatmaps_np[model_idx][img_idx][keypoint_idx]
            sample[model_name + "_heatmap_"] = fo.Heatmap(map=model_heatmap)
        samples.append(sample)
        keypoint_idx += 1
        if keypoint_idx == num_kpts:
            keypoint_idx = 0
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
        img_path = os.path.join(datamod.dataset.root_directory, img_name)
        assert os.path.isfile(img_path)
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
        img = datamod.dataset.__getitem__(idx)["images"].unsqueeze(0)
        img_BHWC = img.permute(0, 2, 3, 1)  # Needs to be BHWC format
        with torch.no_grad():
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
