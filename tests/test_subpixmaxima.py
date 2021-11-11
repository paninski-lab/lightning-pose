from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d
from kornia.geometry.transform import pyrup

from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.utils.dataset_utils import draw_keypoints, generate_heatmaps
from pose_est_nets.losses.losses import MaskedMSEHeatmapLoss
import imgaug.augmenters as iaa
import torch
from torch.utils.data import DataLoader
import pytest

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_generate_keypoints():
    data_transform = [
        iaa.Resize({"height": 384, "width": 384})
    ]  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    example = dataset.__getitem__(idx=0)
    assert example["keypoints"].shape == (torch.Size([34]))
    gt_heatmap = example["heatmaps"].unsqueeze(0)
    gt_keypts = example["keypoints"].unsqueeze(0).reshape(1, 17, 2)
    assert gt_heatmap.shape == (1, 17, 96, 96)
    torch_heatmap = generate_heatmaps(
        gt_keypts, height=384, width=384, output_shape=(96, 96)
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(
        heatmap_gt.to(_TORCH_DEVICE),
        temperature=torch.tensor(100).to(_TORCH_DEVICE))
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(
        heatmap_torch.to(_TORCH_DEVICE),
        temperature=torch.tensor(100).to(_TORCH_DEVICE))
    preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
    confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

    # og_maxima, gt_confidence = SubPixMax.run(heatmap_gt.to(_TORCH_DEVICE))
    # torch_maxima, confidence_t = SubPixMax.run(heatmap_torch.to(_TORCH_DEVICE))
    assert(preds_gt == preds_torch).all()
    assert(confidences_gt == confidences_torch).all()

    # remove model/data from gpu; then cache can be cleared
    del batch_dict
    del heatmap_gt, keypts_gt
    del softmaxes_gt, preds_gt, confidences_gt
    del softmaxes_torch, preds_torch, confidences_torch
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_generate_keypoints_weird_shape():
    OG_SHAPE = (384, 256)
    DOWNSAMPLE_FACTOR = 2
    output_shape = (
        OG_SHAPE[0] // (2 ** DOWNSAMPLE_FACTOR),
        OG_SHAPE[1] // (2 ** DOWNSAMPLE_FACTOR),
    )
    data_transform =[
        iaa.Resize({"height": OG_SHAPE[0], "width": OG_SHAPE[1]})
    ]  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    batch_dict = dataset.__getitem__(idx=0)
    assert(batch_dict["keypoints"].shape == (torch.Size([34])))
    heatmap_gt = batch_dict["heatmaps"].unsqueeze(0)
    keypts_gt = batch_dict["keypoints"].unsqueeze(0).reshape(1, 17, 2)
    assert(heatmap_gt.shape == (1, 17, output_shape[0], output_shape[1]))
    heatmap_torch = generate_heatmaps(
        keypts_gt,
        height=OG_SHAPE[0],
        width=OG_SHAPE[1],
        output_shape=output_shape
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(
        heatmap_gt.to(_TORCH_DEVICE),
        temperature=torch.tensor(100).to(_TORCH_DEVICE))
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(
        heatmap_torch.to(_TORCH_DEVICE),
        temperature=torch.tensor(100).to(_TORCH_DEVICE))
    preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
    confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

    # og_maxima, gt_confidence = SubPixMax.run(heatmap_gt.to(_TORCH_DEVICE))
    # torch_maxima, confidence_t = SubPixMax.run(heatmap_torch.to(_TORCH_DEVICE))
    assert(preds_gt == preds_torch).all()
    assert(confidences_gt == confidences_torch).all()

    # remove model/data from gpu; then cache can be cleared
    del batch_dict
    del heatmap_gt, keypts_gt
    del softmaxes_gt, preds_gt, confidences_gt
    del softmaxes_torch, preds_torch, confidences_torch
    torch.cuda.empty_cache()  # remove tensors from gpu


# tests the batched functionality of generate_heatmaps by comparing it to the ground
# truth heatmaps in the heatmap dataset computed by the previous numpy function draw_
# keypoints.
# TODO: Change gt computation of heatmaps for heatmap dataset to use generate_heatmaps.
def test_generate_keypoints_batched():
    data_transform = [
        iaa.Resize({"height": 384, "width": 384})
    ]  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    dl = DataLoader(dataset=dataset, batch_size=10)
    for batch_dict in dl:
        keypts_gt = batch_dict["keypoints"].reshape(-1, 17, 2)
        heatmaps_torch = generate_heatmaps(
            keypts_gt,
            height=384,
            width=384,
            output_shape=(96, 96)
        )

        # find soft argmax and confidence of ground truth heatmap
        softmaxes_gt = spatial_softmax2d(
            batch_dict["heatmaps"].to(_TORCH_DEVICE),
            temperature=torch.tensor(100).to(_TORCH_DEVICE))
        preds_gt = spatial_expectation2d(
            softmaxes_gt,
            normalized_coordinates=False
        )
        confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

        # find soft argmax and confidence of generated heatmap
        softmaxes_torch = spatial_softmax2d(
            heatmaps_torch.to(_TORCH_DEVICE),
            temperature=torch.tensor(100).to(_TORCH_DEVICE))
        preds_torch = spatial_expectation2d(
            softmaxes_torch,
            normalized_coordinates=False
        )
        confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

        # og_maxima, gt_confidence = SubPixMax.run(heatmap_gt.to(_TORCH_DEVICE))
        # torch_maxima, confidence_t = SubPixMax.run(heatmap_torch.to(_TORCH_DEVICE))
        print(preds_gt[-1])
        print(preds_torch[-1])
        assert torch.allclose(preds_gt, preds_torch)

        loss = MaskedMSEHeatmapLoss(batch_dict["heatmaps"], heatmaps_torch)
        assert loss < 1e-6

    # remove model/data from gpu; then cache can be cleared
    del batch_dict
    del keypts_gt
    del softmaxes_gt, preds_gt, confidences_gt
    del softmaxes_torch, preds_torch, confidences_torch
    torch.cuda.empty_cache()  # remove tensors from gpu
