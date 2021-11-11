from pose_est_nets.utils.heatmap_tracker_utils import SubPixelMaxima
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.utils.dataset_utils import draw_keypoints, generate_heatmaps
from pose_est_nets.losses.losses import MaskedMSEHeatmapLoss
import imgaug.augmenters as iaa
import torch
from torch.utils.data import DataLoader
import pytest

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_subpixmaxima():
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    SubPixMax = SubPixelMaxima(
        output_shape=(96, 96),
        output_sigma=torch.tensor(1.25, device=_TORCH_DEVICE),
        upsample_factor=torch.tensor(2, device=_TORCH_DEVICE),
        coordinate_scale=torch.tensor(4, device=_TORCH_DEVICE),  # 2 ** 2
        confidence_scale=torch.tensor(1, device=_TORCH_DEVICE),
        threshold=None,
        device=_TORCH_DEVICE,
    )
    example = dataset.__getitem__(idx=0)
    maxima, confidence = SubPixMax.run(
        example["heatmaps"].unsqueeze(0).to(_TORCH_DEVICE)
    )
    maxima = maxima.squeeze(0)
    assert maxima.shape == example["keypoints"].shape
    assert maxima.shape[0] // 2 == confidence.shape[1]

    # remove model/data from gpu; then cache can be cleared
    del example
    del maxima, confidence
    torch.cuda.empty_cache()  # remove tensors from gpu

    dl = DataLoader(dataset, batch_size=2)
    batch = next(iter(dl))
    del dataset
    del dl
    torch.cuda.empty_cache()  # remove tensors from gpu

    maxima1, confidence1 = SubPixMax.run(batch["heatmaps"].to(_TORCH_DEVICE))
    print(maxima1.shape, confidence1.shape)

    # remove model/data from gpu; then cache can be cleared
    del batch
    del maxima1, confidence1
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_generate_keypoints():
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
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
    SubPixMax = SubPixelMaxima(
        output_shape=(96, 96),
        output_sigma=torch.tensor(1.25, device=_TORCH_DEVICE),
        upsample_factor=torch.tensor(2, device=_TORCH_DEVICE),
        coordinate_scale=torch.tensor(4, device=_TORCH_DEVICE),  # 2 ** 2
        confidence_scale=torch.tensor(1, device=_TORCH_DEVICE),
        threshold=None,
        device=_TORCH_DEVICE,
    )
    og_maxima, gt_confidence = SubPixMax.run(gt_heatmap.to(_TORCH_DEVICE))
    torch_maxima, confidence_t = SubPixMax.run(torch_heatmap.to(_TORCH_DEVICE))
    assert (og_maxima == torch_maxima).all()
    assert (gt_confidence == confidence_t).all()

    # remove model/data from gpu; then cache can be cleared
    del example
    del gt_heatmap, gt_keypts
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_generate_keypoints_weird_shape():
    data_transform = []
    OG_SHAPE = (384, 256)
    DOWNSAMPLE_FACTOR = 2
    output_shape = (
        OG_SHAPE[0] // (2 ** DOWNSAMPLE_FACTOR),
        OG_SHAPE[1] // (2 ** DOWNSAMPLE_FACTOR),
    )
    data_transform.append(
        iaa.Resize({"height": OG_SHAPE[0], "width": OG_SHAPE[1]})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    batch = dataset.__getitem__(idx=0)
    assert batch["keypoints"].shape == (torch.Size([34]))
    gt_heatmap = batch["heatmaps"].unsqueeze(0)
    gt_keypts = batch["keypoints"].unsqueeze(0).reshape(1, 17, 2)
    assert gt_heatmap.shape == (1, 17, output_shape[0], output_shape[1])
    torch_heatmap = generate_heatmaps(
        gt_keypts, height=OG_SHAPE[0], width=OG_SHAPE[1], output_shape=output_shape
    )
    SubPixMax = SubPixelMaxima(
        output_shape=output_shape,
        output_sigma=torch.tensor(1.25, device=_TORCH_DEVICE),
        upsample_factor=torch.tensor(2, device=_TORCH_DEVICE),
        coordinate_scale=torch.tensor(4, device=_TORCH_DEVICE),  # 2 ** 2
        confidence_scale=torch.tensor(1, device=_TORCH_DEVICE),
        threshold=None,
        device=_TORCH_DEVICE,
    )
    og_maxima, gt_confidence = SubPixMax.run(gt_heatmap.to(_TORCH_DEVICE))
    torch_maxima, confidence_t = SubPixMax.run(torch_heatmap.to(_TORCH_DEVICE))
    assert (og_maxima == torch_maxima).all()
    assert (gt_confidence == confidence_t).all()


# tests the batched functionality of generate_heatmaps by comparing it to the ground
# truth heatmaps in the heatmap dataset computed by the previous numpy function draw_
# keypoints.
# TODO: Change gt computation of heatmaps for heatmap dataset to use generate_heatmaps.
def test_generate_keypoints_batched():
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    dl = DataLoader(dataset=dataset, batch_size=10)
    SubPixMax = SubPixelMaxima(
        output_shape=(96, 96),
        output_sigma=torch.tensor(1.25, device=_TORCH_DEVICE),
        upsample_factor=torch.tensor(2, device=_TORCH_DEVICE),
        coordinate_scale=torch.tensor(4, device=_TORCH_DEVICE),  # 2 ** 2
        confidence_scale=torch.tensor(1, device=_TORCH_DEVICE),
        threshold=None,
        device=_TORCH_DEVICE,
    )
    for batch in dl:
        gt_keypts = batch["keypoints"].reshape(-1, 17, 2)
        heatmaps = generate_heatmaps(
            gt_keypts, height=384, width=384, output_shape=(96, 96)
        )
        gt_maxima, gt_confidence = SubPixMax.run(batch["heatmaps"].to(_TORCH_DEVICE))
        torch_maxima, confidence_t = SubPixMax.run(heatmaps.to(_TORCH_DEVICE))
        assert (gt_maxima == torch_maxima).all()
        loss = MaskedMSEHeatmapLoss(batch["heatmaps"], heatmaps)
        assert loss < 1e-6
