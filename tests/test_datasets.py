import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
import imgaug.augmenters as iaa


def test_heatmap_dataset():
    from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset

    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)

    regression_data = BaseTrackingDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    heatmap_data = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    # first test: both datasets provide the same image at index 0
    assert torch.equal(regression_data[0]["images"], heatmap_data[0]["images"])

    # we get the desired image height and width
    assert (
            heatmap_data[0]["images"].shape[1:] ==
            (heatmap_data.height, heatmap_data.width)
    )
    assert heatmap_data[0]["images"].shape == (3, 384, 384)  # resized image shape
    assert heatmap_data[0]["keypoints"].shape == (34,)
    assert heatmap_data[0]["heatmaps"].shape[1:] == heatmap_data.output_shape
    assert type(regression_data[0]["keypoints"]) == torch.Tensor

    # for idx in range(numLabels):
    #     if torch.any(torch.isnan(regression_data.__getitem__(0)[1][idx])):
    #         print("there is any nan here")
    #         print(torch.max(heatmap_data.__getitem__(0)[1][idx]))
    #         assert torch.max(heatmap_data.__getitem__(0)[1][idx]) == torch.tensor(0)
    #     else:  # TODO: the below isn't passing on idx 5, we have an all zeros heatmap for a label vec without nans
    #         print(f"idx {idx}")
    #         print("no nan's here")
    #         print("labels: {}")
    #         print(torch.unique(heatmap_data.__getitem__(0)[1][idx]))
    #         print("item {}".format(torch.max(heatmap_data.__getitem__(0)[1][idx])))
    #         assert torch.max(heatmap_data.__getitem__(0)[1][idx]) != torch.tensor(0)

    # print(heatmap_data.__getitem__(11)[1])

    # remove model/data from gpu; then cache can be cleared
    del regression_data
    del heatmap_data
    torch.cuda.empty_cache()  # remove tensors from gpu
