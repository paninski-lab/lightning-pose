import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
import imgaug.augmenters as iaa

def test_heatmap_dataset():
        from pose_est_nets.datasets.datasets import BaseTrackingDataset, DLCHeatmapDataset
        data_transform = []
        data_transform.append(
            iaa.Resize({"height": 384, "width": 384})
        )  # dlc dimensions need to be repeatably divisable by 2
        imgaug_transform = iaa.Sequential(data_transform)

        regData = BaseTrackingDataset(root_directory="../../toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv",
                              header_rows=[1, 2], imgaug_transform = imgaug_transform)
        heatmapData = DLCHeatmapDataset(root_directory="../../toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv",
                              header_rows=[1, 2], imgaug_transform = imgaug_transform, mode = 'csv')
        assert(torch.equal(regData.__getitem__(0)[0], heatmapData.__getitem__(0)[0]))
        assert(heatmapData.__getitem__(0)[0].shape[1:] == (heatmapData.height, heatmapData.width))
        assert(heatmapData.__getitem__(0)[1].shape[1:] == heatmapData.output_shape)
        assert(type(regData.__getitem__(0)[1]) == torch.Tensor)
        numLabels = regData.labels.shape[1]
        for idx in range(numLabels):
            if torch.any(torch.isnan(regData.__getitem__(0)[1][idx])):
                print(torch.max(heatmapData.__getitem__(0)[1][idx]))
                assert(torch.max(heatmapData.__getitem__(0)[1][idx]) == torch.tensor(0))
            else:
                assert(torch.max(heatmapData.__getitem__(0)[1][idx]) != torch.tensor(0))
        
        print(heatmapData.__getitem__(11)[1])
        


test_heatmap_dataset()
