import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
import imgaug.augmenters as iaa

def test_heatmap_dataset():
	from pose_est_nets.datasets.datasets import DLCHeatmapDataset
	data_transform = []
	data_transform.append(
	    iaa.Resize({"height": 384, "width": 384})
	)  # dlc dimensions need to be repeatably divisable by 2
	imgaug_transform = iaa.Sequential(data_transform)

	regData = BaseTrackingDataset(root_directory="toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv",
                              header_rows=[1, 2], imgaug_transform = imgaug_transform)
	heatmapData = DLCHeatmapDataset(root_directory="toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv",
                              header_rows=[1, 2], imgaug_transform = imgaug_transform, mode = 'csv')


test_heatmap_dataset()
