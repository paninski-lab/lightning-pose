import os
import torch
import pytorch_lightning as pl
import pytest
import torchvision.transforms as transforms 
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--data_root_dir", type=str, default="/Users/danbiderman/Dropbox/Columbia/1.Dan/Research/Paninski/Video_Datasets/mouseRunningData/", help="path to Rick data")
args, _ = parser.parse_known_args()
assert(os.path.isdir(args.data_root_dir))
def test_init():
	from pose_est_nets.models.regression_tracker import RegressionTracker
	from pose_est_nets.datasets.datasets import TrackingDataset
	model = RegressionTracker(num_targets=4)
	dataset = TrackingDataset(root_directory=args.data_root_dir, csv_path="CollectedData_.csv", header_rows=[1,2], transform=transforms.ToTensor())
	dataloader = torch.utils.data.DataLoader(dataset)
	assert(next(iter(dataloader)) is not None)
	images, labels = next(iter(dataloader))
	assert(images.shape[0]==1 and images.shape[1]==3)
	assert(model(images).shape == (1,4))
