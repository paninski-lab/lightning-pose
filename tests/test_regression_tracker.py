import os 
import numpy as np 
import pandas as pd
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms 
import pytest
print(os.getcwd())
assert(os.path.isdir('toy_datasets'))
@pytest.fixture
def create_dataset():
	from pose_est_nets.datasets.datasets import TrackingDataset
	dataset = TrackingDataset(root_directory="toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv", header_rows=[1,2], transform=transforms.ToTensor())
	return dataset

def test_init(create_dataset):
	from pose_est_nets.models.regression_tracker import RegressionTracker	
	model = RegressionTracker(num_targets=34)
	dataset = create_dataset
	dataloader = torch.utils.data.DataLoader(dataset)	
	images, labels = next(iter(dataloader))	
	preds = model(images) # using the forward method without taking grads 
	assert(preds.dtype==torch.float) 
	loss = model.regression_loss(labels, preds)
	assert(loss.detach().numpy() > -0.00000001)
	assert(loss.shape == torch.Size([])) # scalar has size zero in torch
	assert(preds.shape == (1,34))
	# todo: add a test for the training loop

def test_dataset(create_dataset):	
	dataloader = torch.utils.data.DataLoader(create_dataset)
	assert(next(iter(dataloader)) is not None)
	images, labels = next(iter(dataloader))
	assert(labels.shape == (1,34))
	assert(labels.dtype == torch.float)
	assert(images.shape[0]==1 and images.shape[1]==3)
									

