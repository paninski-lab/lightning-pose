import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
#from pose_est_nets.utils.wrappers import predict_plot_test_epoch
#from pose_est_nets.utils.IO import set_or_open_folder, load_object

#assert (os.path.isdir('toy_datasets'))

@pytest.fixture
def create_dataset():
    from pose_est_nets.datasets.datasets import DLCHeatmapDataset
    data_transform = []
    data_transform.append(iaa.Resize({"height": 384, "width": 384})) #dlc dimensions need to be repeatably divisable by 2
    data_transform = iaa.Sequential(data_transform)    
    dataset = DLCHeatmapDataset(root_directory="toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv",
                              header_rows=[1, 2], transform=data_transform)
    return dataset

@pytest.fixture
def initialize_model():
    from pose_est_nets.models.heatmap_tracker import DLC
    model = DLC(num_targets = 34, resnet_version = 50, transfer = False)
    return model

@pytest.fixture
def initialize_data_module(create_dataset):
    from pose_est_nets.datasets.datasets import TrackingDataModule
    data_module = TrackingDataModule(create_dataset, train_batch_size=4,
                                     validation_batch_size=2, test_batch_size=2,
                                     num_workers=8)
    return data_module

def test_forward(initialize_model, create_dataset):
    #TODO: separate from specific dataset, push random tensors
    model = initialize_model
    dataset = create_dataset
    dataloader = torch.utils.data.DataLoader(dataset)
    images, y_heatmaps = next(iter(dataloader))
    pred_heatmaps = model(images)
    assert (pred_heatmaps.dtype == torch.float)
    assert (images.shape == (1, 3, 384, 384))
    assert (pred_heatmaps.shape == (1, 17, 96, 96))
    loss = model.heatmap_loss(y_heatmaps, pred_heatmaps)
    assert (loss.detach().numpy() > -0.00000001)
    loss = model.regression_loss(pred_heatmaps, pred_heatmaps)
    assert (loss.detach().numpy() == float(0))
    assert (loss.shape == torch.Size([]))

def test_dataset(create_dataset, initialize_data_module):
    dataset = create_dataset
    data_module = initialize_data_module
    data_module.setup()
    train_loader = data_module.train_dataloader()
    images, y_heatmaps = next(iter(train_loader))
    assert(y_heatmaps.shape == (16, 17, 96, 96))
    full_loader = data_module.full_dataloader()
    assert(len(full_loader) == len(dataset))
    for batch in full_loader:
        images, y_heatmaps = batch
        assert(~torch.any(torch.isnan(y_heatmaps)))

def test_loss(initialize_model, initialize_data_module):
    model = initialize_model
    data_module = initialize_data_module
    data_module.setup()
    dataloader = data_module.train_dataloader()
    image, y_heatmap = next(iter(dataloader))
    assert(y_heatmap.shape == (1, 17, 96, 96))
    pred_heatmap = torch.clone(y_heatmap)
    zero_heatmap = torch.zeros(size = (96, 96))
    y_heatmap[0, 1] = zero_heatmap
    y_heatmap[0, 3] = zero_heatmap
    y_heatmap[0, 5] = zero_heatmap
    y_heatmap[0, 7] = zero_heatmap
    loss = model.heatmap_loss(y_heatmap, pred_heatmap)
    assert (loss.detach().numpy() == float(0)) #all the zero_heatmaps in the gt_heatmaps should be ignored while computing the loss

    
    
