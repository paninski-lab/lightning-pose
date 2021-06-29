import os
import torch
import torchvision.transforms as transforms
import pytest
import pytorch_lightning as pl
import shutil
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
from pose_est_nets.utils.IO import set_or_open_folder, load_object

#assert (os.path.isdir('toy_datasets'))

@pytest.fixture
def create_dataset():
    from pose_est_nets.datasets.datasets import TrackingDataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1636, 0.1636, 0.1636], std=[0.1240, 0.1240, 0.1240])
    ])
    dataset = TrackingDataset(root_directory="toy_datasets/toymouseRunningData", csv_path="CollectedData_.csv",
                              header_rows=[1, 2], transform=data_transform)
    return dataset

@pytest.fixture
def initialize_model():
    from pose_est_nets.models.regression_tracker import RegressionTracker
    model = RegressionTracker(num_targets=34, resnet_version=18)
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
    images, labels = next(iter(dataloader))
    preds = model(images)  # using the forward method without taking grads
    assert (preds.dtype == torch.float)
    assert (images.shape == (1,3, 406, 396))
    loss = model.regression_loss(labels, preds)
    assert (loss.detach().numpy() > -0.00000001)
    assert (loss.shape == torch.Size([]))  # scalar has size zero in torch
    assert (preds.shape == (1, 34))
    data = torch.ones(size=(1, 3, 2000, 2000)) # huge image
    assert(model.feature_extractor(data).shape==torch.Size([1, 512, 1, 1]))

def test_preds(initialize_model, create_dataset):
    model = initialize_model
    dataset = create_dataset
    dataloader = torch.utils.data.DataLoader(dataset)
    preds_folder = set_or_open_folder('preds_test')
    preds_dict = predict_plot_test_epoch(model, dataloader, preds_folder)
    assert(preds_dict.keys() is not None)
    assert(len(dataset)+1 == len(os.listdir(preds_folder))) # added 1 for the pkl file
    preds_dict_loaded = load_object(os.path.join(preds_folder, 'preds'))
    assert(preds_dict_loaded.keys() == preds_dict.keys())

def test_reprs_dropout(initialize_model, create_dataset):
    model = initialize_model
    x = torch.randn(size=(2, 3, 406, 396))
    representation = model.feature_extractor(x)
    assert(representation.shape == (2,512,1,1))
    reshaped_representation = model.reshape_representation(representation)
    assert(reshaped_representation.shape == (2, 512))
    drop = model.representation_dropout(reshaped_representation)
    assert(torch.sum(drop==0.)>1)


def test_archi(initialize_model):
    model = initialize_model
    assert (model.feature_extractor[-1].output_size == (1, 1))
    assert (list(model.backbone.children())[-2] == list(model.feature_extractor.children())[-1])
    assert (model.final_layer.in_features == 512)
    assert (model.final_layer.out_features == 34)
    assert (list(model.feature_extractor[-2][-1].children())[-2].weight.requires_grad == True)
# todo: add a test for the training loop

def test_dataset(create_dataset, initialize_data_module):
    from pose_est_nets.datasets.datasets import TrackingDataModule
    data_module = TrackingDataModule(create_dataset, train_batch_size=4,
                                 validation_batch_size=2, test_batch_size=2,
                                 num_workers=8)
    data_module.setup() # setup() needs to be called here if we're not fitting a module

    train_dataloader = data_module.train_dataloader()
    #dataloader = torch.utils.data.DataLoader(create_dataset)
    assert (next(iter(train_dataloader)) is not None)
    images, labels = next(iter(train_dataloader))
    assert (labels.shape == (4, 34))
    assert (labels.dtype == torch.float)
    assert (images.shape[0] == 4 and images.shape[1] == 3)

    val_dataloader = data_module.val_dataloader()
    assert (next(iter(val_dataloader)) is not None)
    images, labels = next(iter(val_dataloader))
    assert (labels.shape == (2, 34))
    assert (labels.dtype == torch.float)
    assert (images.shape[0] == 2 and images.shape[1] == 3)

def test_training(initialize_model, initialize_data_module, create_dataset):
    from pose_est_nets.datasets.datasets import TrackingDataModule
    from pose_est_nets.callbacks.freeze_unfreeze_callback import FeatureExtractorFreezeUnfreeze
    from pytorch_lightning.callbacks import Callback
    # TODO: keep checking the freeze unfreeze callback by checking that the gradients are frozen and unfrozen during training as expected

    class FreezingUnfreezingTester(Callback):

        def on_init_end(self, trainer):
            print('trainer is init now')

        def on_epoch_end(self, trainer, pl_module) -> None:
            if trainer.current_epoch == 0:
                assert (pl_module.final_layer.weight.requires_grad == True)
                assert (list(pl_module.feature_extractor.children())[0].weight.requires_grad == False)
            if trainer.current_epoch == 1:
                assert (pl_module.final_layer.weight.requires_grad == True)
                assert (list(pl_module.feature_extractor.children())[0].weight.requires_grad == False)
            if trainer.current_epoch == 2:  # here's one we ask to unfreeze
                assert (pl_module.final_layer.weight.requires_grad == True)
                assert (list(pl_module.feature_extractor.children())[0].weight.requires_grad == True)

        def on_train_end(self, trainer, pl_module):
            print('training ended')

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min"
    )
    transfer_unfreeze_callback = FeatureExtractorFreezeUnfreeze(2)
    transfer_unfreeze_tester = FreezingUnfreezingTester()
    gpus_to_use = 0
    if torch.cuda.is_available():
        gpus_to_use = 1
    model = initialize_model
    data_module = TrackingDataModule(create_dataset, train_batch_size=4,
                                     validation_batch_size=2, test_batch_size=1,
                                     num_workers=8)
    trainer = pl.Trainer(gpus=gpus_to_use, max_epochs=3,
                         log_every_n_steps=1,
                         auto_scale_batch_size=False,
                         callbacks=[early_stopping, transfer_unfreeze_callback, transfer_unfreeze_tester])  # auto_scale_batch_size not working
    trainer.fit(model=model, datamodule=data_module)
    assert(os.path.exists('lightning_logs/version_0/hparams.yaml'))
    assert(os.path.exists('lightning_logs/version_0/checkpoints'))
    shutil.rmtree('lightning_logs') # should be at teardown, we may not reach to this line if assert fails.

def test_loss():
    from torch.nn import functional as F
    import numpy as np
    labels = torch.tensor([1.0, np.nan, 3.0], dtype=torch.float)
    preds = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float)
    mask = labels == labels # labels is not none
    assert (((labels == labels) == (torch.isnan(labels)==False)).all())
    loss = F.mse_loss(torch.masked_select(labels, mask),
                      torch.masked_select(preds, mask))
    assert(loss.detach().numpy() == 1.0 **2 / 2.)