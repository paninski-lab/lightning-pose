import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pose_est_nets.models.heatmap_tracker import DLC
from pose_est_nets.datasets.datasets import DLCHeatmapDataset, TrackingDataModule
from typing import Any, Callable, Optional, Tuple, List
import json
import argparse
import pandas as pd
import imgaug.augmenters as iaa
import numpy as np
from pose_est_nets.utils.plotting_utils import saveNumericalPredictions, plotPredictions
from pose_est_nets.utils.IO import set_or_open_folder, get_latest_version
from pose_est_nets.utils.wrappers import predict_plot_test_epoch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--no_train",
    action="store_true",
    help="whether you want to skip training the model",
)
parser.add_argument(
    "--load", action="store_true", help="set true to load model from checkpoint"
)
parser.add_argument(
    "--predict",
    action="store_true",
    help="whether or not to generate predictions on test data",
)
parser.add_argument(
    "--save_heatmaps", action="store_true", help="save heatmaps for test data?"
)
parser.add_argument("--max_epochs", type=int, default=500, help="when to stop training")
parser.add_argument(
    "--ckpt",
    type=str,
    default="lightning_logs2/version_1/checkpoints/epoch=271-step=12511.ckpt",
    help="path to model checkpoint if you want to load model from checkpoint",
)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--validation_batch_size", type=int, default=10)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument(
    "--num_gpus", type=int, default=1 if torch.cuda.is_available() else 0
)
parser.add_argument(
    "--num_workers", type=int, default=8 if torch.cuda.is_available() else 0
)  # on local machine multiple workers don't seem to work
# parser.add_argument("--num_keypoints", type = int, default = 108) #fish data default
parser.add_argument(
    "--data_dir", type=str, default="../../deepposekit-tests/dlc_test/mouse_data/data"
)
# fish = '../data'
# mouse = '../../deepposekit-tests/dlc_test/mouse_data/data'
parser.add_argument("--data_path", type=str, default="CollectedData_.csv")
# fish = 'tank_dataset_13.h5'
# mouse = 'CollectedData_.csv'
parser.add_argument(
    "--select_data_mode",
    type=str,
    default="random",
    help="set to deterministic if you want to train and test on specific data for mouse dataset, set to random if you want a random train/test split",
)
parser.add_argument("--downsample_factor", type=int, default=3)
parser.add_argument("--num_train_examples", type=int, default=183)
args = parser.parse_args()

torch.manual_seed(11)

# Hardcoded for fish data for now, in the future we can have feature which will automatically check if a data_transform needs to be applied and select the right transformation
data_transform = []
data_transform.append(
    iaa.Resize({"height": 384, "width": 384})
)  # dlc dimensions need to be repeatably divisable by 2
data_transform = iaa.Sequential(data_transform)

mode = args.data_path.split(".")[-1]
# header rows are hardcoded
header_rows = [1, 2]

if args.select_data_mode == "deterministic":
    print("deterministic")
    train_data = DLCHeatmapDataset(
        root_directory=args.data_dir,
        data_path=args.data_path,
        header_rows=header_rows,
        mode=mode,
        transform=data_transform,
        noNans=True,
        downsample_factor=args.downsample_factor,
    )
    train_data.image_names = train_data.image_names[: args.num_train_examples]
    train_data.labels = train_data.labels[: args.num_train_examples]
    train_data.compute_heatmaps()
    val_data = DLCHeatmapDataset(
        root_directory=args.data_dir,
        data_path=args.data_path,
        header_rows=header_rows,
        mode=mode,
        transform=data_transform,
        noNans=True,
        downsample_factor=args.downsample_factor,
    )
    val_data.image_names = val_data.image_names[183 : 183 + 22]
    val_data.labels = val_data.labels[183 : 183 + 22]
    val_data.compute_heatmaps()
    test_data = DLCHeatmapDataset(
        root_directory=args.data_dir,
        data_path=args.data_path,
        header_rows=header_rows,
        mode=mode,
        transform=data_transform,
        noNans=True,
        downsample_factor=args.downsample_factor,
    )
    test_data.image_names = test_data.image_names[205:]
    test_data.labels = test_data.labels[205:]
    test_data.compute_heatmaps()
    datamod = TrackingDataModule(
        train_data,
        mode=args.select_data_mode,
        train_batch_size=16,
        validation_batch_size=10,
        test_batch_size=1,
        num_workers=args.num_workers,
    )  # dlc configs
    datamod.train_set = train_data
    datamod.valid_set = val_data
    datamod.test_set = test_data
    data = train_data
else:
    print("not deterministic")
    full_data = DLCHeatmapDataset(
        root_directory=args.data_dir,
        data_path=args.data_path,
        header_rows=header_rows,
        mode=mode,
        noNans=True,
        transform=data_transform,
    )
    datamod = TrackingDataModule(
        full_data,
        mode=args.select_data_mode,
        train_batch_size=16,
        validation_batch_size=10,
        test_batch_size=1,
        num_workers=args.num_workers,
    )  # dlc configs
    data = full_data

datamod.setup()
datamod.computePPCA_params()

model = DLC(
    num_targets=data.num_targets,
    resnet_version=50,
    transfer=False,
    downsample_factor=args.downsample_factor,
)
if args.load:
    model = model.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        num_targets=data.num_targets,
        resnet_version=50,
        transfer=False,
        downsample_factor=args.downsample_factor,
    )

model.pca_param_dict = datamod.pca_param_dict
model.output_shape = data.output_shape
model.output_sigma = data.output_sigma
device = "cuda" if torch.cuda.is_available() else "cpu"
model.upsample_factor = torch.tensor(100, device=device)
model.confidence_scale = torch.tensor(255.0, device=device)

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=100, mode="min"
)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor="val_loss")

trainer = pl.Trainer(
    gpus=args.num_gpus,
    log_every_n_steps=15,
    callbacks=[early_stopping, lr_monitor, ckpt_callback],
    auto_scale_batch_size=False,
    reload_dataloaders_every_epoch=False,
    max_epochs=args.max_epochs,
)

if not (args.no_train):
    trainer.fit(model=model, datamodule=datamod)
else:
    datamod.setup()

if args.predict:
    # if (not(args.no_train)):
    #     print("Automatically loading best checkpoint")
    #     model = model.load_from_checkpoint(checkpoint_path = ckpt_callback.best_model_path, num_targets = data.num_targets, resnet_version = 50, transfer = False, downsample_factor = args.downsample_factor)
    print("Starting to predict test images...")
    # Nick's version
    model.eval()
    # trainer.test(model = model, datamodule = datamod)
    threshold = True  # whether or not to refrain from plotting a keypoint if the max value of the heatmap is below a certain threshold
    mode = "test"
    plotPredictions(model, datamod, args.save_heatmaps, threshold, mode)
    threshold = False
    saveNumericalPredictions(
        model, datamod, threshold
    )  # assumes no thresholding for now
    # # Dan's version below:
    # print('entering dans version')
    # folder_name = get_latest_version("lightning_logs")
    # preds_folder = set_or_open_folder(os.path.join("preds", folder_name))
    # preds_dict = predict_plot_test_epoch(model,
    #                                      datamod.test_dataloader(),
    #                                      preds_folder)
