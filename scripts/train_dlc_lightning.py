import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pose_est_nets.models.new_heatmap_tracker import SemiSupervisedHeatmapTracker
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.datasets.utils import split_data_deterministic
from pose_est_nets.datasets.datamodules import UnlabeledDataModule

from typing import Any, Callable, Optional, Tuple, List
import json
import argparse
import pandas as pd
import imgaug.augmenters as iaa
import numpy as np
from pose_est_nets.utils.plotting_utils import saveNumericalPredictions, plotPredictions
from pose_est_nets.utils.IO import set_or_open_folder, get_latest_version
from pose_est_nets.utils.wrappers import predict_plot_test_epoch

# TODO: change the "mode" convention
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
    "--num_workers", type=int, default=os.cpu_count()
)  # From Nick: on local machine multiple workers don't seem to work
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
imgaug_transform = []
imgaug_transform.append(
    iaa.Resize({"height": 384, "width": 384})
)  # dlc dimensions need to be repeatably divisable by 2
imgaug_transform = iaa.Sequential(imgaug_transform)

# header rows are hardcoded
header_rows = [1, 2]

unlabeled_data_path = '../unlabeled_videos/180726_005.mp4' #Nick specific

DATAMODULE = UnlabeledDataModule

#TODO Add deterministic data calculations to data utils folder which can also incorperate processing of dataset view info 
if args.select_data_mode == "deterministic":
    print("deterministic")
    train_data, val_data, test_data = split_data_deterministic(root_directory=args.data_dir, csv_path=args.data_path, header_rows=header_rows,
        imgaug_transform=imgaug_transform,
        downsample_factor=args.downsample_factor)
    
    datamod = DATAMODULE(
        train_data,
        train_batch_size=16,
        validation_batch_size=10,
        test_batch_size=1,
        num_workers=args.num_workers,
        use_deterministic = True,
        unlabeled_video_path = unlabeled_data_path,
    )  # dlc configs
    datamod.train_set = train_data
    datamod.valid_set = val_data
    datamod.test_set = test_data
    data = train_data
else:
    print("not deterministic")
    full_data = HeatmapDataset(
        root_directory=args.data_dir,
        csv_path=args.data_path,
        header_rows=header_rows,
        noNans=True,
        imgaug_transform=imgaug_transform,
        downsample_factor=args.downsample_factor
    )
    datamod = DATAMODULE(
        full_data,
        train_batch_size=16,
        validation_batch_size=10,
        test_batch_size=1,
        num_workers=args.num_workers,
        use_deterministic = False,
        unlabeled_video_path = unlabeled_data_path,
    )  # dlc configs
    data = full_data

datamod.setup()
datamod.computePCA_params()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(data.num_targets)
model = SemiSupervisedHeatmapTracker(
    num_targets=data.num_targets,
    resnet_version=50,
    pretrained=True,
    downsample_factor=args.downsample_factor,
    pca_param_dict = datamod.pca_param_dict,
    output_shape = data.output_shape,
    output_sigma = data.output_sigma,
    upsample_factor = 100,
    confidence_scale = 255.0,
    #device = device
)
if args.load:
    model = model.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        num_targets=data.num_targets,
        resnet_version=50,
        pretrained=True,
        downsample_factor=args.downsample_factor,
        pca_param_dict = datamod.pca_param_dict,
        output_shape = data.output_shape,
        output_sigma = data.output_sigma,
        upsample_factor = 100,
        confidence_scale = 255.0,
        #device = device
    )

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
