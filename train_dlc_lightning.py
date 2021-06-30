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
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import imgaug.augmenters as iaa
from PIL import Image, ImageDraw
from deepposekit.utils.image import largest_factor
from deepposekit.models.backend.backend import find_subpixel_maxima
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

parser = argparse.ArgumentParser()

parser.add_argument("--no_train", help= "whether you want to skip training the model")
parser.add_argument("--load", help = "set true to load model from checkpoint")
parser.add_argument("--predict", help = "whether or not to generate predictions on test data")
parser.add_argument("--ckpt", type = str, default = "lightning_logs2/version_1/checkpoints/epoch=271-step=12511.ckpt", help = "path to model checkpoint if you want to load model from checkpoint")
parser.add_argument("--train_batch_size", type = int, default = 16)
parser.add_argument("--validation_batch_size", type = int, default = 10)
parser.add_argument("--test_batch_size", type = int, default = 1)
parser.add_argument("--num_gpus", type = int, default = 1)
parser.add_argument("--num_workers", type = int, default = 8)

args = parser.parse_args()

model = DLC(num_targets = 34, resnet_version = 50, transfer = False)

if (args.load):
    model = model.load_from_checkpoint(checkpoint_path = args.ckpt, num_targets = 34, resnet_version = 50, transfer = False)

data_transform = []
data_transform.append(iaa.Resize({"height": 384, "width": 384})) #dlc dimensions need to be repeatably divisable by 2
data_transform = iaa.Sequential(data_transform)

full_data = DLCHeatmapDataset(root_directory= '../deepposekit-tests/dlc_test/mouse_data/data', csv_path='CollectedData_.csv', header_rows=[1, 2], transform=data_transform)
datamod = TrackingDataModule(full_data, train_batch_size = 16, validation_batch_size = 10, test_batch_size = 1, num_workers = 8) #dlc configs


early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=50, mode="min"
)

trainer = pl.Trainer(gpus=args.num_gpus, log_every_n_steps = 15, callbacks=[early_stopping], auto_scale_batch_size = False, reload_dataloaders_every_epoch=True)
if (not(args.no_train)):
    trainer.fit(model = model, datamodule = datamod)
else:
	datamod.setup()

if args.predict:
    preds = {}
    f = open('predictions.txt', 'w')
    model.eval()
    trainer.test(model = model, datamodule = datamod)
    model.eval()
    predict_dl = datamod.test_dataloader()
    i = 0
    for idx, batch in enumerate(predict_dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        plt.imshow(heatmap_pred[0, 4])
        plt.savefig('pred_heatmaps/pred_map' + str(i) + '.png')
        plt.clf()
        heatmap_pred = nn.Upsample(scale_factor = 4)(heatmap_pred)
        heatmap_pred = heatmap_pred[0]
        plt.imshow(y[0, 4])
        plt.savefig('gt_heatmaps/gt_map' + str(i) + '.png')
        plt.clf()
        y = nn.Upsample(scale_factor = 4)(y)
        y = y[0]
        pred_keypoints = torch.empty(size = (y.shape[0], 2))
        y_keypoints = torch.empty(size = (y.shape[0], 2))
        for bp_idx in range(y.shape[0]):
            pred_keypoints[bp_idx] = torch.tensor(np.unravel_index(heatmap_pred[bp_idx].argmax(), heatmap_pred[bp_idx].shape))
            y_keypoints[bp_idx] = torch.tensor(np.unravel_index(y[bp_idx].argmax(), y[bp_idx].shape))
        
        #output_shape = full_data.output_shape
        #kernel_size = np.min(output_shape)
        #kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
        #y_hat = find_subpixel_maxima(heatmap_pred.detach(), kernel_size, full_data.output_sigma, upsample_factor=100, coordinate_scale=4.0, confidence_scale = 255.0, data_format = "channels_first")
        #y_gt = find_subpixel_maxima(y.detach(), kernel_size, full_data.output_sigma, upsample_factor=100, coordinate_scale=4.0, confidence_scale = 255.0, data_format = "channels_first") 
        #y_hat = y_hat[1:]
        #y_gt = y_gt[1:]

        plt.imshow(x[0][0])
        plt.scatter(pred_keypoints[:,1], pred_keypoints[:,0], c = 'blue')
        plt.scatter(y_keypoints[:,1], y_keypoints[:,0], c = 'orange')
        plt.savefig('preds_dlc_ptl2/pred' + str(i) + '.png')
        plt.clf()
        i += 1
        
