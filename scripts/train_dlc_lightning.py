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
#from PIL import Image, ImageDraw
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

def upsampleArgmax(heatmap_pred, heatmap_y):
    heatmap_pred = nn.Upsample(scale_factor = 4)(heatmap_pred)
    heatmap_pred = heatmap_pred[0]
    y = nn.Upsample(scale_factor = 4)(heatmap_y)
    y = y[0]
    pred_keypoints = torch.empty(size = (y.shape[0], 2))
    y_keypoints = torch.empty(size = (y.shape[0], 2))
    for bp_idx in range(y.shape[0]):
        pred_keypoints[bp_idx] = torch.tensor(np.unravel_index(heatmap_pred[bp_idx].argmax(), heatmap_pred[bp_idx].shape))
        y_keypoints[bp_idx] = torch.tensor(np.unravel_index(y[bp_idx].argmax(), y[bp_idx].shape))    
    return pred_keypoints, y_keypoints
 
def computeSubPixMax(heatmaps_pred, heatmaps_y, output_shape, threshold):
    kernel_size = np.min(output_shape)
    kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
    pred_keypoints = find_subpixel_maxima(heatmaps_pred.detach(), kernel_size, 1.25, 100, 4, 255.0, "channels_first")
    y_keypoints = find_subpixel_maxima(heatmaps_y.detach(), kernel_size, 1.25, 100, 4, 255.0, "channels_first")
    if threshold:
        pred_kpts_list = []
        y_kpts_list = []
        for i in range(pred_keypoints.shape[1]):
            if pred_keypoints[0, i, 2] > 0.001: #threshold for low confidence predictions
                pred_kpts_list.append(pred_keypoints[0, i, :2].numpy())
            if y_keypoints[0, i, 2] > 0.001:
                y_kpts_list.append(y_keypoints[0, i, :2].numpy())
        return torch.tensor(pred_kpts_list), torch.tensor(y_kpts_list)
    pred_keypoints = pred_keypoints[0,:,:2] #getting rid of the actual max value
    y_keypoints = y_keypoints[0,:,:2]
    return pred_keypoints, y_keypoints

def saveNumericalPredictions(threshold):
    i = 0
    rev_augmenter = []
    rev_augmenter.append(iaa.Resize({"height": 406, "width": 396}))
    rev_augmenter = iaa.Sequential(rev_augmenter)

    model.eval()
    full_dl = datamod.full_dataloader()
    test_dl = datamod.test_dataloader()
    print(len(test_dl))
    #fully_labeled_idxs = full_data.get_fully_labeled_idxs()
    final_gt_keypoints = np.empty(shape = (22, 17, 2))
    final_imgs = np.empty(shape = (22, 406, 396, 1))
    final_preds = np.empty(shape = (22, 17, 2))
    for idx, batch in enumerate(test_dl):
        #if (idx not in fully_labeled_idxs):
        #    continue
        x, y = batch
        heatmap_pred = model.forward(x)
        #pred_keypoints, y_keypoints = upsampleArgmax(heatmap_pred, y)
        output_shape = train_data.output_shape #changed from full_data
        pred_keypoints, y_keypoints = computeSubPixMax(heatmap_pred, y, output_shape, threshold)

        x = x[:,0,:,:] #only taking one image dimension
        x = np.expand_dims(x, axis = 3)
        final_imgs[i], final_gt_keypoints[i] = rev_augmenter(images = x, keypoints = np.expand_dims(y_keypoints, axis = 0))
        final_imgs[i], final_preds[i] = rev_augmenter(images = x, keypoints = np.expand_dims(pred_keypoints, axis = 0))
        i += 1

    final_gt_keypoints = np.reshape(final_gt_keypoints, newshape = (22, 34))
    final_preds = np.reshape(final_preds, newshape = (22, 34))

    np.savetxt('../preds/ptl_test_labels7.csv', final_gt_keypoints, delimiter = ',', newline = '\n')
    np.savetxt('../preds/ptl_test_predictions7.csv', final_preds, delimiter = ',', newline = '\n')

    return

def plotPredictions(save_heatmaps, threshold):
    model.eval()
    predict_dl = datamod.test_dataloader()    
    i = 0
    for idx, batch in enumerate(predict_dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        if (save_heatmaps):
            plt.imshow(heatmap_pred[0, 4].detach())
            plt.savefig('../preds/test_pred_heatmaps7/pred_map' + str(i) + '.png')
            plt.clf()
            plt.imshow(y[0, 4].detach())
            plt.savefig('../preds/test_gt_heatmaps7/gt_map' + str(i) + '.png')
            plt.clf()
        #pred_keypoints, y_keypoints = upsampleArgmax(heatmap_pred, y)
        output_shape = train_data.output_shape #changed from full_data
        pred_keypoints, y_keypoints = computeSubPixMax(heatmap_pred, y, output_shape, threshold)
        plt.imshow(x[0][0])
        plt.scatter(pred_keypoints[:,0], pred_keypoints[:,1], c = 'blue')
        plt.scatter(y_keypoints[:,0], y_keypoints[:,1], c = 'orange')
        #plt.savefig('../preds/test_preds_ptl7/pred' + str(i) + '.png')
        plt.clf()
        i += 1


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

torch.manual_seed(11)

model = DLC(num_targets = 34, resnet_version = 50, transfer = False)

if (args.load):
    model = model.load_from_checkpoint(checkpoint_path = args.ckpt, num_targets = 34, resnet_version = 50, transfer = False)

data_transform = []
data_transform.append(iaa.Resize({"height": 384, "width": 384})) #dlc dimensions need to be repeatably divisable by 2
data_transform = iaa.Sequential(data_transform)


train_data = DLCHeatmapDataset(root_directory= '../../deepposekit-tests/dlc_test/mouse_data/data', csv_path='CollectedData_.csv', header_rows=[1, 2], transform=data_transform)
print(train_data.image_names)
train_data.image_names = train_data.image_names[:183]
train_data.labels = train_data.labels[:183]
train_data.compute_heatmaps()
val_data = DLCHeatmapDataset(root_directory= '../../deepposekit-tests/dlc_test/mouse_data/data', csv_path='CollectedData_.csv', header_rows=[1, 2], transform=data_transform)
val_data.image_names = val_data.image_names[183:183+22]
val_data.labels = val_data.labels[183:183+22]
val_data.compute_heatmaps()
test_data = DLCHeatmapDataset(root_directory= '../../deepposekit-tests/dlc_test/mouse_data/data', csv_path='CollectedData_.csv', header_rows=[1, 2], transform=data_transform)
test_data.image_names = test_data.image_names[205:]
test_data.labels = test_data.labels[205:]
test_data.compute_heatmaps()

datamod = TrackingDataModule(train_data, train_batch_size = 16, validation_batch_size = 10, test_batch_size = 1, num_workers = args.num_workers) #dlc configs
datamod.train_set = train_data
datamod.valid_set = val_data
datamod.test_set = test_data

#datamod = TrackingDataModule(full_data, train_batch_size = 16, validation_batch_size = 10, test_batch_size = 1, num_workers = args.num_workers) #dlc configs

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=100, mode="min"
)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval = 'epoch')

#datamod.setup()

trainer = pl.Trainer(gpus=args.num_gpus, log_every_n_steps = 15, callbacks=[early_stopping, lr_monitor], auto_scale_batch_size = False, reload_dataloaders_every_epoch=False)

if (not(args.no_train)):
    trainer.fit(model = model, datamodule = datamod)
else:
    datamod.setup()

if args.predict:
    model.eval()
    trainer.test(model = model, datamodule = datamod)
    threshold = False #whether or not to refrain from plotting a keypoint if the max value of the heatmap is below a certain threshold
    save_heatmaps = True #whether or not to save heatmap images, note they will be in the downsampled dimensions
    plotPredictions(save_heatmaps, threshold)
    threshold = False
    #saveNumericalPredictions(threshold)
    

