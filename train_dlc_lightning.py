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


model = DLC(num_targets = 34, resnet_version = 50, transfer = False)

if (args.load):
    model = model.load_from_checkpoint(checkpoint_path = args.ckpt, num_targets = 34, resnet_version = 50, transfer = False)

data_transform = []
data_transform.append(iaa.Resize({"height": 384, "width": 384})) #dlc dimensions need to be repeatably divisable by 2
data_transform = iaa.Sequential(transform)

full_data = DLCHeatmapDataset(root_directory= 'mouse_data/data', csv_path='CollectedData_.csv', header_rows=[1, 2], transform=data_transform)
datamod = TrackingDataModule(full_data, train_batch_size = 16, validation_batch_size = 10, test_batch_size = 1, num_workers = 8) #dlc configs


early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=100, mode="min"
)

trainer = pl.Trainer(gpus=args.num_gpus, log_every_n_steps = 15, callbacks=[early_stopping], auto_scale_batch_size = False, reload_dataloaders_every_epoch=True)
if (not(args.no_train)):
    trainer.fit(model = model, datamodule = datamod)
else:
	datamod.setup()

if (args.predict)
	model.eval()
    trainer.test(model = model, datamodule = datamod)
    model.eval()
    # preds = trainer.predict(model = model, datamodule = datamod, return_predictions = True)
    preds = {}
    i = 1
    f = open('predictions.txt', 'w')
    predict_dl = datamod.test_dataloader()
    for batch in predict_dl:
        if i > 10:
            break
        x, y = batch
        plt.clf()
        out = model.forward(x)
        plt.imshow(x[0, 0])
        preds[i] = out.numpy().tolist()
        plt.scatter(out.numpy()[:,0::2], out.numpy()[:,1::2], c = 'blue')
        plt.scatter(y.numpy()[:,0::2], y.numpy()[:,1::2], c = 'orange')
        plt.savefig("predsdlc/test" + str(i) + ".png")
        i += 1
    f.write(json.dumps(preds))
    f.close()




