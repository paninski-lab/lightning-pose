import os
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pose_est_nets.models.regression_tracker import RegressionTracker
from pose_est_nets.datasets.datasets import BaseTrackingDataset
from pose_est_nets.datasets.datamodules import UnlabeledDataModule
from pose_est_nets.callbacks.freeze_unfreeze_callback import (
    FeatureExtractorFreezeUnfreeze,
)
import matplotlib.pyplot as plt
import json
import argparse
import torch
from pose_est_nets.utils.IO import set_or_open_folder
from pose_est_nets.utils.wrappers import predict_plot_test_epoch
import imgaug.augmenters as iaa
import yaml

parser = argparse.ArgumentParser()
# checkout https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
parser.add_argument(
    "--data_dir", type=str, default=None, help="folder with csv and images"
)
parser.add_argument(
    "--unlabeled_video_dir", type=str, default=None, help="folder with unlabeled videos"
)
parser.add_argument(
    "--predict",
    action="store_true",
    help="whether or not to generate predictions on test data",
)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--validation_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--num_gpus", type=int, default=0)
parser.add_argument("--max_epochs", type=int, default=3)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--early_stop_patience", type=int, default=6)
parser.add_argument("--unfreezing_epoch", type=int, default=50)
parser.add_argument("--dropout_rate", type=float, default=0.2)
parser.add_argument(
    "--semisup_losses",
    type=str,
    nargs="+",  # more than one arg
    help="e.g., --semisup_losses pca bla to parse as a list of strs [pca, bla]",
)
parser.add_argument("--downsample_factor", type=int, default=2)
parser.add_argument(
    "--image_resize_dims",
    type=int,
    nargs=2,
    help="e.g., --image_resize_dims 384 384]",
)
args = parser.parse_args()
# # To show the results of the given option to screen.
# for _, value in parser.parse_args()._get_kwargs():
#     if value is not None:
#         print(value)

# define data transformations
data_transform = []
data_transform.append(
    iaa.Resize(
        {"height": args.image_resize_dims[0], "width": args.image_resize_dims[1]}
    )
)
imgaug_transform = iaa.Sequential(data_transform)

dataset = BaseTrackingDataset(
    root_directory=args.data_dir,
    csv_path="CollectedData_.csv",
    header_rows=[1, 2],
    imgaug_transform=imgaug_transform,
)
video_files = [args.video_directory + "/" + f for f in os.listdir(args.video_directory)]
assert os.path.exists(
    video_files[0]
)  # TODO: temporary. taking just the first video file


# with open("pose_est_nets/losses/default_hypers.yaml") as f:
#     loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)
# print(loss_param_dict)
# semi_super_losses_to_use = ["pca"]
# datamod = UnlabeledDataModule(
#     dataset=dataset,
#     video_paths_list=video_files[0],
#     specialized_dataprep="pca",
#     loss_param_dict=loss_param_dict,
# )

# model = RegressionTracker(
#     num_targets=34,
#     resnet_version=50,
#     transfer=True,
#     representation_dropout_rate=args.dropout_rate,
# )

# data_module = TrackingDataModule(
#     dataset,
#     train_batch_size=args.train_batch_size,
#     validation_batch_size=args.validation_batch_size,
#     test_batch_size=args.test_batch_size,
#     num_workers=args.num_workers,
# )

# early_stopping = pl.callbacks.EarlyStopping(
#     monitor="val_loss", patience=args.early_stop_patience, mode="min"
# )

# transfer_unfreeze_callback = FeatureExtractorFreezeUnfreeze(args.unfreezing_epoch)

# callback_list = []
# if (
#     args.early_stop_patience < 100
# ):  # patience values above 100 are impractical, train to convergence
#     callback_list.append(early_stopping)
# if args.unfreezing_epoch > 0:  # if unfreezing_epoch=0, don't use the callback
#     callback_list.append(transfer_unfreeze_callback)

# trainer = pl.Trainer(
#     gpus=args.num_gpus,
#     log_every_n_steps=15,
#     callbacks=callback_list,
#     auto_scale_batch_size=False,
#     check_val_every_n_epoch=10,
#     max_epochs=args.max_epochs,
# )  # auto_scale_batch_size not working

# trainer.fit(model=model, datamodule=data_module)
