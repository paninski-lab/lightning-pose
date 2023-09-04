from omegaconf import DictConfig, OmegaConf
import os
import lightning.pytorch as pl

from lightning_pose.utils import pretty_print_str, pretty_print_cfg
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import predict_dataset
from lightning_pose.utils.scripts import (
    export_predictions_and_labeled_video,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
    get_callbacks,
    calculate_train_batches,
    compute_metrics,
)

import matplotlib.pyplot as plt

# read hydra configuration file from lightning-pose/scripts/configs/config_mirror-mouse-example.yaml
cfg = OmegaConf.load("/home/farzad/projects/lightning-pose/scripts/configs/config_7m.yaml")

# get absolute data and video directories for toy dataset
path="/mnt/scratch2/farzad/7m"
data_dir = os.path.join(path, cfg.data.data_dir)
video_dir = os.path.join(path, cfg.data.data_dir, cfg.data.video_dir)
cfg.data.data_dir = data_dir
cfg.data.video_dir = video_dir

assert os.path.isdir(cfg.data.data_dir), "data_dir not a valid directory"
assert os.path.isdir(cfg.data.video_dir), "video_dir not a valid directory"


# build dataset, model, and trainer

# make training short for a demo (we usually do 300)
# cfg.training.min_epochs = 300
# cfg.training.max_epochs = 400
# cfg.training.batch_size = 64

# build imgaug transform
imgaug_transform = get_imgaug_transform(cfg=cfg)

# build dataset
dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)
# dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=None)

# build datamodule; breaks up dataset into train/val/test
data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

# build loss factory which orchestrates different losses
loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

# build model
model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

# logger
logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

# early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
callbacks = get_callbacks(cfg)

# calculate number of batches for both labeled and unlabeled data per epoch
limit_train_batches = calculate_train_batches(cfg, dataset)

# set up trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=cfg.training.max_epochs,
    min_epochs=cfg.training.min_epochs,
    check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
    log_every_n_steps=cfg.training.log_every_n_steps,
    callbacks=callbacks,
    logger=logger,
    limit_train_batches=limit_train_batches,
)


# Train the model (approx 15-20 mins on this T4 GPU machine)
trainer.fit(model=model, datamodule=data_module)

from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the date and time as a string in the desired format
formatted_now = now.strftime("%Y-%m-%d/%H-%M-%S")

output_directory = os.path.join(path+"/outputs", formatted_now)
os.makedirs(output_directory)
print(f"Created an output directory at: {output_directory}")

# get best ckpt
best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)

# check if best_ckpt is a file
if not os.path.isfile(best_ckpt):
    raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

# make unaugmented data_loader if necessary
if cfg.training.imgaug != "default":
    cfg_pred = cfg.copy()
    cfg_pred.training.imgaug = "default"
    imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
    dataset_pred = get_dataset(
        cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred
    )
    data_module_pred = get_data_module(cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
    data_module_pred.setup()
else:
    data_module_pred = data_module

# compute and save frame-wise predictions
pretty_print_str("Predicting train/val/test images...")
preds_file = os.path.join(output_directory, "predictions.csv")
predict_dataset(
    cfg=cfg,
    trainer=trainer,
    model=model,
    data_module=data_module_pred,
    ckpt_file=best_ckpt,
    preds_file=preds_file,
)

# compute and save various metrics
try:
    compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
except Exception as e:
    print(f"Error computing metrics\n{e}")

artifacts = os.listdir(output_directory)
print("Generated the following diagnostic csv files:")
print(artifacts)



# for this demo data, we define 
cfg.eval.test_videos_directory = video_dir
# feel free to change this according to the folder you want to predict
assert os.path.isdir(cfg.eval.test_videos_directory)

if cfg.eval.test_videos_directory is None:
    filenames = []
else:
    filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
    vidstr = "video" if (len(filenames) == 1) else "videos"
    pretty_print_str(
        f"Found {len(filenames)} {vidstr} to predict on (in cfg.eval.test_videos_directory)")

for i, video_file in enumerate(filenames):
    assert os.path.isfile(video_file)
    pretty_print_str(f"Predicting video: {video_file}...")
    # get save name for prediction csv file
    video_pred_dir = os.path.join(output_directory, "video_preds")
    img_pred_dir = os.path.join(output_directory, "img_preds")
    video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
    prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
    # get save name labeled video csv
    if cfg.eval.save_vids_after_training:
        labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
        labeled_mp4_file = os.path.join(labeled_vid_dir, video_pred_name + "_labeled.mp4")
    else:
        labeled_mp4_file = None
    # predict on video
    export_predictions_and_labeled_video(
        video_file=video_file,
        cfg=cfg,
        ckpt_file=best_ckpt,
        prediction_csv_file=prediction_csv_file,
        labeled_mp4_file=labeled_mp4_file,
        trainer=trainer,
        model=model,
        data_module=data_module_pred,
        save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
    )
    # compute and save various metrics
    try:
        compute_metrics(
            cfg=cfg, preds_file=prediction_csv_file, data_module=data_module_pred
        )
    except Exception as e:
        print(f"Error predicting on video {video_file}:\n{e}")
        continue




