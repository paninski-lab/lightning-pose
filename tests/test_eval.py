import torch
import pytorch_lightning as pl
from pose_est_nets.utils.plotting_utils import predict_dataset, predict_videos
from pose_est_nets.models.heatmap_tracker import HeatmapTracker
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.datasets.datamodules import BaseDataModule
from omegaconf import OmegaConf
import yaml
import imgaug.augmenters as iaa
import os

hydra_output_directory = "toy_datasets/toy_model"
cfg_path = "toy_datasets/toy_model/config.yaml"
cfg = OmegaConf.load(cfg_path)

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BATCH_SIZE = 12
_HEIGHT = 256  # TODO: should be different numbers?
_WIDTH = 256

data_transform = []
data_transform.append(
    iaa.Resize({
        "height": cfg.data.image_resize_dims.height,
        "width": cfg.data.image_resize_dims.width
    })
)  # dlc dimensions need to be repeatably divisable by 2
imgaug_transform = iaa.Sequential(data_transform)
heatmap_data = HeatmapDataset(
    root_directory="toy_datasets/toymouseRunningData",
    csv_path="CollectedData_.csv",
    header_rows=[1, 2],
    imgaug_transform=imgaug_transform,
)
heatmap_module = BaseDataModule(heatmap_data)  # and default args
heatmap_module.setup()

ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor="val_loss", 
    dirpath=hydra_output_directory,
    save_last=True,
    save_top_k=0
)

trainer = pl.Trainer(
        gpus=1 if _TORCH_DEVICE == "cuda" else 0,
        max_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[ckpt_callback]
)  

model = HeatmapTracker(num_targets=34)
trainer.fit(model=model, datamodule=heatmap_module)
model_ckpt = hydra_output_directory + "/last.ckpt"

def test_predict_dataset_and_predict_video():
    assert(os.path.exists(model_ckpt))
    predict_dataset(
        cfg=cfg, 
        datamod=heatmap_module, 
        hydra_output_directory=hydra_output_directory, 
        ckpt_file=model_ckpt
    )
    assert(os.path.exists("toy_datasets/toy_model/predictions.csv"))
    os.remove("toy_datasets/toy_model/predictions.csv")
    assert(os.path.exists("toy_datasets/toy_model/dataset_split_indices.json"))
    os.remove("toy_datasets/toy_model/dataset_split_indices.json")

    predict_videos(
        video_path="toy_datasets/toymouseRunningData/unlabeled_videos/",
        ckpt_file=model_ckpt,
        cfg_file=cfg,
        save_file="toy_datasets/toy_model/video_predictions.csv"
    )
    assert(os.path.exists("toy_datasets/toy_model/video_predictions.csv"))
    os.remove("toy_datasets/toy_model/video_predictions.csv")

    os.remove(model_ckpt)


