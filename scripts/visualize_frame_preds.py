"""process:
1. give paths to checkpoints (with models)
2. initialize those models
3. setup a datamodule with (train/test/val images; one or more datamodules needed?)
4. get predictions and log them to fiftyone
5. launch fiftyone
6. separately connect with ssh tunnel and inspect"""

import os
import time
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
import imgaug.augmenters as iaa
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.utils.fiftyone_plotting_utils import make_dataset_and_evaluate
from pose_est_nets.utils.plotting_utils import (
    get_model_class,
    load_model_from_checkpoint,
)
from pose_est_nets.utils.io import (
    get_absolute_hydra_path_from_hydra_str,
    ckpt_path_from_base_path,
)
from pose_est_nets.utils.io import get_absolute_data_paths


def check_eval_model_names(names, hydra_paths) -> bool:
    hydra_paths_list = OmegaConf.to_object(hydra_paths)
    flag = False
    if names is None:
        print(
            "cfg.eval.model_names is None, taking name from the model's cfg.model.model_name"
        )
        flag = True
    else:
        eval_model_names = OmegaConf.to_object(names)
        if isinstance(eval_model_names, list):
            if len(eval_model_names) != len(hydra_paths_list):
                print(
                    "Warning: cfg.eval.model_names and cfg.eval.hydra_paths are lists of different lengths, while they should match. taking name from the model's cfg.model.model_name"
                )
                flag = True
    return flag


def check_old_model_names(
    model_name: str, model_ind: int, bestmodels: dict, hydra_relative_path
) -> str:
    if model_name in bestmodels.keys():
        print(
            "Warning: Got more than one model with the same name. \n",
            "Check cfg.model.model_name for each model. \n",
            "In this situation we recommended to provide concise names in cfg.eval.model_names corresponding to the cfg.eval.hydra_paths",
        )
        model_name = model_name + "_" + str(model_ind)
        print(
            "Renamed the model corresponding to {} as: {}".format(
                hydra_relative_path, model_name
            )
        )
    return model_name


@hydra.main(config_path="configs", config_name="config")
def predict(cfg: DictConfig):

    data_dir, video_dir = get_absolute_data_paths(cfg.data)

    # How to transform the images
    data_transform = []
    data_transform.append(
        iaa.Resize(
            {
                "height": cfg.data.image_resize_dims.height,
                "width": cfg.data.image_resize_dims.width,
            }
        )
    )
    imgaug_transform = iaa.Sequential(data_transform)
    # Losses used for models
    loss_param_dict = OmegaConf.to_object(cfg.losses)
    if cfg.model.losses_to_use is not None:
        losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)
    else:
        losses_to_use = []
    # Assume we are using heatmap dataset for evaluation even if no models are heatmap models
    dataset = HeatmapDataset(
        root_directory=data_dir,
        csv_path=cfg.data.csv_file,
        header_rows=OmegaConf.to_object(cfg.data.header_rows),
        imgaug_transform=imgaug_transform,
        downsample_factor=cfg.data.downsample_factor,
    )
    # We don't need the unsupervised functionality for testing purposes on labeled frames, but keep for now because it sets up loss_param_dict
    datamod = UnlabeledDataModule(
        dataset=dataset,
        video_paths_list=video_dir,  # just a single path for now
        specialized_dataprep=losses_to_use,
        loss_param_dict=loss_param_dict,
        train_batch_size=cfg.training.train_batch_size,
        val_batch_size=cfg.training.val_batch_size,
        test_batch_size=cfg.training.test_batch_size,
        num_workers=cfg.training.num_workers,
    )

    bestmodels = {}

    use_original_model_names = check_eval_model_names(
        cfg.eval.model_names, cfg.eval.hydra_paths
    )

    for model_ind, hydra_relative_path in enumerate(cfg.eval.hydra_paths):
        absolute_cfg_path = get_absolute_hydra_path_from_hydra_str(hydra_relative_path)
        model_cfg = OmegaConf.load(
            os.path.join(absolute_cfg_path, ".hydra/config.yaml")
        )  # path for the cfg file saved from the current trained model

        model_name = model_cfg.model.model_name

        if use_original_model_names:  # the name you gave the model when you trained it
            display_name = model_cfg.model.model_name
        else:  # the new display name from the eval config
            display_name = cfg.eval.model_names[model_ind]

        ckpt_file = ckpt_path_from_base_path(
            base_path=absolute_cfg_path, model_name=model_name
        )

        model = load_model_from_checkpoint(cfg=cfg, ckpt_file=ckpt_file)

        display_name = check_old_model_names(
            display_name, model_ind, bestmodels, hydra_relative_path
        )

        bestmodels[display_name] = model

    make_dataset_and_evaluate(cfg, datamod, bestmodels)


if __name__ == "__main__":
    predict()
