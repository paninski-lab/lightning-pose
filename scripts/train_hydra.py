import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import imgaug.augmenters as iaa
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.models.regression_tracker import RegressionTracker, SemiSupervisedRegressionTracker
from pose_est_nets.models.new_heatmap_tracker import HeatmapTracker, SemiSupervisedHeatmapTracker

import os

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: move the datapaths from cfg.training
@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print(cfg)
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
    if cfg.data.data_type == 'regression':
        dataset = BaseTrackingDataset(
            root_directory=cfg.data.data_dir,
            csv_path=cfg.data.csv_path,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform         
        )
    elif cfg.data.data_type == 'heatmap':
        print(OmegaConf.to_object(cfg.data.header_rows), type(OmegaConf.to_object(cfg.data.header_rows)))
        dataset = HeatmapDataset(
            root_directory=cfg.data.data_dir,
            csv_path=cfg.data.csv_path,
            header_rows=OmegaConf.to_object(cfg.data.header_rows),
            imgaug_transform=imgaug_transform,
            downsample_factor=cfg.data.downsample_factor
        )
    else:
        print("INVALID DATASET SPECIFIED")
        exit()

    if not (cfg.model['semi_supervised']):
        datamod = BaseDataModule(
            dataset=dataset,
            train_batch_size=cfg.training.train_batch_size,
            validation_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers
        )
        if cfg.data.data_type == 'regression':
            model = RegressionTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=50
            )

        elif cfg.data.data_type == 'heatmap':
            model = HeatmapTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=50,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=dataset.output_shape    
            )
        else:
            print("INVALID DATASET SPECIFIED")
            exit()
   
    else:
        loss_param_dict = OmegaConf.to_object(cfg.losses)
        losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)    
        datamod = UnlabeledDataModule(
            dataset=dataset,
            video_paths_list=cfg.data.video_dir, #just a single path for now
            specialized_dataprep=losses_to_use, 
            loss_param_dict=loss_param_dict, 
            train_batch_size=cfg.training.train_batch_size,
            validation_batch_size=cfg.training.val_batch_size,
            test_batch_size=cfg.training.test_batch_size,
            num_workers=cfg.training.num_workers
        )
        if cfg.data.data_type == 'regression':
            model = SemiSupervisedRegressionTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                loss_params=loss_param_dict,
                semi_super_losses_to_use=losses_to_use
            )

        elif cfg.data.data_type == 'heatmap':
            model = SemiSupervisedHeatmapTracker(
                num_targets=cfg.data.num_targets,
                resnet_version=cfg.model.resnet_version,
                downsample_factor=cfg.data.downsample_factor,
                output_shape=dataset.output_shape,
                loss_params=loss_param_dict,
                semi_super_losses_to_use=losses_to_use
            )
    trainer = pl.Trainer(
        gpus=1 if _TORCH_DEVICE == "cuda" else 0,
        max_epochs=1,
        log_every_n_steps=1,
    ) 
    trainer.fit(model=model, datamodule=datamod)
    
if __name__ == "__main__":
    train()


#     print(OmegaConf.to_yaml(cfg))
#     # Init dataset
    
#     print(dataset)

#     video_files = [
#         cfg.training.video_dir + "/" + f for f in os.listdir(cfg.training.video_dir)
#     ]
#     assert os.path.exists(
#         video_files[0]
#     )  # TODO: temporary. taking just the first video file

#     # TODO: make sure that we're managing the discrepency between loss param dict and our new hydra approach
#     # datamod = UnlabeledDataModule(
#     #     dataset=dataset,
#     #     video_paths_list=video_files[0],
#     #     specialized_dataprep="pca",
#     #     loss_param_dict=loss_param_dict,
#     # )

