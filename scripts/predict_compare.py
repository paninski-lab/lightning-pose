"""want:
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
# from pose_est_nets.models.heatmap_tracker import (
#     HeatmapTracker,
#     SemiSupervisedHeatmapTracker,
# )
# from pose_est_nets.models.regression_tracker import (
#     RegressionTracker,
#     SemiSupervisedRegressionTracker,
# )
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.utils.fiftyone_plotting_utils import make_dataset_and_evaluate
from pose_est_nets.utils.plotting_utils import get_model_class

# path = "/home/jovyan/pose-estimation-nets/outputs/2021-09-30/00-08-34/tb_logs/my_test_model/version_0/checkpoints/epoch=1-step=105.ckpt"
# Semi supervised heatmap tracker
# path = "/home/ubuntu/pose-estimation-nets/outputs/2021-09-30/05-41-05/tb_logs/my_test_model/version_0/checkpoints/epoch=299-step=15899.ckpt"
# # regular heatmap tracker
# path2 = "/home/ubuntu/pose-estimation-nets/outputs/2021-09-30/04-27-31/tb_logs/my_test_model/version_0/checkpoints/epoch=249-step=13249.ckpt"
# # Semi supervised regression tracker
# path3 = "/home/ubuntu/pose-estimation-nets/outputs/2021-09-30/22-19-37/tb_logs/my_test_model/version_0/checkpoints/epoch=169-step=9009.ckpt"
# # regular regression tracker
# path4 = "/home/ubuntu/pose-estimation-nets/outputs/2021-09-30/23-50-18/tb_logs/my_test_model/version_0/checkpoints/epoch=146-step=7790.ckpt"

# MODEL_CLASS_FROM_TYPE = {
#     "RegressionTracker": RegressionTracker,
#     "SemiSupervisedRegressionTracker": SemiSupervisedRegressionTracker,
#     "HeatmapTracker": HeatmapTracker,
#     "SemiSupervisedHeatmapTracker": SemiSupervisedHeatmapTracker
# }

@hydra.main(config_path="configs", config_name="config")
def predict(cfg: DictConfig):
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
    losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)
    # Assume we are using heatmap dataset for evaluation even if no models are heatmap models
    dataset = HeatmapDataset(
        root_directory=cfg.data.data_dir,
        csv_path=cfg.data.csv_path,
        header_rows=OmegaConf.to_object(cfg.data.header_rows),
        imgaug_transform=imgaug_transform,
        downsample_factor=cfg.data.downsample_factor,
    )
    #We don't need the unsupervised functionality for testing purposes on labeled frames, but keep for now because it sets up loss_param_dict
    datamod = UnlabeledDataModule(
        dataset=dataset,
        video_paths_list=cfg.data.video_dir,  # just a single path for now
        specialized_dataprep=losses_to_use,
        loss_param_dict=loss_param_dict,
        train_batch_size=cfg.training.train_batch_size,
        val_batch_size=cfg.training.val_batch_size,
        test_batch_size=cfg.training.test_batch_size,
        num_workers=cfg.training.num_workers,
    )
    # datamod = BaseDataModule(
    #     dataset=dataset,
    #     train_batch_size=cfg.training.train_batch_size,
    #     validation_batch_size=cfg.training.val_batch_size,
    #     test_batch_size=cfg.training.test_batch_size,
    #     num_workers=cfg.training.num_workers,
    # )

    # if "SemiSupervisedRegressionTracker" in cfg.eval.model_types or "SemiSupervisedHeatmapTracker" in cfg.eval.model_types:
    # else:


    # for now this works without saving the pca params to dict
    bestmodels = {}
    for model_name, hydra_path in zip(cfg.eval.model_names, cfg.eval.hydra_paths):
        model_config = OmegaConf.load("../../" + hydra_path + '.hydra/config.yaml')
        ModelClass = get_model_class(model_config.model.model_type, model_config.model.semi_supervised)
        ckpt_path = "../../" + hydra_path + "tb_logs/my_test_model/version_0/checkpoints/"
        model_path = ckpt_path + os.listdir(ckpt_path)[0]
        if model_config.model.semi_supervised:
            model = ModelClass.load_from_checkpoint(
                model_path, semi_super_losses_to_use=OmegaConf.to_object(model_config.model.losses_to_use), loss_params=loss_param_dict #loss param dict is generic
            )
        else:
            model = ModelClass.load_from_checkpoint(
                model_path
            )
        bestmodels[model_name] = model
    print("loaded weights")
    make_dataset_and_evaluate(cfg, datamod, bestmodels)


    # model = SemiSupervisedHeatmapTracker.load_from_checkpoint(
    #     path, semi_super_losses_to_use=losses_to_use, loss_params=loss_param_dict
    # )
    # model2 = HeatmapTracker.load_from_checkpoint(path2)
    # model3 = SemiSupervisedRegressionTracker.load_from_checkpoint(
    #     path3, semi_super_losses_to_use=losses_to_use, loss_params=loss_param_dict
    # )
    # model4 = RegressionTracker.load_from_checkpoint(path4)

    # model = model.load_from_checkpoint(path, strict=False)
    
    # bestmodels = {
    #     "semi_supervised_heatmap_tracker": model,
    #     "base_heatmap_tracker": model2,
    #     "semi_supervised_regression_tracker": model3,
    #     "base_regression_tracker": model4,
    # }

if __name__ == "__main__":
    predict()
