"""want:
1. give paths to checkpoints (with models)
2. initialize those models
3. setup a datamodule with (train/test/val images; one or more datamodules needed?)
4. get predictions and log them to fiftyone
5. launch fiftyone
6. separately connect with ssh tunnel and inspect"""
import os
import pytorch_lightning as pl
from pose_est_nets.models.new_heatmap_tracker import SemiSupervisedHeatmapTracker
import hydra
from omegaconf import DictConfig, OmegaConf


path = "/home/jovyan/pose-estimation-nets/outputs/2021-09-29/19-14-09/tb_logs/my_test_model/version_0/checkpoints/epoch=1-step=105.ckpt"
assert os.path.isfile(path)


@hydra.main(config_path="configs", config_name="config")
def predict(cfg: DictConfig):
    loss_param_dict = OmegaConf.to_object(cfg.losses)
    losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)
    print(loss_param_dict)
    model = SemiSupervisedHeatmapTracker.load_from_checkpoint(
        path,
        loss_params=loss_param_dict,
        semi_super_losses_to_use=losses_to_use,
        strict=False,
    )


if __name__ == "__main__":
    predict()
