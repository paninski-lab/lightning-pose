import hydra
from omegaconf import DictConfig, OmegaConf
import imgaug.augmenters as iaa
from pose_est_nets.datasets.datasets import BaseTrackingDataset
from pose_est_nets.datasets.datamodules import UnlabeledDataModule

import os

# TODO: move the datapaths from cfg.training
@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    data_transform = []
    data_transform.append(
        iaa.Resize(
            {
                "height": cfg.training.image_resize_dims.height,
                "width": cfg.training.image_resize_dims.width,
            }
        )
    )
    print(type(cfg.losses))
    print(cfg.losses.keys())  # we can indeed iterate over these.
    imgaug_transform = iaa.Sequential(data_transform)
    print(imgaug_transform)
    print(OmegaConf.to_yaml(cfg))
    # Init dataset
    dataset = BaseTrackingDataset(
        root_directory=cfg.training.data_dir,
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    print(dataset)

    video_files = [
        cfg.training.video_dir + "/" + f for f in os.listdir(cfg.training.video_dir)
    ]
    assert os.path.exists(
        video_files[0]
    )  # TODO: temporary. taking just the first video file

    # TODO: make sure that we're managing the discrepency between loss param dict and our new hydra approach
    # datamod = UnlabeledDataModule(
    #     dataset=dataset,
    #     video_paths_list=video_files[0],
    #     specialized_dataprep="pca",
    #     loss_param_dict=loss_param_dict,
    # )


if __name__ == "__main__":
    train()
