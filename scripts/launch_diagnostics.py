"""Visualize predictions of models in a fiftyone dashboard."""

import hydra
from omegaconf import DictConfig
from lightning_pose.utils.fiftyone_plotting_utils import make_dataset_and_viz_from_csvs


@hydra.main(config_path="configs", config_name="config")
def fiftyone_from_csvs(cfg: DictConfig) -> None:
    make_dataset_and_viz_from_csvs(cfg)


if __name__ == "__main__":
    fiftyone_from_csvs()
