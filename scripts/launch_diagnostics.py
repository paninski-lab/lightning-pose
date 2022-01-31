"""Visualize predictions of models in a fiftyone dashboard."""

import hydra
from omegaconf import DictConfig
from lightning_pose.utils.fiftyone import FiftyOneImagePlotter, check_dataset
import fiftyone as fo


@hydra.main(config_path="configs", config_name="config")
def visualize_training_dataset(cfg: DictConfig) -> None:
    fo_keypoint_class = FiftyOneImagePlotter(cfg=cfg)  # initializes everything
    dataset = fo_keypoint_class.create_dataset()  # loops over images and models
    check_dataset(dataset)

    if cfg.eval.fiftyone.launch_app_from_script:
        # launch an interactive session
        session = fo.launch_app(dataset, remote=True)
        session.wait()
    # otherwise launch from an ipython session


if __name__ == "__main__":
    visualize_training_dataset()
