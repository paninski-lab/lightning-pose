"""Visualize predictions of models in a fiftyone dashboard."""

import hydra
from omegaconf import DictConfig
from lightning_pose.utils.fiftyone import FiftyOneImagePlotter
from lightning_pose.utils.fiftyone_plotting_utils import make_dataset_and_viz_from_csvs
import fiftyone as fo


def check_dataset(dataset: fo.Dataset) -> None:
    try:
        dataset.compute_metadata(skip_failures=False)
    except ValueError:
        print("Encountered error in metadata computation. See print:")
        print(dataset.exists("metadata", False))
        print(
            "The above print should indicate bad image samples, e.g., with bad paths."
        )


@hydra.main(config_path="configs", config_name="config")
def visualize_training_dataset(cfg: DictConfig) -> None:
    fo_keypoint_class = FiftyOneImagePlotter(cfg=cfg)  # initializes everything
    dataset = fo_keypoint_class.create_dataset()  # loops over images and models
    check_dataset(dataset)

    # launch an interactive session
    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == "__main__":
    visualize_training_dataset()
