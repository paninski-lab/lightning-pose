"""Visualize predictions of models in a fiftyone dashboard."""

import fiftyone as fo
import hydra
from omegaconf import DictConfig

from lightning_pose.utils import pretty_print_str
from lightning_pose.utils.fiftyone import FiftyOneImagePlotter, check_dataset


@hydra.main(config_path="configs", config_name="config_mirror-mouse-example")
def build_fo_dataset(cfg: DictConfig) -> None:

    pretty_print_str("Launching a job that creates FiftyOne.Dataset")

    fo_plotting_instance = FiftyOneImagePlotter(cfg=cfg)  # initializes everything
    dataset = fo_plotting_instance.create_dataset()  # internally loops over models
    check_dataset(dataset)  # create metadata and print if there are problems
    fo_plotting_instance.dataset_info_print()  # print the name of the dataset

    if cfg.eval.fiftyone.launch_app_from_script:
        # launch an interactive session
        session = fo.launch_app(
            dataset,
            remote=cfg.eval.fiftyone.remote,
            address=cfg.eval.fiftyone.address,
            port=cfg.eval.fiftyone.port,
        )
        session.wait()
    # otherwise launch from an ipython session


if __name__ == "__main__":
    build_fo_dataset()
