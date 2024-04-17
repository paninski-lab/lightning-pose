"""Example model training script."""

import hydra
from omegaconf import DictConfig

from lightning_pose.train import train


@hydra.main(config_path="configs", config_name="config_mirror-mouse-example")
def train_model(cfg: DictConfig):
    """Main fitting function, accessed from command line.

    To train a model on the example dataset provided with the Lightning Pose package with this
    script, run the following command from inside the lightning-pose directory
    (make sure you have activated your conda environment):

    ```
    python scripts/train_hydra.py
    ```

    Note there are no arguments - this tells the script to default to the example data.


    To train a model on your own dataset, overwrite the default config_path and config_name args:

    ```
    python scripts/train_hydra.py --config-path=<PATH/TO/YOUR/CONFIGS/DIR> --config-name=<CONFIG_NAME.yaml>  # noqa
    ```

    For more information on training models, see the docs at
    https://lightning-pose.readthedocs.io/en/latest/source/user_guide/training.html

    """

    train(cfg)


if __name__ == "__main__":
    train_model()
