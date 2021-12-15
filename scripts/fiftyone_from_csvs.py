import hydra
from omegaconf import DictConfig
from pose_est_nets.utils.fiftyone_plotting_utils import make_dataset_and_viz_from_csvs

# takes the models specified in the eval.hydra_paths field 
# and visualizes their predictions on a fiftyone dashboard
@hydra.main(config_path="configs", config_name="config")
def fiftyone_from_csvs(cfg: DictConfig) -> None:
    make_dataset_and_viz_from_csvs(cfg)

if __name__ == "__main__":
    fiftyone_from_csvs()

    