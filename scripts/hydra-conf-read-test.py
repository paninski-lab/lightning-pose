
import hydra
from omegaconf import DictConfig, ListConfig
import os

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """Main fitting function, accessed from command line."""

    print("Our Hydra config file:")
    pretty_print(cfg)

def pretty_print(cfg):

    for key, val in cfg.items():
        if key == "eval":
            continue
        print("--------------------")
        print("%s parameters" % key)
        print("--------------------")
        for k, v in val.items():
            print("{}: {}".format(k, v))
        print()
    print("\n\n")

if __name__ == "__main__":
    """
python scripts/hydra-conf-read-test.py
python scripts/hydra-conf-read-test.py --config-dir scripts/configs --config-name config
python scripts/hydra-conf-read-test.py --config-dir scripts/configs_mirror-mouse --config-name config_mirror-mouse
grid run --localdir -- grid-hpo.sh --script scripts/hydra-conf-read-test.sh 
grid run --localdir -- run grid-hpo.sh --script scripts/hydra-conf-read-test.sh --config-dir scripts/configs --config-name config
grid run --localdir -- grid-hpo.sh --script scripts/hydra-conf-read-test.sh  --config-dir scripts/configs_mirror-mouse --config-name config_mirror-mouse

    """
    train()

