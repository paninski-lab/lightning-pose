
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
# will still work not not correct
python scripts/hydra-conf-read-test.py --config-dir scripts/configs_mirror-mouse  --config-name config
python scripts/hydra-conf-read-test.py --config-dir scripts --config-name config

# correctly way to read the file
python scripts/hydra-conf-read-test.py --config-path configs --config-name config
python scripts/hydra-conf-read-test.py --config-path configs_mirror-mouse --config-name config_mirror-mouse 

# config dir test
grid run --instance_type t2.medium --localdir -- grid-hpo.sh --script scripts/hydra-conf-read-test.sh --config-dir "['scripts/configs_mirror-mouse', 'script/configs']" --config-name "['config','config_mirror-mouse']"

# config path test
grid run --instance_type t2.medium --localdir -- grid-hpo.sh --script scripts/hydra-conf-read-test.sh --config-path "['configs_mirror-mouse', 'configs']" --config-name "['config','config_mirror-mouse']" --training.rng_seed_data_pt "[1,2]"

grid run --instance_type t2.medium --localdir -- grid-hpo.sh --script scripts/hydra-conf-read-test.sh --config-path "['configs_mirror-mouse', 'configs']" --config-name "['config','config_mirror-mouse']"

# actual run 
grid run --instance_type g4dn.xlarge --localdir -- grid-hpo.sh --script scripts/train_hydra.sh --config-path "['configs_mirror-mouse', 'configs']" --config-name "['config','config_mirror-mouse']" --training.rng_seed_data_pt "[1,2]"

    """
    train()

