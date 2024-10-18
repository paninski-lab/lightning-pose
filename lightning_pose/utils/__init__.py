import torch
from omegaconf import ListConfig

# to ignore imports for sphix-autoapidoc
__all__ = [
    "pretty_print_str",
    "pretty_print_cfg",
]


def pretty_print_str(string: str, symbol: str = "-") -> None:
    str_length = len(string)
    print(symbol * str_length)
    print(string)
    print(symbol * str_length)


def pretty_print_cfg(cfg):

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
