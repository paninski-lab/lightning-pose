"""Utility functions for configuration display."""

from omegaconf import DictConfig, ListConfig

# to ignore imports for sphix-autoapidoc
__all__ = [
    "pretty_print_str",
    "pretty_print_cfg",
]


def pretty_print_str(string: str, symbol: str = "-") -> None:
    """Print a string surrounded by a horizontal rule made of the given symbol.

    Args:
        string: the text to print.
        symbol: single character used to draw the horizontal rule.
    """
    str_length = len(string)
    print(symbol * str_length)
    print(string)
    print(symbol * str_length)


def pretty_print_cfg(cfg: DictConfig | ListConfig) -> None:
    """Print a human-readable summary of the config, skipping the ``eval`` section.

    Args:
        cfg: hydra config to display.
    """
    for key, val in cfg.items():
        if key == "eval":
            continue
        print("--------------------")
        print(f"{key} parameters")
        print("--------------------")
        if hasattr(val, "items"):
            for k, v in val.items():
                print(f"{k}: {v}")
        if isinstance(val, str):
            print(val)
        print()
    print("\n\n")
