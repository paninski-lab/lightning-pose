"""Command modules for the lightning-pose CLI."""

from . import train, predict, crop, remap

# List of all available commands
COMMANDS = {
    "train": train,
    "predict": predict,
    "crop": crop,
    "remap": remap,
}
