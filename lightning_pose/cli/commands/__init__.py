"""Command modules for the lightning-pose CLI."""

from . import crop, predict, remap, train

# List of all available commands
COMMANDS = {
    "train": train,
    "predict": predict,
    "crop": crop,
    "remap": remap,
}
