"""Command modules for the lightning-pose CLI."""

from . import crop, predict, remap, run_app, train

# List of all available commands
COMMANDS = {
    "train": train,
    "predict": predict,
    "crop": crop,
    "remap": remap,
    "run_app": run_app,
}
