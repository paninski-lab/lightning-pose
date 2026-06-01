"""Command modules for the lightning-pose CLI."""

from . import create_bbox, crop, predict, remap, run_app, smooth_bbox, train

# List of all available commands
COMMANDS = {
    'train': train,
    'predict': predict,
    'create_bbox': create_bbox,
    'smooth_bbox': smooth_bbox,
    'crop': crop,
    'remap': remap,
    'run_app': run_app,
}
