#!/bin/bash

# Train model and store artifacts in wandb

config_path='/content/lightning-pose/active_loop/configs/config_ibl_active.yaml'

python /content/lightning-pose/active_loop/call_active_loop.py $config_path





