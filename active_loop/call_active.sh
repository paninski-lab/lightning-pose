#!/bin/bash

# Train model and store artifacts in wandb

config_path='/data/libraries/lightning-pose/active_loop/configs/config_ibl_active.yaml'

python call_active_loop.py $config_path





