#!/bin/bash

# Train model and store artifacts in wandb

config_path="${LIGHTPOSE_DIR}/active_loop/configs/config_ibl_active.yaml"

cd ${LIGHTPOSE_DIR}
python active_loop/call_active_loop.py $config_path





