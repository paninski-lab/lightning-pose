#!/bin/bash

# Train model and store artifacts in wandb

config_path="${LIGHTPOSE_DIR}/active_pipeline/configs/config_ibl_active.yaml"

cd ${LIGHTPOSE_DIR}
python active_pipeline/call_active_loop.py $config_path





