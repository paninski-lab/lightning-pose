#!/bin/bash

# Train model and store artifacts in wandb

config_path='/data/libraries/lightning-pose/active_loop/configs'
# base config

config_name='config_ibl_experiment.yaml'

csv_iter1="../iteration_active_loop/experiment0/iteration_1/random_10_CollectedData.csv"

pushd ../
python scripts/train_hydra.py \
  --config-path=$config_path \
  --config-name=$config_name \
  training.min_epochs=10 \
  training.max_epochs=11 \
  wandb.logger=1 \
  wandb.params.project="lp-demo-paw" \
  #data.csv_file=${csv_iter1} \

popd

