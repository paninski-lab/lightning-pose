# -*- coding: utf-8 -*-
"""
Call active loop pipeline
"""
from omegaconf import OmegaConf
import sys
from pathlib import Path
from datetime import datetime
import wandb
import copy
import os

BASE_DIR = Path(os.path.abspath(__file__)).resolve().parents[1]
sys.path.append(os.path.join(str(BASE_DIR), 'active_pipeline'))
from active_utils import active_loop_step
#%%

def call_active_all(active_cfg):
    """
    Call active learning algorithm
    :param active_loop_config:
    :return: active_loop_config with experiments
    """
    # Read experiment config file
    exp_cfg = OmegaConf.load(active_cfg.active_pipeline.experiment_cfg)

    # update config file parameters if needed
    #exp_cfg = OmegaConf.merge(exp_cfg, active_cfg.active_pipeline.experiment_kwargs)

    # Run active loop iterations
    for current_iteration in range(active_cfg.active_pipeline.start_iteration,
                                   active_cfg.active_pipeline.end_iteration + 1):

        print('\n\n Experiment iter {}'.format(current_iteration), flush=True)

        iteration_key_current = 'iteration_{}'.format(current_iteration)

        if current_iteration == 0:
            # step 1: select [frames to label is skipped in demo mode.
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration, 'baseline')
        else:
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration,
                                                           active_cfg[iteration_key_current].method)

        # TODO*: if output run is present -- only calculate additional metrics
        # TODO: add option if _test does not exits
        train_output_dirs = run_train(active_cfg[iteration_key_current], exp_cfg)

        # step 3: fill in active pipeline details and call active loop
        active_cfg.active_pipeline.current_iteration = current_iteration
        if active_cfg[iteration_key_current].output_prev_run != [""]:
          active_cfg[iteration_key_current].output_prev_run = active_cfg[iteration_key_current].output_prev_run + train_output_dirs 
        else:
          active_cfg[iteration_key_current].output_prev_run = train_output_dirs
        
        #print("##### DIR is:", active_cfg[iteration_key_current].output_prev_run)
        active_cfg[iteration_key_current].csv_file_prev_run = exp_cfg.data.csv_file

        new_train_file = active_loop_step(active_cfg)

        # step 4 : update the config for the next run:
        exp_cfg.data.csv_file = new_train_file

    # write new active_cfg file
    return active_cfg


def run_train(active_iter_cfg, experiment_cfg):
    train_output_dirs = []
    if active_iter_cfg.method == "Ensembling":
      for seed in active_iter_cfg.use_seeds:
        exp_cfg = copy.deepcopy(experiment_cfg)
        exp_cfg.training.rng_seed_model_pt = active_iter_cfg.use_seeds[seed]
        train_output_dir = make_run(exp_cfg)
        train_output_dirs.append(train_output_dir)
    else:
      exp_cfg = copy.deepcopy(experiment_cfg)
      train_output_dir = make_run(exp_cfg)
      train_output_dirs.append(train_output_dir)

    return train_output_dirs


def make_run(cfg):
  sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
  import train_hydra
  cwd = os.getcwd()
  new_dir = make_run_dir()
  os.chdir(new_dir)
  train_output_dir = train_hydra.train(cfg)
  os.chdir(cwd)
  if cfg.wandb.logger:
    wandb.finish()
  return train_output_dir


def make_run_dir():
  today_str = datetime.now().strftime("%y-%m-%d")
  ctime_str = datetime.now().strftime("%H-%M-%S")
  new_dir = f"./outputs/{today_str}/{ctime_str}"
  os.makedirs(new_dir, exist_ok=False)
  return new_dir


if __name__ == "__main__":
    # read active config file
    active_loop_cfg = OmegaConf.load(sys.argv[1])
    # active_loop_step(active_loop_cfg)
    call_active_all(active_loop_cfg)