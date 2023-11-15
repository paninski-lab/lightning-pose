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
import shutil

import numpy as np
import pandas as pd


BASE_DIR = Path(os.path.abspath(__file__)).resolve().parents[1]
sys.path.append(os.path.join(str(BASE_DIR), 'active_pipeline'))
from active_utils import active_loop_step, get_vids, subsample_frames_from_df
#%%

def call_active_all(active_cfg):
    """
    Call active learning algorithm
    :param active_loop_config:
    :return: active_loop_config with experiments
    """
    exp_cfg = OmegaConf.load(active_cfg.active_pipeline.experiment_cfg)
    # update config file parameters if needed
    #exp_cfg = OmegaConf.merge(exp_cfg, active_cfg.active_pipeline.experiment_kwargs)
    for current_iteration in range(active_cfg.active_pipeline.start_iteration,
                                   active_cfg.active_pipeline.end_iteration + 1):

        print('\n\n Experiment iter {}'.format(current_iteration), flush=True)

        active_cfg.active_pipeline.current_iteration = current_iteration
        iteration_key_current = 'iteration_{}'.format(current_iteration)

        # TODO: add option if _test does not exits
        # TODO*: if output run is present -- only calculate additional metrics
        if current_iteration == 0:

          file_new_path = exp_cfg.data.csv_file.replace(".csv","_True.csv") # Set the Path for the Original Dataset

          # This if part is to load the original dataset
          if os.path.exists(file_new_path):
            new_df_file = pd.read_csv(file_new_path, header = [0,1,2], index_col=0)
            new_df_file.to_csv(exp_cfg.data.csv_file)

          # Load the Compeleted Original Dataset
          labeled_df = pd.read_csv(exp_cfg.data.csv_file,header = [0,1,2], index_col=0)

          # Set the Path for the first common 100 frames
          first100_path = os.path.join(os.path.dirname(exp_cfg.data.csv_file),"new_100.csv")

          # Set the Path for IND Dataset
          ref_data_path = exp_cfg.data.csv_file.replace(".csv","_active_test.csv")

          # Set the Path for Compeleted Original Dataset
          true_data_path = exp_cfg.data.csv_file.replace(".csv","_True.csv")

          # Save the Compeleted Original Dataset to the folder (Because the Compeleted Original Dataset will be split into 100 frames dataset and the active_test dataset)
          labeled_df.to_csv(true_data_path)

          # Set the random_seeds
          random_seeds = active_cfg[iteration_key_current].use_seeds[0]

          # Set the number of videos per iteration used to sample frames
          num_vids = active_cfg[iteration_key_current].num_vids

          train_frames = active_cfg[iteration_key_current].train_frames

          train_prob = active_cfg[iteration_key_current].train_prob

          if active_cfg[iteration_key_current].method == "random":
            
            
            new_df, used_vids = subsample_frames_from_df(labeled_df, num_vids, train_frames, train_prob, random_seeds)
            labeled_df.drop(index = new_df.index, inplace=True)
            labeled_df.to_csv(ref_data_path)
            new_df.to_csv(exp_cfg.data.csv_file)     
            new_df.to_csv(first100_path)

          else:

            new_df = pd.read_csv(first100_path, header = [0,1,2], index_col=0)
            print("The Training Set is:", new_df.shape[0])
            used_vids, _ = get_vids(new_df, num_vids, random_seeds)
            labeled_df.drop(index = new_df.index, inplace=True)
            labeled_df.to_csv(ref_data_path)
            new_df.to_csv(exp_cfg.data.csv_file)

        if len(active_cfg[iteration_key_current].output_prev_run) == 0:
            # if model is provided, train a model using the config file:
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration, active_cfg[iteration_key_current].method)
            dir_file_name = exp_cfg.data.csv_file.replace(".csv", '_output_dir.txt')
            if current_iteration == 0 and active_cfg[iteration_key_current].method == "random":
                train_output_dirs = run_train(active_cfg[iteration_key_current], exp_cfg)
                with open(dir_file_name, "w") as file:
                    for i in train_output_dirs:
                        file.write(i)
            elif current_iteration == 0 and active_cfg[iteration_key_current].method != "random":
                new_dir = make_run_dir()
                train_output_dirs = []
                with open(dir_file_name, "r") as file:
                    train_output_dirs.append(file.read())

                source_folder = train_output_dirs[0]
                destination_folder = os.path.abspath(new_dir)
                copy_directory_contents(source_folder, destination_folder)

            # step 3: fill in active pipeline details and call active loop
            else:
                train_output_dirs = run_train(active_cfg[iteration_key_current], exp_cfg)
            active_cfg[iteration_key_current].output_prev_run = train_output_dirs
            active_cfg[iteration_key_current].csv_file_prev_run = exp_cfg.data.csv_file

        new_train_file, used_vids = active_loop_step(active_cfg, used_vids)
        # step 4 : update the config for the next run:
        exp_cfg.data.csv_file = new_train_file

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
  new_dir = f"./Outputs/{today_str}/{ctime_str}"
  os.makedirs(new_dir, exist_ok=False)
  return new_dir


"""
def copy_directory_contents(src_dir, dst_dir):
    # 确保目标目录存在，如果不存在就创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源目录中的所有文件和文件夹
    for item in os.listdir(src_dir):
        # 构建完整的文件/文件夹路径
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        # 判断是文件还是文件夹
        if os.path.isdir(s):
            # 如果是文件夹，则递归复制
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            # 如果是文件，则复制文件
            shutil.copy2(s, d)
            
"""

def copy_directory_contents(src_dir, dst_dir):
    # Ensure the destination directory exists, create it if it does not
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Iterate over all files and folders in the source directory
    for item in os.listdir(src_dir):
        # Build the full path for files/folders
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        # Check if it is a directory or a file
        if os.path.isdir(s):
            # If it is a directory, recursively copy it
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            # If it is a file, copy the file
            shutil.copy2(s, d)



if __name__ == "__main__":
    # read active config file
    active_loop_cfg = OmegaConf.load(sys.argv[1])
    # active_loop_step(active_loop_cfg)
    call_active_all(active_loop_cfg)