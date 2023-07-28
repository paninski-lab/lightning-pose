# -*- coding: utf-8 -*-
"""
Call active loop functions
"""
import os
import random
import pandas as pd
from omegaconf import OmegaConf
import sys
from pathlib import Path
import numpy as np
from hydra import compose, initialize, initialize_config_dir
from datetime import datetime
import wandb
import copy

import pandas as pd
import os
from lightning_pose.utils.io import get_keypoint_names
import matplotlib.pyplot as plt
BASE_DIR = Path(os.path.abspath(__file__)).resolve().parents[1]
import train_hydra
from active_loop.active_pipeline import active_loop_step
#%%
def find_common_elements(*lists):
  # Convert each list to sets and find the intersection of all sets
  common_elements = set(lists[0]).intersection(*lists[1:])
  return list(common_elements)


def make_run_dir():
  today_str = datetime.now().strftime("%y-%m-%d")
  ctime_str = datetime.now().strftime("%H-%M-%S")
  new_dir = f"./outputs/{today_str}/{ctime_str}"
  os.makedirs(new_dir, exist_ok=False)
  return new_dir


def calculate_ensemble_frames(prev_output_dirs, num_frames, header_rows=[0,1,2]):
  all_keypoints = []
  all_indices = []
  all_np=list()
  all_var=list()

  # Find common elements across all csv files
  for run_idx, folder_path in enumerate(prev_output_dirs):
    csv_file = os.path.join(folder_path, "predictions_new.csv")
    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    all_indices.append(csv_data.index.values)
  common_elements = np.asarray(find_common_elements(*all_indices))

  # read xy keypoints from each csv file
  for run_idx, folder_path in enumerate(prev_output_dirs):
    csv_file = os.path.join(folder_path, "predictions_new.csv")
    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    all_indices.append(csv_data.index.values)
    # filter by common elements
    csv_data = csv_data.loc[common_elements]
    all_np.append(csv_data.iloc[:,:-1].to_numpy())
    # num_keypoints x (x, y, likelihood) + ('train')
    # train is always true for predictions_new in test mode
    frame_ids = csv_data.index.to_numpy()
    keypoints = csv_data.to_numpy()[...,:-1]
    print(keypoints.shape)

    xy_keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)[..., :-1]
    var=keypoints.reshape(keypoints.shape[0], -1, 3)[..., -1:]
    all_var.append(var)
    print(xy_keypoints.shape)
    all_keypoints.append(xy_keypoints)
  csv_data_mean=csv_data
  csv_data_var=csv_data
  all_keypoints = np.stack(all_keypoints,0)
  all_var=np.stack(all_var,0)
  all_np=np.stack(all_np,0)
  all_np=all_np.mean(axis=0)
  csv_data_mean.iloc[:,:-1]=all_np
  # calculate variance
  variance_xy = all_keypoints.var(0)
  var_likely=all_var.var(0)
  print(variance_xy.shape)
  print(var_likely.shape)
  var_total=np.concatenate((variance_xy,var_likely),-1)
  print(var_total.shape)
  var_total=var_total.reshape(-1,12) #### might cause some issue
  csv_data_var.iloc[:,:-1]=var_total
  selected_indices = np.argsort(variance_xy.sum(-1).sum(-1))[:num_frames]

  matched_rows =  csv_data.iloc[selected_indices]

  return matched_rows

def calculate_ensemble_frames_for_active_test(prev_output_dirs, num_frames, header_rows=[0,1,2]):
  all_keypoints = []
  all_indices = []
  all_np=list()
  all_var=list()
  # Find common elements across all csv files
  for run_idx, folder_path in enumerate(prev_output_dirs):
    csv_file = os.path.join(folder_path, "predictions_active_test.csv")
    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    all_indices.append(csv_data.index.values)
  common_elements = np.asarray(find_common_elements(*all_indices))

  # read xy keypoints from each csv file
  for run_idx, folder_path in enumerate(prev_output_dirs):
    csv_file = os.path.join(folder_path, "predictions_active_test.csv")
    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    all_indices.append(csv_data.index.values)
    # filter by common elements
    csv_data = csv_data.loc[common_elements]
    all_np.append(csv_data.iloc[:,:-1].to_numpy())
    # num_keypoints x (x, y, likelihood) + ('train')
    # train is always true for predictions_new in test mode
    frame_ids = csv_data.index.to_numpy()
    keypoints = csv_data.to_numpy()[...,:-1]
    print(keypoints.shape)

    xy_keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)[..., :-1]
    var=keypoints.reshape(keypoints.shape[0], -1, 3)[..., -1:]
    all_var.append(var)
    print(xy_keypoints.shape)
    all_keypoints.append(xy_keypoints)
  csv_data_mean=csv_data
  csv_data_var=csv_data
  all_keypoints = np.stack(all_keypoints,0)
  all_var=np.stack(all_var,0)
  all_np=np.stack(all_np,0)
  all_np=all_np.mean(axis=0)
  csv_data_mean.iloc[:,:-1]=all_np
  # calculate variance
  variance_xy = all_keypoints.var(0)
  var_likely=all_var.var(0)
  print(variance_xy.shape)
  print(var_likely.shape)
  var_total=np.concatenate((variance_xy,var_likely),-1)
  print(var_total.shape)
  var_total=var_total.reshape(-1,12) #### might cause some issue
  csv_data_var.iloc[:,:-1]=var_total
  selected_indices = np.argsort(variance_xy.sum(-1).sum(-1))[:num_frames]

  matched_rows =  csv_data.iloc[selected_indices]

  return csv_data_mean, csv_data_var


def low_energy_random_sampling(energy_func, all_data,num_frames):
    """
    Args:
        energy_func (callable): Energy Function
        num_frames (int): number of Frames

    Returns:
        samples (list): List of Indcie
        
    """
    samples = []
    while len(samples) < num_frames:


        sample = np.unique(random.sample(range(len(all_data)), 1))


        energy = energy_func(sample)
        
        probability = np.exp(-energy)
        
        uniform_sample = np.unique(random.sample(range(len(all_data)), 1))
        
        if uniform_sample < probability:
            samples.append(int(sample))
    
    return np.array(samples)


def energy_function(x):
    return x**2


def initialize_iteration_folder(data_dir):
    """ Initialize the iteration folder
    :param data_dir: where the iteration folder will be created.
    clone
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def select_frames(active_iter_cfg, data_cfg):
    """
    Step 2: select frames to label
    Implement the logic for selecting frames based on the specified method:
    :param active_iter_cfg: active loop config file
    :return:
      : selected_indices_file: csv file with selected frames from active loop.
    """
    #TODO():may not want to overwrite file
    method = active_iter_cfg.method
    num_frames = active_iter_cfg.num_frames
    output_dir = active_iter_cfg.iteration_folder
    prev_output_dirs = active_iter_cfg.output_prev_run #list of directories of previous runs
    
    # We need to know the index of our selected data (list)
    if method == 'random':
      # select random frames from eval data in prev run.
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      selected_indices = np.unique(random.sample(range(len(all_data)), num_frames))  # Get index from either places
      selected_frames = all_data.iloc[selected_indices]
      matched_rows = all_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
      # Save the selected frames to a CSV file
      matched_rows.to_csv(selected_indices_file)

    elif method == 'random_energy':
        # TODO
        # select random frames from eval data in prev run.
        all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2], index_col=0)
        selected_indices = low_energy_random_sampling(energy_function,all_data , num_frames)#np.unique(random.sample(range(len(all_data)), num_frames))  # Get index from either places
        selected_frames = all_data.iloc[selected_indices]
        matched_rows=all_data.loc[selected_frames.index]
        selected_indices_file = f'iteration_{method}_indices.csv'
        selected_indices_file = os.path.join(output_dir, selected_indices_file)
      # Save the selected frames to a CSV file
        matched_rows.to_csv(selected_indices_file)

    elif method == 'uncertainty_sampling':
     # TODO:
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      csv_file = os.path.join(prev_output_dirs[0], "predictions_new.csv")
      csv_active_test_file = os.path.join(prev_output_dirs[0], "predictions_active_test.csv")
      margin = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
      margin_numpy = margin.to_numpy()[...,:-1]
      uncertainty = margin_numpy.reshape(margin_numpy.shape[0], -1, 3)[..., -1]
      margin["Uncertainity"] = uncertainty.sum(-1).astype(float)

      active_test_margin = pd.read_csv(csv_active_test_file, header=[0, 1, 2], index_col=0)
      active_test_margin_numpy = active_test_margin.to_numpy()[...,:-1]
      uncertainty = active_test_margin_numpy.reshape(active_test_margin_numpy.shape[0], -1, 3)[..., -1]
      active_test_margin["Uncertainity"] = uncertainty.sum(-1).astype(float)

      active_test_margin["Uncertainity"].plot(kind='hist', bins=20)
      plt.title('Histogram of Uncertainity Sampling')
      plt.xlabel("Sum of Uncertainity Sampling")
      plt.ylabel('Number of Frames')
      plt_path=os.path.join(output_dir, "Histogram of Uncertainity Sampling")
      plt.savefig(plt_path)
      plt.show()

      selected_frames = margin.nsmallest(num_frames, "Uncertainity")
      selected_frames = selected_frames.drop("Uncertainity", axis=1)
      matched_rows=all_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
      matched_rows.to_csv(selected_indices_file)

    elif method == 'margin sampling':
      # TODO:
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      csv_file = os.path.join(prev_output_dirs[0], "predictions_new_heatmap.csv")
      csv_active_test_file = os.path.join(prev_output_dirs[0], "predictions_active_test_heatmap.csv")
      margin = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
      margin['sum'] = margin.sum(axis=1)
      active_test_margin = pd.read_csv(csv_active_test_file, header=[0, 1, 2], index_col=0)
      active_test_margin['sum'] = active_test_margin.sum(axis=1)
      active_test_margin['sum'].plot(kind='hist', bins=20)
      plt.title('Histogram of Margin Sampling')
      plt.xlabel("Sum of Margin Sampling")
      plt.ylabel('Number of Frames')
      plt_path=os.path.join(output_dir, "Histogram of Margin Sampling")
      plt.savefig(plt_path)
      plt.show()
      selected_frames = margin.nsmallest(num_frames, 'sum')
      selected_frames = selected_frames.drop('sum', axis=1)
      matched_rows=all_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
      matched_rows.to_csv(selected_indices_file)


    elif method == 'Ensembling':
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      matched_rows = calculate_ensemble_frames(prev_output_dirs, num_frames)
      csv_data_mean, csv_data_var = calculate_ensemble_frames_for_active_test(prev_output_dirs, num_frames)
      matched_rows=all_data.loc[matched_rows.index]
      csv_mean_file = os.path.join(prev_output_dirs[-1], "predictions_active_test.csv")
      csv_data_mean.to_csv(csv_mean_file)
      csv_var_file = os.path.join(prev_output_dirs[-1], "predictions_var_active_test.csv")
      csv_data_var.drop(csv_data_var.columns[-1], axis=1, inplace=True)
      csv_data_var.drop(csv_data_var.columns[2::3], axis=1, inplace=True)
      csv_data_var.to_csv(csv_var_file)
      csv_data_var['sum'] = csv_data_var.sum(axis=1)
      csv_data_var['sum'].plot(kind='hist', bins=20)
      plt.title('Histogram of Ensembling')
      plt.xlabel("Ensembling")
      plt.ylabel('Number of Frames')
      plt_path=os.path.join(output_dir, "Histogram of Ensembling")
      plt.savefig(plt_path)
      plt.show()
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
    # Save the selected frames to a CSV file
      matched_rows.to_csv(selected_indices_file)
      
    elif method == "Single PCAS":
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      csv_file = os.path.join(prev_output_dirs[0], "predictions_new_pca_singleview_error.csv")
      single_pca = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
      single_pca['sum'] = single_pca.iloc[:,[0,1]].sum(axis=1)
      single_pca['sum'].plot(kind='hist', bins=20)
      plt.title('Histogram of Single PCAS')
      plt.xlabel("Sum of Single PCAS")
      plt.ylabel('Number of Frames')
      plt_path=os.path.join(output_dir, "Histogram of Single PCAS")
      plt.savefig(plt_path)
      plt.show()
      selected_frames = single_pca.nlargest(num_frames, 'sum')
      selected_frames = selected_frames.drop('sum', axis=1)
      matched_rows=all_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)

    # Save the selected frames to a CSV file
      matched_rows.to_csv(selected_indices_file)

    elif method == 'Equal Variance':
      # TODO:
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      csv_file = os.path.join(prev_output_dirs[0], "predictions_new_equalvariance.csv")
      csv_active_test_file = os.path.join(prev_output_dirs[0], "predictions_active_test_equalvariance.csv")
      margin = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
      margin['sum'] = margin.sum(axis=1)
      active_test_margin = pd.read_csv(csv_active_test_file, header=[0, 1, 2], index_col=0)
      active_test_margin['sum'] = active_test_margin.sum(axis=1)
      active_test_margin['sum'].plot(kind='hist', bins=20)
      plt.title('Histogram of Cosine Similarity')
      plt.xlabel("Sum of Cosine Similarity")
      plt.ylabel('Number of Frames')
      plt_path=os.path.join(output_dir, "Histogram of Cosine Similarity")
      plt.savefig(plt_path)
      plt.show()
      selected_frames = margin.nsmallest(num_frames, 'sum')
      selected_frames = selected_frames.drop('sum', axis=1)
      matched_rows=all_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
      matched_rows.to_csv(selected_indices_file)

    else:
      NotImplementedError(f'{method} is not implemented yet.')

    return selected_indices_file





def update_config_yaml(config_file, merged_data_dir):
    """
    # Step 4: Update the config.yaml file
    :param config_file:
    :param merged_data_dir:
    :return:
    """
    # Load the config file
    cfg = OmegaConf.load(config_file)

    OmegaConf.update(cfg,'data.csv_file',merged_data_dir,merge=True)

    OmegaConf.save(cfg, config_file)



def call_active_all(active_cfg):
    """
    Call active learning algorithm
    :param active_loop_config:
    :return: active_loop_config with experiments
    """
    # Read experiment config file
    exp_cfg = OmegaConf.load(active_cfg.active_loop.experiment_cfg)

    # update config file parameters if needed
    exp_cfg = OmegaConf.merge(exp_cfg, active_cfg.active_loop.experiment_kwargs)

    # Run active loop iterations
    for current_iteration in range(active_cfg.active_loop.start_iteration,
                                   active_cfg.active_loop.end_iteration + 1):

        print('\n\n Experiment iter {}'.format(current_iteration), flush=True)

        iteration_key_current = 'iteration_{}'.format(current_iteration)

        if current_iteration == 0:
            # step 1: select [frames to label is skipped in demo mode.
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration, 'baseline')
        else:
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration,
                                                           active_cfg[iteration_key_current].method)


        train_output_dirs = run_train(active_cfg[iteration_key_current], exp_cfg)

        # step 3: fill in active pipeline details and call active loop
        active_cfg.active_loop.current_iteration = current_iteration
        active_cfg[iteration_key_current].output_prev_run = train_output_dirs
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
  cwd = os.getcwd()
  new_dir = make_run_dir()
  os.chdir(new_dir)
  train_output_dir = train_hydra.train(cfg)
  os.chdir(cwd)
  if cfg.wandb.logger:
    wandb.finish()
  return train_output_dir




if __name__ == "__main__":
    # read active config file
    active_loop_cfg = OmegaConf.load(sys.argv[1])
    # active_loop_step(active_loop_cfg)
    call_active_all(active_loop_cfg)