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
#%%
def find_common_elements(*lists):
  # Convert each list to sets and find the intersection of all sets
  common_elements = set(lists[0]).intersection(*lists[1:])
  return list(common_elements)

def calculate_ensemble_frames(prev_output_dirs, num_frames, header_rows=[0,1,2]):
  all_keypoints = []
  all_indices = []
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
    # num_keypoints x (x, y, likelihood) + ('train')
    # train is always true for predictions_new in test mode
    frame_ids = csv_data.index.to_numpy()
    keypoints = csv_data.to_numpy()[...,:-1]
    xy_keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)[..., :-1]
    all_keypoints.append(xy_keypoints)

  all_keypoints = np.stack(all_keypoints,0)

  # calculate variance
  variance_xy = all_keypoints.var(0)
  selected_indices = np.argsort(variance_xy.sum(-1).sum(-1))[:num_frames]

  matched_rows =  csv_data.iloc[selected_indices]
  return matched_rows


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
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2], index_col=0)

      all_data['sum'] = all_data.iloc[:, [3, 6]].sum(axis=1)

    # Select the top 10 rows with the smallest sum
      selected_frames = all_data.nsmallest(num_frames, 'sum')

      selected_frames = selected_frames.drop('sum', axis=1)

      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)

    # Save the selected frames to a CSV file
      selected_frames.to_csv(selected_indices_file)

    elif method == 'margin sampling':
      # TODO:
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
      csv_file = os.path.join(prev_output_dirs[0], "predictions_new_heatmap.csv")
      margin = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
      margin['sum'] = margin.sum(axis=1)
      margin['sum'].plot(kind='hist', bins=20)
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
      matched_rows = calculate_ensemble_frames(prev_output_dirs, num_frames)
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

    else:
      NotImplementedError(f'{method} is not implemented yet.')

    return selected_indices_file


def merge_collected_data(active_iter_cfg, selected_frames_file):
    """
    # Step 3: Merge new CollectedData.csv with the original CollectedData.csv:

    # merge Collected_data.csv to include iteration_{}_indices.csv
    # remove iteration_{}_indices.csv from  {CollectedData}_new.csv

    :param active_iter_cfg:
    :return:
    """

    # read train frames
    train_data_file = os.path.join(active_iter_cfg.train_data_file_prev_run)
    train_data = pd.read_csv(train_data_file, header=[0,1,2], index_col=0)

    # read active test frames:
    act_test_data_file=train_data_file.replace(".csv","_active_test.csv") ### New Add
    act_test_data = pd.read_csv(act_test_data_file, header=[0,1,2], index_col=0)
    act_test_data.to_csv(active_iter_cfg.act_test_data_file)
    
    # read selected frames
    selected_frames_df = pd.read_csv(selected_frames_file, header=[0,1,2], index_col=0)

    # concat train data and selected frames and merge
    # TODO: check relative to the data path.
    new_train_data = pd.concat([train_data, selected_frames_df])
    new_train_data.to_csv(active_iter_cfg.train_data_file)

    # remove selected_frames from val data
    val_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2], index_col=0)
    val_data.drop(index=selected_frames_df.index, inplace=True)
    val_data.to_csv(active_iter_cfg.eval_data_file)
    return


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


def active_loop_step(active_loop_cfg):
    """
    TODO(haotianxiansti) update comments
    TODO(haotianxiansti) update to use hydra?
    # Step 6: Launch the next active_loop iteration
    """
    # read yaml file
    experiment_cfg = OmegaConf.load(active_loop_cfg.active_loop.experiment_cfg)

    # read params for current active loop iteration
    iteration_number = active_loop_cfg.active_loop.current_iteration
    iterations_folders = active_loop_cfg.active_loop.iterations_folders

    iteration_key = 'iteration_{}'.format(iteration_number)
    active_iter_cfg = active_loop_cfg[iteration_key]

    dir_collected_csv = Path(experiment_cfg.data.data_dir, experiment_cfg.data.csv_file).parent.absolute()
    iteration_folder = os.path.abspath(str(dir_collected_csv / active_loop_cfg.active_loop.iterations_folders / iteration_key)) ####iterations_folder -> iteration_folders

    # Read train and eval files

    # might need to reconsider making these as lists
    train_data_file_prev_run = str(Path(experiment_cfg.data.data_dir,
                                        active_iter_cfg.csv_file_prev_run))
    eval_data_file_prev_run = train_data_file_prev_run.replace('.csv', '_new.csv')
    act_test_data_file_prev_run = train_data_file_prev_run.replace('.csv', '_active_test.csv') ### New add

    # update active loop params to config file
    active_iter_cfg.iteration_key = iteration_key
    active_iter_cfg.iteration_prefix = '{}_{}'.format(active_iter_cfg.method,
                                                      active_iter_cfg.num_frames)
    active_iter_cfg.iteration_folder = iteration_folder
    active_iter_cfg.train_data_file_prev_run = train_data_file_prev_run
    active_iter_cfg.eval_data_file_prev_run = eval_data_file_prev_run
    active_iter_cfg.act_test_data_file_prev_run = act_test_data_file_prev_run ### New add

    # TODO:check if needs to be relative
    active_iter_cfg.train_data_file = os.path.join(
        active_iter_cfg.iteration_folder,
        '{}_{}'.format(active_iter_cfg.iteration_prefix,
                       os.path.basename(train_data_file_prev_run))
    )
    active_iter_cfg.eval_data_file = active_iter_cfg.train_data_file.replace('.csv', '_new.csv')
    active_iter_cfg.act_test_data_file = active_iter_cfg.train_data_file.replace('.csv', '_active_test.csv') ### New add

    # Active Loop parameters
    # Step 1: Initialize the iteration folder
    #  TODO(haotianxiansti):  add code for iter 0 (select frames when no labeles are present)
    initialize_iteration_folder(active_iter_cfg.iteration_folder)


    iteration_key_next = 'iteration_{}'.format(iteration_number) # +1 is removed
    selected_frames_file = select_frames(active_iter_cfg, experiment_cfg.data)

    # Now, we have in the directory:
    # created Collected_data_new_merged and Collected_data_merged.csv
    merge_collected_data(active_iter_cfg, selected_frames_file)
    # run algorithm with new config file
    # TODO: check location of new csv for iteration relative to data_dir
    # it should have CollectedData.csv and CollectedData_new.csv
    relpath = os.path.relpath(active_iter_cfg.train_data_file, experiment_cfg.data.data_dir)
    #print('rerun algorithm with new config file:\n{}'.format(relpath), flush=True)

    return relpath


def call_active_all(active_cfg):
    """
    # Step 5: Call active learning algorithm
    :param config:
    :return:
    """
    # Read experiment config file
    exp_cfg = OmegaConf.load(active_cfg.active_loop.experiment_cfg)

    # inherit params from active loop:
    exp_cfg.wandb.params.project = active_cfg.project
    if active_cfg.active_loop.fast_dev_run == 1:
        exp_cfg.training.fast_dev_run = True

    for current_iteration in range(active_cfg.active_loop.start_iteration,
                                   active_cfg.active_loop.end_iteration + 1):

        print('\n\n Experiment iter {}'.format(current_iteration), flush=True)

        if current_iteration == 0:
            # step 1: select frames to label is skipped in demo mode.
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration, 'baseline')

        # step 2: train model using exp_cfg
        iteration_key_current = 'iteration_{}'.format(current_iteration)
        train_output_dirs = run_train(active_cfg[iteration_key_current], exp_cfg)

        # step 3: call active loop

        iteration_key = 'iteration_{}'.format(current_iteration)  # think here!!!#####
        active_cfg.active_loop.current_iteration = current_iteration
        active_cfg[iteration_key].output_prev_run = train_output_dirs  # need to uncomment
        active_cfg[iteration_key].csv_file_prev_run = exp_cfg.data.csv_file  #?

        #print('\n\nActive loop config after iter {}'.format(current_iteration), active_cfg, flush=True)
        new_train_file = active_loop_step(active_cfg)

        # update config file
        exp_cfg.data.csv_file = new_train_file
        exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration + 1,
                                                       active_cfg[iteration_key].method)

    # write new active_cfg file
    return active_cfg


def run_train(active_iter_cfg, cfg):
    train_output_dirs = []
    if active_iter_cfg.method == "Ensembling":
      for seed in active_iter_cfg.use_seeds:
        exp_cfg = copy.deepcopy(cfg)
        exp_cfg.training.rng_seed_model_pt=active_iter_cfg.use_seeds[seed]
        train_output_dir = make_run(exp_cfg)
        train_output_dirs.append(train_output_dir)
    else:
      exp_cfg = copy.deepcopy(cfg)
      train_output_dir = make_run(exp_cfg)
      train_output_dirs.append(train_output_dir)

    return train_output_dirs


def make_run(cfg):
  sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
  import train_hydra
  cwd = os.getcwd()
  today_str = datetime.now().strftime("%y-%m-%d")
  ctime_str = datetime.now().strftime("%H-%M-%S")
  new_dir = f"./outputs/{today_str}/{ctime_str}" #_active_iter_{str(current_iteration)}"
  os.makedirs(new_dir, exist_ok=False)
  os.chdir(new_dir)
  train_output_dir = train_hydra.train(cfg)
  os.chdir(cwd)
  wandb.finish()
  return train_output_dir


if __name__ == "__main__":
    # read active config file
    active_loop_cfg = OmegaConf.load(sys.argv[1])
    # active_loop_step(active_loop_cfg)
    call_active_all(active_loop_cfg)