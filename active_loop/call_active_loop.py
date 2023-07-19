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


def select_frames(active_iter_cfg,num_keypoints,seeds_list,prev_output_dir):
    """
    Step 2: select frames to label
    Implement the logic for selecting frames based on the specified method:
    :param active_iter_cfg: active loop config file
    :return:
      : selected_indices_file: csv file with selected frames from active loop.
    """
    #TODO(may not want to overwrite file)
    method = active_iter_cfg.method
    num_frames = active_iter_cfg.num_frames
    output_dir = active_iter_cfg.iteration_folder
    #prev_output_dir=active_iter_cfg.output_prev_run #parent dir for prediction csv
    
    # We need to know the index of our selected data (list)
    if method == 'random':
      # select random frames from eval data in prev run.
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2], index_col=0)
      selected_indices = low_energy_random_sampling(energy_function,all_data , num_frames)#np.unique(random.sample(range(len(all_data)), num_frames))  # Get index from either places
      selected_frames = all_data.iloc[selected_indices]
      matched_rows=all_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
    # Save the selected frames to a CSV file
      matched_rows.to_csv(selected_indices_file)

    elif method == 'uncertainity sampling':
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
      all_data_collect_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2],index_col=0)
      #prev_output_dir
      for file in os.listdir(folder_path):
          file_path = os.path.join(prev_output_dir, file)
          if file_path.endswith("predictions_new.csv"):
            all_data=pd.read_csv(file_path, header=[0,1,2],index_col=0)

      all_data['sum'] = all_data.iloc[:, [4,8,12,16]].sum(axis=1)

    # Select the top 10 rows with the smallest sum
      selected_frames = all_data.nsmallest(num_frames, 'sum')

      all_data=all_data.drop('sum', axis=1)

      selected_frames = selected_frames.drop('sum', axis=1)

      selected_frames.set_index(('scorer', 'bodyparts', 'coords'), inplace=True)

      all_data_collect_data.set_index(('scorer', 'bodyparts', 'coords'), inplace=True)

      matched_rows=all_data_collect_data.loc[selected_frames.index]

      '''
      all_data_collect_data=all_data_collect_data.reset_index()
      matched_rows=matched_rows.reset_index()

      matched_rows.to_csv("test_match.csv")
      '''
      selected_indices_file = f'iteration_{method}_indices.csv'

      selected_indices_file = os.path.join(output_dir, selected_indices_file)

    # Save the selected frames to a CSV file
      matched_rows.to_csv(selected_indices_file)

    elif method == 'Ensembling':
      
      all_data_collect_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2],index_col=0) #prediction_new.csv

      folder_path = prev_output_dir
      pd_file={}
      finger={}
      num_finger_coord=num_keypoints*2
      for i in range(num_finger_coord):
        finger[str(i)]=pd.DataFrame()
      seed=0

      while seed<len(seeds_list):
        for file in os.listdir(folder_path):
              file_path = os.path.join(folder_path, file)
              if file_path.endswith("predictions_new_seed_"+str(seed)+".csv"): #change#########
                pd_file[str(seed)]=pd.read_csv(file_path, header=[0,1,2], index_col=0)#file_path
                seed+=1

      column_num=[0,1,3,4,6,7,9,10]
      for i in range(len(seeds_list)):
        for finger_num in range(num_finger_coord):
          finger[str(finger_num)]=pd.concat([finger[str(finger_num)],pd_file[str(i)].iloc[:,[column_num[finger_num]]]],axis=1)

      for i in range(8):
        finger[str(i)]=finger[str(i)].var(axis=1)

      combined_df = pd.concat([finger[str(i)] for i in range(8)], axis=1)
      combined_df=combined_df.sum(axis=1)
      var_df=pd.read_csv(file_path, header=[0,1,2], index_col=0)
      var_df = var_df.reindex(combined_df.index)
      var_df["var"]=pd.DataFrame(combined_df)
      selected_frames = var_df.nlargest(num_frames, "var")
      selected_frames=selected_frames.drop("var", axis=1)
      matched_rows=all_data_collect_data.loc[selected_frames.index]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
    # Save the selected frames to a CSV file
      matched_rows.to_csv(selected_indices_file)

    else:
      NotImplementedError(f'{method} is not implemented yet.')

    return selected_indices_file


def merge_collected_data(active_iter_cfg, selected_frames_file):
    """
    # Step 3: Merge new CollectedData.csv with the original CollectedData.csv
    # merge Collected_data.csv to include iteration_random_indices.csv
    # remove iteration_random_indices.csv from  {CollectedData}_new.csv

    :param active_iter_cfg:
    :return:
    """

    train_data_file = os.path.join(active_iter_cfg.train_data_file_prev_run)
    train_data = pd.read_csv(train_data_file, header=[0,1,2], index_col=0)
    # read selected frames
    selected_frames_df = pd.read_csv(selected_frames_file, header=[0,1,2], index_col=0)

    # concat train data and selected frames and merge
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
    iterations_folder = active_loop_cfg.active_loop.iterations_folder

    iteration_key = 'iteration_{}'.format(iteration_number)
    active_iter_cfg = active_loop_cfg[iteration_key]
    iteration_folder = os.path.abspath(str(
        Path(experiment_cfg.data.data_dir,
             experiment_cfg.data.csv_file).parent.absolute() / iterations_folder / iteration_key)
    )

    # read train and eval files
    train_data_file_prev_run = str(Path(experiment_cfg.data.data_dir,
                                        active_iter_cfg.csv_file_prev_run))
    eval_data_file_prev_run = train_data_file_prev_run.replace('.csv', '_new.csv')

    # update params to config file
    active_iter_cfg.iteration_key = iteration_key
    active_iter_cfg.iteration_prefix = '{}_{}'.format(active_iter_cfg.method,
                                                      active_iter_cfg.num_frames)
    active_iter_cfg.iteration_folder = iteration_folder
    active_iter_cfg.train_data_file_prev_run = train_data_file_prev_run
    active_iter_cfg.eval_data_file_prev_run = eval_data_file_prev_run
    active_iter_cfg.train_data_file = os.path.join(
        active_iter_cfg.iteration_folder,
        '{}_{}'.format(active_iter_cfg.iteration_prefix,
                       os.path.basename(train_data_file_prev_run))
    )
    active_iter_cfg.eval_data_file = active_iter_cfg.train_data_file.replace('.csv', '_new.csv')

    # Active Loop parameters
    # Step 1: Initialize the iteration folder
    #  TODO(haotianxiansti):  add code for iter 0 (select frames when no labeles are present)
    initialize_iteration_folder(active_iter_cfg.iteration_folder)

    num_keypoints=experiment_cfg.data.num_keypoints
    seeds_list=[0,1,2,3,4,5]
    prev_output_dir=active_iter_cfg.output_prev_run
    selected_frames_file = select_frames(active_iter_cfg,num_keypoints,seeds_list,prev_output_dir)

    # Now, we have in the directory:
    # created Collected_data_new_merged and Collected_data_merged.csv
    merge_collected_data(active_iter_cfg, selected_frames_file)
    # run algorithm with new config file
    # make relative to data_dir
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

    num_iterations = active_cfg.active_loop.end_iteration - active_cfg.active_loop.start_iteration + 1
    for current_iteration in range(active_cfg.active_loop.start_iteration,
                                   active_cfg.active_loop.end_iteration + 1):

        current_iteration=active_cfg.active_loop.current_iteration
        print('\n\n Experiment iter {}'.format(current_iteration), flush=True)

        if current_iteration == 0:
            # step 1: select frames to label is skipped in demo mode.
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration, 'baseline')

        # step 2: train model using exp_cfg
        train_output_dir = run_train(exp_cfg)

        # step 3: call active loop
        if current_iteration + 1 >  active_cfg.active_loop.end_iteration:
          break
        iteration_key = 'iteration_{}'.format(current_iteration + 1)
        active_cfg.active_loop.current_iteration = current_iteration
        active_cfg[iteration_key].output_prev_run = train_output_dir #need to uncomment
        active_cfg[iteration_key].csv_file_prev_run = exp_cfg.data.csv_file
        #print('\n\nActive loop config after iter {}'.format(current_iteration), active_cfg, flush=True)
        new_train_file = active_loop_step(active_cfg)

        # update config file
        exp_cfg.data.csv_file = new_train_file
        exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration + 1,
                                                       active_cfg[iteration_key].method)

    # write new active_cfg file
    return active_cfg


def run_train(cfg):
    sys.path.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
    import train_hydra
    cwd = os.getcwd()
    today_str = datetime.now().strftime("%y-%m-%d")
    ctime_str = datetime.now().strftime("%H-%M-%S")
    new_dir = f"./outputs/{today_str}/{ctime_str}"
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