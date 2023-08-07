from pathlib import Path
import os
from omegaconf import OmegaConf, open_dict
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np


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
    xy_keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)[..., :-1]
    var=keypoints.reshape(keypoints.shape[0], -1, 3)[..., -1:]
    all_var.append(var)
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
  var_total=np.concatenate((variance_xy,var_likely),-1)
  var_total=var_total.reshape(-1,12) #### might cause some issue
  csv_data_var.iloc[:,:-1]=var_total
  selected_indices = np.argsort(variance_xy.sum(-1).sum(-1))[:num_frames]

  matched_rows =  csv_data.iloc[selected_indices]

  return matched_rows, csv_data_mean, csv_data_var

def find_common_elements(*lists):
  # Convert each list to sets and find the intersection of all sets
  common_elements = set(lists[0]).intersection(*lists[1:])
  return list(common_elements)


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

    xy_keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)[..., :-1]
    var=keypoints.reshape(keypoints.shape[0], -1, 3)[..., -1:]
    all_var.append(var)

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

  var_total=np.concatenate((variance_xy,var_likely),-1)

  var_total=var_total.reshape(-1,12) #### might cause some issue
  csv_data_var.iloc[:,:-1]=var_total
  selected_indices = np.argsort(variance_xy.sum(-1).sum(-1))[:num_frames]

  matched_rows =  csv_data.iloc[selected_indices]

  return matched_rows, csv_data_mean, csv_data_var


def low_energy_random_sampling(energy_func, all_data, num_frames):
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
  return x ** 2


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
  train_data = pd.read_csv(train_data_file, header=[0, 1, 2], index_col=0)

  # read active test frames:
  act_test_data_file = train_data_file.replace(".csv", "_active_test.csv")  ### New Add
  act_test_data = pd.read_csv(act_test_data_file, header=[0, 1, 2], index_col=0)
  act_test_data.to_csv(active_iter_cfg.act_test_data_file)

  # read selected frames
  selected_frames_df = pd.read_csv(selected_frames_file, header=[0, 1, 2], index_col=0)

  # concat train data and selected frames and merge
  # TODO: check relative to the data path.
  new_train_data = pd.concat([train_data, selected_frames_df])
  new_train_data.to_csv(active_iter_cfg.train_data_file)

  # remove selected_frames from val data
  val_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
  val_data.drop(index=selected_frames_df.index, inplace=True)
  val_data.to_csv(active_iter_cfg.eval_data_file)
  return

def Histogram_Plot(method, data, output_dir):
  data.plot(kind='hist', bins=20)
  plot_title= f'Histogram of {method}'
  plt.title(plot_title)
  xtitle=f'{method}'
  plt.xlabel(xtitle)
  plt.ylabel('Number of Frames')
  plot_title= f'Histogram of {method}'
  plt_path = os.path.join(output_dir, plot_title)
  plt.savefig(plt_path)
  plt.show()
  plt.cla()

def select_common_frames(prev_output_dirs,predict_new_name, predict_active_name, header_rows=[0,1,2]):

  all_indices = []
  for run_idx, folder_path in enumerate(prev_output_dirs):
    csv_file = os.path.join(folder_path, predict_new_name)
    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    all_indices.append(csv_data.index.values)
  common_elements = np.asarray(find_common_elements(*all_indices))

  return common_elements

def subsample_frames_from_df(labels_df, train_frames, train_prob, rng_seed, vid_dir=None):

    # find all possible videos
    img_files = labels_df.index
    vid_names = np.unique([c.split('/')[1] for c in img_files.to_numpy()])


    # select some videos to sample frames from
    n_total_frames = int(np.ceil(int(train_frames) * 1.0 / train_prob))

    # grab frames from selected vids until we reach total number of desired frames
    n_frames = 0
    n_vids_to_select = 1
    new_df = None
    vids = None
    while n_frames < int(n_total_frames):
        if n_vids_to_select >= vid_names.shape[0]:
            new_df = labels_df.copy()
            vids = vid_names
            break
        np.random.seed(int(rng_seed))
        vids = np.random.choice(vid_names, size=n_vids_to_select)
        good_idxs = labels_df.index.str.contains('|'.join(vids))
        new_df = labels_df[good_idxs]
        n_frames = new_df.shape[0]
        n_vids_to_select += 1

    return new_df, vids

def select_frames_calculate(active_iter_cfg, data_cfg, header_rows=[0,1,2]): #This now step 2

  all_indices = []

  method = active_iter_cfg.method
  num_frames = active_iter_cfg.num_frames
  output_dir = active_iter_cfg.iteration_folder
  prev_output_dirs = active_iter_cfg.output_prev_run
  
  all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)

  if method == 'margin sampling':

    predict_new_name = "predictions_new_heatmap.csv" 
    predict_active_name = "predictions_active_test_heatmap.csv"

  elif method == "Single PCAS":
    predict_new_name = "predictions_new_pca_singleview_error.csv"
    predict_active_name = "predictions_active_test_pca_singleview_error.csv"
  
  elif method == 'Equal Variance':
    predict_new_name = "predictions_new_equalvariance.csv"
    predict_active_name = "predictions_active_test_equalvariance.csv"

  else:
    predict_new_name = "predictions_new.csv" 
    predict_active_name = "predictions_active_test.csv"

  common_elements =select_common_frames(prev_output_dirs, 
    predict_new_name, predict_active_name, header_rows=[0,1,2])

  # read xy keypoints from each csv file
  if method == 'Ensembling':
    all_keypoints = []
    all_np=list()
    all_var=list()

  selected_frames_list=list()

  for run_idx, folder_path in enumerate(prev_output_dirs):
    csv_file = os.path.join(folder_path, predict_new_name)
    csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
    all_indices.append(csv_data.index.values)
    # filter by common elements
    csv_data = csv_data.loc[common_elements]
    csv_active_test_file=os.path.join(folder_path, predict_active_name)
    if os.path.isfile(csv_active_test_file):
      csv_active_test_data=pd.read_csv(csv_active_test_file, header=header_rows, index_col=0)
      active_test_data_flag=True

    else:
      csv_active_test_data=None
      active_test_data_flag=False

    sep_flag=True

    selected_frames = select_frames(active_iter_cfg, data_cfg, csv_data, csv_active_test_data, active_test_data_flag)
    selected_frames_list.append(selected_frames)
    matched_rows = all_data.loc[selected_frames.index]

    

    if method != "Ensembling" and len(prev_output_dirs) > 1 :
      selected_indices_file_individual =  f'iteration_{method}_{folder_path[-8:]}_indices.csv'
      selected_indices_file_individual = os.path.join(output_dir, selected_indices_file_individual)
      matched_rows.to_csv(selected_indices_file_individual)
  
  if method != "Ensembling" and len(prev_output_dirs) > 1 :

    selected_frame_total_df = pd.concat(selected_frames_list, axis=0)
    selected_frames = select_frames(active_iter_cfg, data_cfg, selected_frame_total_df, csv_active_test_data=None, active_test_data_flag=False)
    matched_rows = all_data.loc[selected_frames.index]

  


  selected_indices_file = f'iteration_{method}_indices.csv'
  selected_indices_file = os.path.join(output_dir, selected_indices_file)
  matched_rows.to_csv(selected_indices_file)


  return selected_indices_file

def select_frames(active_iter_cfg, data_cfg, csv_data, csv_active_test_data, active_test_data_flag):
  """
  Step 2: select frames to label
  Implement the logic for selecting frames based on the specified method:
  :param active_iter_cfg: active loop config file
  :return:
    : selected_indices_file: csv file with selected frames from active loop.
  """
  # TODO():may not want to overwrite file
  method = active_iter_cfg.method
  num_frames = active_iter_cfg.num_frames
  output_dir = active_iter_cfg.iteration_folder
  prev_output_dirs = active_iter_cfg.output_prev_run  # list of directories of previous runs

  # We need to know the index of our selected data (list)
  if method == 'random':
    # select random frames from eval data in prev run.

    selected_indices = np.unique(random.sample(range(len(csv_data)), num_frames))  # Get index from either places
    selected_frames = csv_data.iloc[selected_indices]

  if method == "random vid limit":

    df, _ = subsample_frames_from_df(csv_data, 1, 0.1, 0, vid_dir=None)
    selected_indices= df.index
    selected_frames = csv_data.loc[selected_indices]
  

  elif method == 'random_energy':
    # TODO
    # select random frames from eval data in prev run.
    all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
    selected_indices = low_energy_random_sampling(energy_function, all_data,
                                                  num_frames)  # np.unique(random.sample(range(len(all_data)), num_frames))  # Get index from either places
    selected_frames = all_data.iloc[selected_indices]
    matched_rows = all_data.loc[selected_frames.index]
    selected_indices_file = f'iteration_{method}_indices.csv'
    selected_indices_file = os.path.join(output_dir, selected_indices_file)
    # Save the selected frames to a CSV file
    matched_rows.to_csv(selected_indices_file)

  elif method == 'uncertainty_sampling':
    # TODO:
    margin=csv_data
    margin_numpy = margin.to_numpy()[..., :-1]
    uncertainty = margin_numpy.reshape(margin_numpy.shape[0], -1, 3)[..., -1]
    margin["Uncertainity"] = uncertainty.sum(-1).astype(float)
    margin["Uncertainity"].plot(kind='hist', bins=20)
    Histogram_Plot(method, margin["Uncertainity"], output_dir)

    if active_test_data_flag == True:
      active_test_margin = csv_active_test_data #pd.read_csv(csv_active_test_file, header=[0, 1, 2], index_col=0)
      active_test_margin_numpy = active_test_margin.to_numpy()[..., :-1]
      uncertainty = active_test_margin_numpy.reshape(active_test_margin_numpy.shape[0], -1, 3)[..., -1]
      active_test_margin["Uncertainity"] = uncertainty.sum(-1).astype(float)

      Histogram_Plot(method, active_test_margin["Uncertainity"], output_dir)

    selected_frames = margin.nsmallest(num_frames, "Uncertainity")
    selected_frames = selected_frames.drop("Uncertainity", axis=1)
    
  elif method == 'margin sampling':
    # TODO:
    margin = csv_data
    margin['sum'] = margin.sum(axis=1)
    Histogram_Plot(method, margin['sum'], output_dir)

    if active_test_data_flag == True:
      active_test_margin = csv_active_test_data
      active_test_margin['sum'] = active_test_margin.sum(axis=1)
      Histogram_Plot(method, margin['sum'], output_dir)

    selected_frames = margin.nsmallest(num_frames, 'sum')
    selected_frames = selected_frames.drop('sum', axis=1)

  elif method == 'Ensembling':
    all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0, 1, 2], index_col=0)
    matched_rows, csv_data_mean, csv_data_var = calculate_ensemble_frames(prev_output_dirs, num_frames)

    if active_test_data_flag == True:
      _, csv_data_mean, csv_data_var = calculate_ensemble_frames_for_active_test(prev_output_dirs, num_frames)
      csv_mean_file = os.path.join(prev_output_dirs[-1], "predictions_active_test.csv")
      csv_data_mean.to_csv(csv_mean_file)
      csv_var_file = os.path.join(prev_output_dirs[-1], "predictions_var_active_test.csv")
      csv_data_var.drop(csv_data_var.columns[-1], axis=1, inplace=True)
      csv_data_var.drop(csv_data_var.columns[2::3], axis=1, inplace=True)
      csv_data_var.to_csv(csv_var_file)
      csv_data_var['sum'] = csv_data_var.sum(axis=1)
      Histogram_Plot(method, csv_data_var['sum'], output_dir)

    else:
      csv_mean_file = os.path.join(prev_output_dirs[-1], "predictions_new.csv")
      csv_data_mean.to_csv(csv_mean_file)
      csv_var_file = os.path.join(prev_output_dirs[-1], "predictions_var_new.csv")
      csv_data_var.drop(csv_data_var.columns[-1], axis=1, inplace=True)
      csv_data_var.drop(csv_data_var.columns[2::3], axis=1, inplace=True)
      csv_data_var.to_csv(csv_var_file)
      csv_data_var['sum'] = csv_data_var.sum(axis=1)
      Histogram_Plot(method, csv_data_var['sum'], output_dir)


    matched_rows = all_data.loc[matched_rows.index]
    selected_frames = matched_rows
    selected_indices_file = f'iteration_{method}_indices.csv'
    selected_indices_file = os.path.join(output_dir, selected_indices_file)
    # Save the selected frames to a CSV file
    matched_rows.to_csv(selected_indices_file)

  elif method == "Single PCAS":

    single_pca = csv_data
    single_pca['sum'] = single_pca.iloc[:, [0, 1]].sum(axis=1)
    Histogram_Plot(method, single_pca['sum'], output_dir)
    if active_test_data_flag == True:
       single_pca_active_test = csv_active_test_data
       single_pca_active_test['sum'] = single_pca_active_test.iloc[:, [0, 1]].sum(axis=1)
       Histogram_Plot(method, single_pca_active_test['sum'], output_dir)
    selected_frames = single_pca.nlargest(num_frames, 'sum')
    selected_frames = selected_frames.drop('sum', axis=1)


  elif method == 'Equal Variance':
    # TODO:
    
    margin = csv_data
    margin['sum'] = margin.sum(axis=1)
    Histogram_Plot(method, margin['sum'], output_dir)
    if active_test_data_flag == True:
      active_test_margin = csv_active_test_data
      active_test_margin['sum'] = active_test_margin.sum(axis=1)
      Histogram_Plot(method, active_test_margin['sum'], output_dir)
    selected_frames = margin.nsmallest(num_frames, 'sum')
    selected_frames = selected_frames.drop('sum', axis=1)

  else:
    NotImplementedError(f'{method} is not implemented yet.')

  return selected_frames


def initialize_iteration_folder(data_dir):
  """ Initialize the iteration folder
  :param data_dir: where the iteration folder will be created.
  clone
  """
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def active_loop_step(active_loop_cfg):
    """
    TODO(haotianxiansti) update comments
    TODO(haotianxiansti) clean up code
    # Step 6: Launch the next active_loop iteration
    """
    # read yaml file
    experiment_cfg = OmegaConf.load(active_loop_cfg.active_pipeline.experiment_cfg)

    # read params for current active loop iteration
    iteration_number = active_loop_cfg.active_pipeline.current_iteration
    iterations_folders = active_loop_cfg.active_pipeline.iterations_folders

    iteration_key = 'iteration_{}'.format(iteration_number)
    active_iter_cfg = active_loop_cfg[iteration_key]

    dir_collected_csv = Path(experiment_cfg.data.data_dir, experiment_cfg.data.csv_file).parent.absolute()
    ####iterations_folder -> iteration_folders
    iteration_folder = os.path.abspath(str(dir_collected_csv / active_loop_cfg.active_pipeline.iterations_folders / iteration_key))

    # Read train and eval files
    # might need to reconsider making these as lists
    train_data_file_prev_run = str(Path(experiment_cfg.data.data_dir,
                                        active_iter_cfg.csv_file_prev_run))
    eval_data_file_prev_run = train_data_file_prev_run.replace('.csv', '_new.csv')
    act_test_data_file_prev_run = train_data_file_prev_run.replace('.csv', '_active_test.csv') ### New add

    # update active loop params to config file
    # add additional keys:
    OmegaConf.set_struct(active_iter_cfg, True)
    with open_dict(active_iter_cfg):
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

    # Active Loop: Initialize the iteration folder
    #  TODO(haotianxiansti):  add code for iter 0 (select frames when no labeles are present)
    initialize_iteration_folder(active_iter_cfg.iteration_folder)

    selected_frames_file = select_frames_calculate(active_iter_cfg, experiment_cfg.data, header_rows=[0,1,2])

    # Now, we have the directory:
    # created Collected_data_new_merged and Collected_data_merged.csv
    merge_collected_data(active_iter_cfg, selected_frames_file)
    # run algorithm with new config file
    # TODO: check location of new csv for iteration relative to data_dir
    # it should have CollectedData.csv and CollectedData_new.csv
    relpath = os.path.relpath(active_iter_cfg.train_data_file, experiment_cfg.data.data_dir)

    return relpath


def update_config_yaml(config_file, merged_data_dir):
  """
  # Step 4: Update the config.yaml file
  :param config_file:
  :param merged_data_dir:
  :return:
  """
  # Load the config file
  cfg = OmegaConf.load(config_file)

  OmegaConf.update(cfg, 'data.csv_file', merged_data_dir, merge=True)

  OmegaConf.save(cfg, config_file)

