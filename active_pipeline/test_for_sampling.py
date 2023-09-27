import unittest
import pandas as pd

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from call_active_loop import call_active_all
from active_utils import active_loop_step, get_vids, subsample_frames_from_df, select_frames_calculate


def create_cfg(path) -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(path,"r"), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


class TestSubsampleFrames(unittest.TestCase):

    def test_sample_videos(self):

        labels_df = pd.read_csv("/content/drive/MyDrive/crim13/CollectedData_active_test.csv",header=[0,1,2],index_col=0) # here put active_learning pool

        num_vids = 5  # number of vids
        train_frames = 1  # number of frames = train_frames / train_prob
        train_prob = 0.1  
        rng_seed = 42  # random seed
        used_vids = []  # void list
        iter0_flag = True  # flag for first iteration

        # get sampling functions
        new_df, used_vids = subsample_frames_from_df(
            labels_df, num_vids, train_frames, train_prob, rng_seed, used_vids, iter0_flag
        ) # number of frames = train_frames / train_prob

        # do we select 5 vids? In the first run
        self.assertEqual(len(used_vids), num_vids) 

        self.assertEqual(len(new_df), train_frames / train_prob) #For iteration 0, it should be number of frames = train_frames / train_prob

        # do we have correct frames
        expected_total_frames = int(train_frames / train_prob)
        self.assertEqual(new_df.shape[0], expected_total_frames)

        # get sampling functions
        new_df, used_vids = subsample_frames_from_df(
            labels_df, num_vids, train_frames, train_prob, rng_seed, used_vids, iter0_flag = False
        )

        # do we select 5 vids?
        self.assertEqual(len(used_vids), num_vids*2)
      
    def test_selected_frames(self):
       
        num_vids = 4 # in the dummyz csv we only have 4 vids

        cfg = create_cfg("/content/lightning-pose/active_pipeline/configs/config_ibl_active.yaml")
        cfg.active_pipeline.start_iteration = 0
        cfg.active_pipeline.current_iteration = 0
        cfg.active_pipeline.end_iteration = 0
        new_active_cfg = call_active_all(cfg)
        _, used_vids = active_loop_step(new_active_cfg, list())
        new_active_cfg['iteration_0'].num_frames = 10
        new_active_cfg['iteration_0'].train_frames = 1 #num_frames = train_frames / train_prob (i.e. 50 frames = 5/0.1)
        new_active_cfg['iteration_0'].train_prob = 0.1 
        self.assertEqual(len(used_vids), num_vids)
        experiment_cfg = OmegaConf.load(new_active_cfg.active_pipeline.experiment_cfg)
        selected_frames_file, active_test_data_flag, used_vids = select_frames_calculate(new_active_cfg['iteration_0'], experiment_cfg.data, used_vids)
        new_df = pd.read_csv(selected_frames_file, header=[0,1,2], index_col = 0)
        self.assertEqual(len(used_vids), num_vids)
        self.assertEqual(len(new_df), new_active_cfg['iteration_0'].num_frames) #should be the same with how many frames we sampled

if __name__ == '__main__':
    unittest.main()
