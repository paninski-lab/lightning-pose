import unittest
import pandas as pd
from active_utils import active_loop_step, get_vids, subsample_frames_from_df

class TestSubsampleFrames(unittest.TestCase):

    def test_sample_videos(self):

        labels_df = pd.read_csv("/content/drive/MyDrive/crim13/CollectedData_new.csv",header=[0,1,2],index_col=0) # here put active_learning pool

        num_vids = 5  # number of vids
        train_frames = 50  # number of frames = train_frames / train_prob
        train_prob = 0.1  
        rng_seed = 42  # random seed
        used_vids = []  # void list
        iter0_flag = True  # flag for first iteration

        # get sampling functions
        new_df, used_vids = subsample_frames_from_df(
            labels_df, num_vids, train_frames, train_prob, rng_seed, used_vids, iter0_flag
        )

        # do we select 5 vids?
        self.assertEqual(len(used_vids), 5)

        # do we have correct frames
        expected_total_frames = int(train_frames / train_prob)
        self.assertEqual(new_df.shape[0], expected_total_frames)

        # get sampling functions
        new_df, used_vids = subsample_frames_from_df(
            labels_df, num_vids, train_frames, train_prob, rng_seed, used_vids, iter0_flag = False
        )

        # do we select 5 vids?
        self.assertEqual(len(used_vids), 10)

if __name__ == '__main__':
    unittest.main()
