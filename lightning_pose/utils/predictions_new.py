import torch
import hydra 
from omegaconf import DictConfig
import numpy as np
import os
import pandas as pd
from typing import List, Tuple
from torchtyping import TensorType
from lightning_pose.data.utils import count_frames
import pytorch_lightning as pl


class PredictionHandler:
    def __init__(self, cfg: DictConfig, data_module: pl.LightningDataModule) -> None:
        self.cfg = cfg
        self.data_module = data_module
    
    @property
    def keypoint_names(self):
        return self.data_module.dataset.keypoint_names
    
    @property
    def do_context(self):
        return self.data_module.dataset.do_context

    def unpack_preds(self, preds: List[Tuple[TensorType["batch", "two_times_num_keypoints"], TensorType["batch", "num_keypoints"]]], frame_count: int) -> Tuple[TensorType["num_frames", "two_times_num_keypoints"], TensorType["num_frames", "num_keypoints"]]:
        """ unpack list of preds coming out from pl.trainer.predict, confs tuples into tensors.
        It still returns unnecessary final rows, which should be discarded at the dataframe stage.
        This works for the output of predict_loader, suitable for batch_size=1, sequence_length=16, step=16"""
        # stack the predictions into rows.
        # loop over the batches, and stack 
        stacked_preds = torch.vstack([pred[0] for pred in preds])
        stacked_confs = torch.vstack([pred[1] for pred in preds])
        # eliminate last rows
        # this is true just for the case of e.g., batch_size=1, sequence_length=16, step=sequence_length
        # the context dataloader just doesn't include those extra frames.
        # should be == 0 for the context model. it discards unnecessary batches.
        if self.do_context == False:
            # in this dataloader, the last sequence has a few extra frames.
            num_rows_to_discard = stacked_preds.shape[0] - frame_count
            if num_rows_to_discard > 0:
                stacked_preds = stacked_preds[:-num_rows_to_discard]
                stacked_confs = stacked_confs[:-num_rows_to_discard]
        
        if self.do_context == True:
            # make first two rows and last two rows with confidence 0.0.
            stacked_confs[:2, :] = 0.0
            stacked_confs[-2:, :] = 0.0
            # fill the predicted values with row 2 and row -2. 
            stacked_preds[:2, :] = stacked_preds[2, :]
            stacked_preds[-2:, :] = stacked_preds[-3, :]

        return stacked_preds, stacked_confs
    
    def make_pred_arr_undo_resize(
        self,
        keypoints_np: np.array,
        confidence_np: np.array,
    ) -> np.array:
        """Resize keypoints and add confidences into one numpy array.

        Args:
            keypoints_np: shape (n_frames, n_keypoints * 2)
            confidence_np: shape (n_frames, n_keypoints)

        Returns:
            np.ndarray: cols are (bp0_x, bp0_y, bp0_likelihood, bp1_x, bp1_y, ...)

        """
        assert keypoints_np.shape[0] == confidence_np.shape[0]  # num frames in the dataset
        assert keypoints_np.shape[1] == (
            confidence_np.shape[1] * 2
        )  # we have two (x,y) coordinates and a single likelihood value

        num_joints = confidence_np.shape[-1]  # model.num_keypoints
        predictions = np.zeros((keypoints_np.shape[0], num_joints * 3))
        predictions[:, 0] = np.arange(keypoints_np.shape[0])
        # put x vals back in original pixel space
        x_resize = self.cfg.data.image_resize_dims.width
        x_og = self.cfg.data.image_orig_dims.width
        predictions[:, 0::3] = keypoints_np[:, 0::2] / x_resize * x_og
        # put y vals back in original pixel space
        y_resize = self.cfg.data.image_resize_dims.height
        y_og = self.cfg.data.image_orig_dims.height
        predictions[:, 1::3] = keypoints_np[:, 1::2] / y_resize * y_og
        predictions[:, 2::3] = confidence_np

        return predictions
    
    def make_dlc_pandas_index(self) -> pd.MultiIndex:
        xyl_labels = ["x", "y", "likelihood"]
        pdindex = pd.MultiIndex.from_product(
            [["%s_tracker" % self.cfg.model.model_type], self.keypoint_names, xyl_labels],
            names=["scorer", "bodyparts", "coords"],
        )
        return pdindex
    
    def __call__(self, video_file: str, preds: List[Tuple[TensorType["batch", "two_times_num_keypoints"], TensorType["batch", "num_keypoints"]]])-> pd.DataFrame:
        """
        Call this function to get a pandas dataframe of the predictions for a single video.
        Assuming you've already run trainer.predict(), and have a list of Tuple predictions.
        Args:
            preds: list of tuples of (predictions, confidences)
            video_file: path to video file
        Returns:
            pd.DataFrame: index is (frame, bodypart, x, y, likelihood)
        """
        frame_count = count_frames(video_file)
        stacked_preds, stacked_confs = self.unpack_preds(preds=preds, frame_count=frame_count)
        pred_arr = self.make_pred_arr_undo_resize(stacked_preds.numpy(), stacked_confs.numpy())
        pdindex = self.make_dlc_pandas_index()
        df = pd.DataFrame(pred_arr, columns=pdindex)
        return df
