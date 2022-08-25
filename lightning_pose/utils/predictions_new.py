import torch
import hydra 
from omegaconf import DictConfig
import numpy as np
import os
import pandas as pd
from typing import List, Tuple, Union
from torchtyping import TensorType
from lightning_pose.data.utils import count_frames
import pytorch_lightning as pl


class PredictionHandler:
    def __init__(self, cfg: DictConfig, data_module: pl.LightningDataModule, video_file: Union[str, None]) -> None:
        self.cfg = cfg
        self.data_module = data_module
        self.video_file = video_file
        if video_file is not None:
            assert os.path.isfile(video_file)
    
    @property
    def frame_count(self) -> int:
        """Returns the number of frames in the video or the labeled dataset"""
        if self.video_file is not None:
            return count_frames(self.video_file)
        else:
            return len(self.data_module.dataset)
    
    @property
    def keypoint_names(self):
        return self.data_module.dataset.keypoint_names
    
    @property
    def do_context(self):
        return self.data_module.dataset.do_context

    def unpack_preds(
        self,
        preds: List[Tuple[TensorType["batch", "two_times_num_keypoints"],
                          TensorType["batch", "num_keypoints"]]]
    ) -> Tuple[
            TensorType["num_frames", "two_times_num_keypoints"],
            TensorType["num_frames", "num_keypoints"]
    ]:
        """ unpack list of preds coming out from pl.trainer.predict, confs tuples into tensors.
        It still returns unnecessary final rows, which should be discarded at the dataframe stage.
        This works for the output of predict_loader, suitable for batch_size=1, sequence_length=16, step=16"""
        # stack the predictions into rows.
        # loop over the batches, and stack 
        stacked_preds = torch.vstack([pred[0] for pred in preds])
        stacked_confs = torch.vstack([pred[1] for pred in preds])

        if self.video_file is not None: # dealing with dali loaders
            if self.do_context == False:
                # in this dataloader, the last sequence has a few extra frames.
                num_rows_to_discard = stacked_preds.shape[0] - self.frame_count
                if num_rows_to_discard > 0:
                    stacked_preds = stacked_preds[:-num_rows_to_discard]
                    stacked_confs = stacked_confs[:-num_rows_to_discard]
            
            if self.do_context == True:
                # fix shifts in the context model
                stacked_preds = self.fix_context_preds_confs(stacked_preds)
                stacked_confs = self.fix_context_preds_confs(stacked_confs, is_confidence=True)

        return stacked_preds, stacked_confs
    
    def fix_context_preds_confs(self, stacked_preds: TensorType, is_confidence: bool = False):
        """
        In the context model, ind=0 is associated with image[2], and ind=1 is associated with image[3],
        so we need to shift the predictions and confidences by two and eliminate the edges.
        NOTE: confidences are not zero in the first and last two images, they are instead replicas of images[-2] and images[-3]
        """
        # first pad the first two rows for which we have no valid preds.
        preds_1 = torch.tile(stacked_preds[0], (2,1)) # copying twice the prediction for image[2]
        preds_2 = stacked_preds[0:-2] # throw out the last two rows.
        preds_combined = torch.vstack([preds_1, preds_2])
        # after concat this has the same length. everything is shifted by two rows. 
        # but we don't have valid predictions for the last two elements, so we pad with element -3.
        preds_combined[-2:, :] = preds_combined[-3, :]

        if is_confidence == True:
            # zeroing out those first and last two rows (after we've shifted everything above)
            preds_combined[:2, :] = 0.0
            preds_combined[-2:, :] = 0.0
        
        return preds_combined

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
    
    def add_split_indices_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add split indices to the dataframe.
        """
        df["set"] = np.array(["unused"] * df.shape[0])
        
        dataset_split_indices = {
            "train": self.data_module.train_dataset.indices,
            "validation": self.data_module.val_dataset.indices,
            "test": self.data_module.test_dataset.indices
        }
        
        for key, val in dataset_split_indices.items():
            df.loc[val, "set"] = np.repeat(key, len(val))
        return df

    def __call__(
        self,
        preds: List[Tuple[TensorType["batch", "two_times_num_keypoints"],
                          TensorType["batch", "num_keypoints"]]]
    )-> pd.DataFrame:
        """
        Call this function to get a pandas dataframe of the predictions for a single video.
        Assuming you've already run trainer.predict(), and have a list of Tuple predictions.
        Args:
            preds: list of tuples of (predictions, confidences)
            video_file: path to video file
        Returns:
            pd.DataFrame: index is (frame, bodypart, x, y, likelihood)
        """
        stacked_preds, stacked_confs = self.unpack_preds(preds=preds)
        pred_arr = self.make_pred_arr_undo_resize(
            stacked_preds.cpu().numpy(), stacked_confs.cpu().numpy())
        pdindex = self.make_dlc_pandas_index()
        df = pd.DataFrame(pred_arr, columns=pdindex)
        if self.video_file is None:
            # specify which image is train/test/val/unused
            df = self.add_split_indices_to_df(df)
            df.index = self.data_module.dataset.image_names

        return df
