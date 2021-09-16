import torch
import pandas as pd
from torch import cuda
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from typing import Callable, Optional, Tuple, List
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from pose_est_nets.utils.heatmap_tracker_utils import format_mouse_data
from pose_est_nets.utils.dataset_utils import draw_keypoints
from pose_est_nets.utils.datamod_utils import video_pipe, LightningWrapper, PCA_prints
import h5py
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typeguard import typechecked
import sklearn

#Maybe make torch manual seed a global variable?
TORCH_MANUAL_SEED = 42
_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
_SEQUENCE_LENGTH_UNSUPERVISED = 7
_INITIAL_PREFETCH_SIZE = 16
_BATCH_SIZE_UNSUPERVISED = 1  # sequence_length * batch_size = num_images passed
_DALI_RANDOM_SEED = 123456


class TrackingDataModule(pl.LightningDataModule):
    def __init__(  # TODO: add documentation and args
        self,
        dataset,
        mode,
        train_batch_size,
        validation_batch_size,
        test_batch_size,
        num_workers: Optional[int] = 8,
    ):
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # maybe can make the view information more general when deciding on a specific format for csv files
        self.num_views = 2  # changes with dataset, 2 for mouse, 3 for fish
        self.mode = mode
        

    def setup(self, stage: Optional[str] = None):  # TODO: clean up
        print("Setting up DataModule...")
        datalen = self.fulldataset.__len__()
        print(
            "Number of labeled images in the full dataset (train+val+test): {}".format(
                datalen
            )
        )

        if self.mode == "deterministic":
            return

        train_size = round(datalen * 0.8)
        valid_size = round(datalen * 0.1)
        test_size = datalen - (train_size + val_size)
        self.train_set, self.valid_set, self.test_set = random_split(
            self.fulldataset,
            [
                train_size,
                valid_size,
                test_size,
            ],  
            generator=torch.Generator().manual_seed(TORCH_MANUAL_SEED),
        )

        print(
            "Size of -- train set: {}, validation set: {}, test set: {}".format(
                len(self.train_set), len(self.valid_set), len(self.test_set)
            )
        )

class UnlabledDataModule(TrackingDataModule):
    def __init__(  # TODO: add documentation and args
        self,
        dataset,
        mode,
        train_batch_size,
        validation_batch_size,
        test_batch_size,
        num_workers: Optional[int] = 8,
        unlabeled_video_path: Optional[str] = None,
    ):
        super().__init__(dataset, mode, train_batch_size, validation_batch_sizem test_batch_size, num_workers)
        self.unlabeled_video_path = unlabeled_video_path

    def setup_unlabeled(self, video_path):
        # device_id = self.local_rank
        # shard_id = self.global_rank
        # num_shards = self.trainer.world_size
        data_pipe = video_pipe(
            batch_size=_BATCH_SIZE_UNSUPERVISED,
            num_threads=self.num_workers
            // 2,  # because the other workers do the labeled dataloading
            device_id=0,  # TODO: be careful when scaling to multinode
            resize_dims=[self.fulldataset.height, self.fulldataset.width],
            random_shuffle=True,
            # shard_id=shard_id,
            # num_shards=num_shards,
            filenames=video_files,
            seed=_DALI_RANDOM_SEED,
        )

        self.semi_supervised_loader = LightningWrapper(
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,  # TODO: verify that
        )  # changed output_map to account for dummy labels

    # TODO: could be separated from this class
    # TODO: return something?
    def computePCA_params(
        self,
        components_to_keep: Optional[int] = 3,
        empirical_epsilon_percentile: Optional[float] = 90.0,
    ) -> None:
        print("Computing PCA on the labels...")
        param_dict = {}
        # TODO: I don't follow the ifs, clarify with Nick
        if type(self.train_set) == torch.utils.data.dataset.Subset:
            indxs = torch.tensor(self.train_set.indices)
            data_arr = torch.index_select(self.train_set.dataset.labels, 0, indxs)
            num_body_parts = self.train_set.dataset.num_targets
        else:
            data_arr = self.train_set.labels  # won't work for random splitting
            num_body_parts = self.train_set.num_targets
        arr_for_pca = format_mouse_data(data_arr)
        pca = PCA(n_components=4, svd_solver="full")
        pca.fit(arr_for_pca.T)
        print("Done!")

        print(
            "arr_for_pca shape: {}".format(arr_for_pca.shape)
        )  # TODO: have prints as tests
        PCA_prints(pca, components_to_keep)  # print important params
        # mu = torch.mean(arr_for_pca, axis=1) # TODO: needed only for probabilistic version
        # param_dict["obs_offset"] = mu  # TODO: needed only for probabilistic version
        param_dict["kept_eigenvectors"] = torch.tensor(
            pca.components_[:components_to_keep],
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )
        param_dict["discarded_eigenvectors"] = torch.tensor(
            pca.components_[components_to_keep:],
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )

        # compute the labels' projections on the discarded components, to estimate the e.g., 90th percentile and determine epsilon
        # absolute value is important -- projections can be negative.
        proj_discarded = torch.abs(
            torch.matmul(
                arr_for_pca.T,
                param_dict["discarded_eigenvectors"].clone().detach().cpu().T,
            )
        )
        # setting axis = 0 generalizes to multiple discarded components
        epsilon = np.percentile(
            proj_discarded.numpy(), empirical_epsilon_percentile, axis=0
        )
        param_dict["epsilon"] = torch.tensor(
            epsilon,
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )

        self.pca_param_dict = param_dict

    def full_dataloader(self):  # TODO: we're not really using it
        return DataLoader(self.fulldataset, batch_size=1, num_workers=self.num_workers)

    def unlabeled_dataloader(self):
        return self.semi_supervised_loader

    ## That's the clean train_dataloader that works. can revert to it if needed
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_set,
    #         batch_size=self.train_batch_size,
    #         num_workers=self.num_workers,
    #     )

    def train_dataloader(  # TODO: verify that indeed the semi_supervised_loader does its job
        self,
    ):  # TODO: I don't like that the function returns a list or a dataloader.
        # if self.trainer.current_epoch % 2 == 0:
        #    return self.semi_supervised_loader
        # else:
        # return DataLoader(self.train_set, batch_size = self.train_batch_size, num_workers = self.num_workers)    
        loader = {
            "labeled": DataLoader(
                self.train_set,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers
                // 2,  # TODO: keep track of num_workers
            ),
            "unlabeled": self.unlabeled_dataloader(),
        }
        return loader
        
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.validation_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers
        )
