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
from pose_est_nets.datasets.utils import (
    clean_any_nans,
)  # TODO: merge the two utils above
from pose_est_nets.datasets.DALI import video_pipe, LightningWrapper
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
import h5py
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typeguard import typechecked
import sklearn
from typing_extensions import Literal
from pose_est_nets.datasets.utils import split_sizes_from_probabilities

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@typechecked
def PCA_prints(pca: sklearn.decomposition._pca.PCA, components_to_keep: int) -> None:
    print("Results of running PCA on labels:")
    print(
        "explained_variance_ratio_: {}".format(
            np.round(pca.explained_variance_ratio_, 3)
        )
    )
    print(
        "total_explained_var: {}".format(
            np.round(np.sum(pca.explained_variance_ratio_[:components_to_keep]), 3)
        )
    )


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: float = None,
        test_probability: float = None,
        train_frames: float = None,
        torch_seed: int = 42,
    ):
        """Data module splits a dataset into train, val, and test data loaders.

        Args:
            dataset (torch.utils.data.Dataset): base dataset to be split into train/val/test
            use_deterministic (bool):
            train_batch_size (int): number of samples of training batches
            val_batch_size (int): number of samples in validation batches
            test_batch_size (int): number of samples in test batches
            num_workers (int): number of threads used for prefetching data
            train_probability (float): fraction of full dataset used for training
            val_probability (float): fraction of full dataset used for validation
            test_probability (float): fraction of full dataset used for testing
            train_frames (float or int): if integer, select this number of training frames from the
                initially selected train frames (defined by `train_probability`); if float, must be
                between 0 and 1 (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed (int, optional):

        """
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # maybe can make the view information more general when deciding on a specific format for
        # csv files
        self.use_deterministic = use_deterministic
        # info about dataset splits
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_set = None  # populated by self.setup()
        self.val_set = None  # populated by self.setup()
        self.test_set = None  # populated by self.setup()
        self.torch_seed = torch_seed

    def setup(self, stage: Optional[str] = None):  # TODO: clean up
        print("Setting up DataModule...")
        datalen = self.fulldataset.__len__()
        print(
            "Number of labeled images in the full dataset (train+val+test): {}".format(
                datalen
            )
        )

        if self.use_deterministic:
            return

        # split data based on provided probabilities
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=self.train_probability,
            val_probability=self.val_probability,
            test_probability=self.test_probability)

        self.train_set, self.val_set, self.test_set = random_split(
            self.fulldataset,
            data_splits_list,
            generator=torch.Generator().manual_seed(self.torch_seed),
        )

        # further subsample training data if desired
        if self.train_frames is not None:
            split = True
            if self.train_frames >= len(self.train_set):
                # take max number of train frames
                print("Warning! Requested training frames exceeds training set size; using all")
                n_frames = len(self.train_set)
                split = False
            elif self.train_frames == 1:
                # assume this is a fraction; use full dataset
                n_frames = len(self.train_set)
                split = False
            elif self.train_frames > 1:
                # take this number of train frames
                n_frames = int(self.train_frames)
            elif self.train_frames > 0:
                # take this fraction of train frames
                n_frames = int(self.train_frames * len(self.train_set))
            else:
                raise ValueError("train_frames must be >0")
            if split:
                self.train_set, _ = random_split(
                    self.train_set,
                    [n_frames, len(self.train_set) - n_frames],
                    generator=torch.Generator().manual_seed(self.torch_seed),
                )

        print(
            "Size of -- train set: {}, validation set: {}, test set: {}".format(
                len(self.train_set), len(self.val_set), len(self.test_set)
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )


class UnlabeledDataModule(BaseDataModule):
    def __init__(  # TODO: add documentation and args
        self,
        dataset,
        video_paths_list: List[str],
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        unlabeled_batch_size: int = 1,
        unlabeled_sequence_length: int = 16,
        dali_seed: int = 123456,
        torch_seed: int = 42,
        specialized_dataprep: Optional[Literal["pca"]] = None,  # Get rid of optional?
        loss_param_dict: Optional[dict] = None,
    ):
        super().__init__(
            dataset=dataset,
            use_deterministic=use_deterministic,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            torch_seed=torch_seed
        )
        self.video_paths_list = video_paths_list
        self.num_workers_for_unlabeled = num_workers // 2
        self.num_workers_for_labeled = num_workers // 2
        self.unlabeled_batch_size = unlabeled_batch_size
        self.unlabeled_sequence_length = unlabeled_sequence_length
        self.dali_seed = dali_seed
        self.torch_seed = torch_seed
        self.semi_supervised_loader = None  # initialized in setup_unlabeled
        super().setup()
        self.setup_unlabeled()
        self.loss_param_dict = loss_param_dict
        if specialized_dataprep:  # it's not None
            if "pca" in specialized_dataprep:
                self.computePCA_params()

    def setup_unlabeled(self):
        data_pipe = video_pipe(
            batch_size=self.unlabeled_batch_size,
            sequence_length=self.unlabeled_sequence_length,
            num_threads=self.num_workers_for_unlabeled,  # because the other workers do the labeled dataloading
            device_id=0,  # TODO: be careful when scaling to multinode
            resize_dims=[self.fulldataset.height, self.fulldataset.width],
            random_shuffle=True,
            filenames=self.video_paths_list,
            seed=self.dali_seed,
        )

        self.semi_supervised_loader = LightningWrapper(
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,  # TODO: seems harmless, but verify at some point what "reseting" means
        )
        # self.computePCA_params() #Setup must be run before running this

    # TODO: could be separated from this class
    # TODO: return something?
    def computePCA_params(  # Should only call this now if pca in loss name dict
        self,
        components_to_keep: int = 3,
        empirical_epsilon_percentile: float = 90.0,
    ) -> None:
        print("Computing PCA on the labels...")
        # Nick: Subset inherits from dataset, it doesn't have access to dataset.labels
        if type(self.train_set) == torch.utils.data.dataset.Subset:
            indxs = torch.tensor(self.train_set.indices)
            regressionData = (
                super(type(self.fulldataset), self.fulldataset)
                if type(self.fulldataset) == HeatmapDataset
                else self.fulldataset
            )
            data_arr = torch.index_select(
                self.fulldataset.labels.detach().clone(), 0, indxs
            )
            if self.fulldataset.imgaug_transform:
                i = 0
                for idx in indxs:
                    data_arr[i] = regressionData.__getitem__(idx)[1].reshape(-1, 2)
                    i += 1
        else:
            data_arr = (
                self.train_set.labels.detach().clone()
            )  # won't work for random splitting
            if self.train_set.imgaug_transform:
                for i in range(len(data_arr)):
                    data_arr[i] = super(
                        type(self.train_set), self.train_set
                    ).__getitem__(i)[1]

        # TODO: format_mouse_data is specific to Rick's dataset, change when we're scaling to more data sources
        arr_for_pca = format_mouse_data(data_arr)
        print("initial_arr_for_pca shape: {}".format(arr_for_pca.shape))
        # Dan's cleanup:
        good_arr_for_pca = clean_any_nans(arr_for_pca, dim=0)
        pca = PCA(n_components=4, svd_solver="full")
        pca.fit(good_arr_for_pca.T)
        print("Done!")

        print(
            "good_arr_for_pca shape: {}".format(good_arr_for_pca.shape)
        )  # TODO: have prints as tests
        PCA_prints(pca, components_to_keep)  # print important params
        self.loss_param_dict["pca"]["kept_eigenvectors"] = torch.tensor(
            pca.components_[:components_to_keep],
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )
        self.loss_param_dict["pca"]["discarded_eigenvectors"] = torch.tensor(
            pca.components_[components_to_keep:],
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )

        # compute the labels' projections on the discarded components, to estimate the e.g., 90th percentile and determine epsilon
        # absolute value is important -- projections can be negative.
        proj_discarded = torch.abs(
            torch.matmul(
                arr_for_pca.T,
                self.loss_param_dict["pca"]["discarded_eigenvectors"]
                .clone()
                .detach()
                .cpu()
                .T,
            )
        )
        # setting axis = 0 generalizes to multiple discarded components
        epsilon = np.nanpercentile(
            proj_discarded.numpy(), empirical_epsilon_percentile, axis=0
        )
        print(epsilon)
        self.loss_param_dict["pca"]["epsilon"] = torch.tensor(
            epsilon,
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )

    def unlabeled_dataloader(self):
        return self.semi_supervised_loader

    def train_dataloader(
        self,
    ):
        loader = {
            "labeled": DataLoader(
                self.train_set,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers_for_labeled,
            ),
            "unlabeled": self.unlabeled_dataloader(),
        }
        return loader

    # TODO: check if necessary
    def predict_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers_for_labeled,
        )
