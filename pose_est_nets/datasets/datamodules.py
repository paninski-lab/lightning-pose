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
from pose_est_nets.datasets.DALI import video_pipe, LightningWrapper
import h5py
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typeguard import typechecked
import sklearn
from typing_extensions import Literal
from pose_est_nets.datasets.utils import split_sizes_from_probabilities

# Maybe make torch manual seed a global variable?
TORCH_MANUAL_SEED = 42
_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
_SEQUENCE_LENGTH_UNSUPERVISED = 7
_INITIAL_PREFETCH_SIZE = 16
_BATCH_SIZE_UNSUPERVISED = 1  # sequence_length * batch_size = num_images passed
_DALI_RANDOM_SEED = 123456


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
    def __init__(  # TODO: add documentation and args
        self,
        dataset,
        use_deterministic: Optional[bool] = False,
        train_batch_size: Optional[int] = 16,
        validation_batch_size: Optional[int] = 16,
        test_batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 8,
    ):
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # maybe can make the view information more general when deciding on a specific format for csv files
        self.use_deterministic = use_deterministic

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

        data_splits_list = split_sizes_from_probabilities(datalen, 0.8, 0.1)

        self.train_set, self.valid_set, self.test_set = random_split(
            self.fulldataset,
            data_splits_list,
            generator=torch.Generator().manual_seed(TORCH_MANUAL_SEED),
        )

        print(
            "Size of -- train set: {}, validation set: {}, test set: {}".format(
                len(self.train_set), len(self.valid_set), len(self.test_set)
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
            self.valid_set,
            batch_size=self.validation_batch_size,
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
        use_deterministic: Optional[bool] = False,
        train_batch_size: Optional[int] = 16,
        validation_batch_size: Optional[int] = 16,
        test_batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 8,
        unlabeled_video_path: Optional[str] = None,
    ):
        super().__init__(
            dataset,
            use_deterministic,
            train_batch_size,
            validation_batch_size,
            test_batch_size,
            num_workers,
        )
        self.unlabeled_video_path = unlabeled_video_path
        self.num_workers_for_unlabeled = num_workers // 2
        self.num_workers_for_labeled = num_workers // 2
        self.setup_unlabeled()

    def setup_unlabeled(self):
        data_pipe = video_pipe(
            batch_size=_BATCH_SIZE_UNSUPERVISED,
            num_threads=self.num_workers_for_unlabeled,  # because the other workers do the labeled dataloading
            device_id=0,  # TODO: be careful when scaling to multinode
            resize_dims=[self.fulldataset.height, self.fulldataset.width],
            random_shuffle=True,
            filenames=self.unlabeled_video_path,
            seed=_DALI_RANDOM_SEED,
        )

        self.semi_supervised_loader = LightningWrapper(
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,  # TODO: seems harmless, but verify at some point what "reseting" means
        )

    # TODO: could be separated from this class
    # TODO: return something?
    def computePCA_params(
        self,
        components_to_keep: Optional[int] = 3,
        empirical_epsilon_percentile: Optional[float] = 90.0,
    ) -> None:
        print("Computing PCA on the labels...")
        param_dict = {}
        # Nick: Subset inherits from dataset, it doesn't have access to dataset.labels
        if type(self.train_set) == torch.utils.data.dataset.Subset:
            indxs = torch.tensor(self.train_set.indices)
            data_arr = torch.index_select(self.train_set.dataset.labels, 0, indxs)
            num_body_parts = self.train_set.dataset.num_targets
        else:
            data_arr = self.train_set.labels  # won't work for random splitting
            num_body_parts = self.train_set.num_targets
        # TODO: format_mouse_data is specific to Rick's dataset, change when we're scaling to more data sources
        data_arr_resized = torch.tensor(shape = data_arr.reshaped)
        
        

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
