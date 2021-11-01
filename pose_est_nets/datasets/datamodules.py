"""Data modules split a dataset into train, val, and test modules."""

import numpy as np
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import os
import pytorch_lightning as pl
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, random_split
from typeguard import typechecked
from typing import Literal, List, Optional, Tuple, Union

from pose_est_nets.datasets.dali import video_pipe, LightningWrapper
from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
from pose_est_nets.datasets.utils import clean_any_nans, split_sizes_from_probabilities
from pose_est_nets.utils.heatmap_tracker_utils import format_mouse_data

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseDataModule(pl.LightningDataModule):
    """Splits a labeled dataset into train, val, and test data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[Union[float, int]] = None,
        torch_seed: int = 42,
    ) -> None:
        """Data module splits a dataset into train, val, and test data loaders.

        Args:
            dataset: base dataset to be split into train/val/test
            use_deterministic: TODO: use deterministic split of data...?
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed: control data splits

        """
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # maybe can make the view information more general when deciding on a
        # specific format for csv files
        self.use_deterministic = use_deterministic
        # info about dataset splits
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_dataset = None  # populated by self.setup()
        self.val_dataset = None  # populated by self.setup()
        self.test_dataset = None  # populated by self.setup()
        self.torch_seed = torch_seed

    def setup(self, stage: Optional[str] = None):  # stage arg needed for ptl

        datalen = self.dataset.__len__()
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
            test_probability=self.test_probability,
        )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            data_splits_list,
            generator=torch.Generator().manual_seed(self.torch_seed),
        )

        # further subsample training data if desired
        if self.train_frames is not None:
            split = True
            if self.train_frames >= len(self.train_dataset):
                # take max number of train frames
                print(
                    "Warning! Requested training frames exceeds training "
                    + "set size; using all"
                )
                n_frames = len(self.train_dataset)
                split = False
            elif self.train_frames == 1:
                # assume this is a fraction; use full dataset
                n_frames = len(self.train_dataset)
                split = False
            elif self.train_frames > 1:
                # take this number of train frames
                n_frames = int(self.train_frames)
            elif self.train_frames > 0:
                # take this fraction of train frames
                n_frames = int(self.train_frames * len(self.train_dataset))
            else:
                raise ValueError("train_frames must be >0")
            if split:
                self.train_dataset, _ = random_split(
                    self.train_dataset,
                    [n_frames, len(self.train_dataset) - n_frames],
                    generator=torch.Generator().manual_seed(self.torch_seed),
                )

        print(
            "Size of -- train set: {}, val set: {}, test set: {}".format(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )


class UnlabeledDataModule(BaseDataModule):
    """Data module that contains labeled and unlabled data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        video_paths_list: Union[List[str], str],
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[float] = None,
        unlabeled_batch_size: int = 1,
        unlabeled_sequence_length: int = 16,
        dali_seed: int = 123456,
        torch_seed: int = 42,
        device_id: int = 0,
        specialized_dataprep: Optional[Literal["pca"]] = None,
        loss_param_dict: Optional[dict] = None,
    ) -> None:
        """Data module that contains labeled and unlabeled data loaders.

        Args:
            dataset: pytorch Dataset for labeled data
            video_paths_list: absolute paths of videos ("unlabeled" data)
            use_deterministic: TODO: use deterministic split of data...?
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed: control data splits
            unlabeled_batch_size: number of sequences to load per unlabeled
                batch
            unlabeled_sequence_length: number of frames per sequence of
                unlabeled data
            dali_seed: control randomness of unlabeled data loading
            torch_seed: control randomness of labeled data loading
            device_id: gpu for unlabeled data loading
            specialized_dataprep:
            loss_param_dict: details of loss types for unlabeled data
                (influences processing)

        """
        super().__init__(
            dataset=dataset,
            use_deterministic=use_deterministic,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            train_probability=train_probability,
            val_probability=val_probability,
            test_probability=test_probability,
            train_frames=train_frames,
            torch_seed=torch_seed,
        )
        self.video_paths_list = video_paths_list
        self.num_workers_for_unlabeled = num_workers // 2
        self.num_workers_for_labeled = num_workers // 2
        self.unlabeled_batch_size = unlabeled_batch_size
        self.unlabeled_sequence_length = unlabeled_sequence_length
        self.dali_seed = dali_seed
        self.torch_seed = torch_seed
        self.device_id = device_id
        self.semi_supervised_loader = None  # initialized in setup_unlabeled
        super().setup()
        self.setup_unlabeled()
        self.loss_param_dict = loss_param_dict
        if specialized_dataprep:  # it's not None
            if "pca" in specialized_dataprep:
                self.computePCA_params()

    def setup_unlabeled(self):

        # get input data
        if isinstance(self.video_paths_list, list):
            # presumably a list of files
            filenames = self.video_paths_list
        elif isinstance(self.video_paths_list, str) and os.path.isfile(
            self.video_paths_list
        ):
            # single video file
            filenames = self.video_paths_list
        elif isinstance(self.video_paths_list, str) and os.path.isdir(
            self.video_paths_list
        ):
            # directory of videos
            import glob

            extensions = ["mp4"]  # allowed file extensions
            filenames = []
            for extension in extensions:
                filenames.extend(
                    glob.glob(os.path.join(self.video_paths_list, "*.%s" % extension))
                )
        else:
            raise ValueError(
                "`video_paths_list` must be a list of files, a single file, "
                + "or a directory name"
            )

        data_pipe = video_pipe(
            filenames=filenames,
            resize_dims=[self.dataset.height, self.dataset.width],
            random_shuffle=True,
            seed=self.dali_seed,
            sequence_length=self.unlabeled_sequence_length,
            batch_size=self.unlabeled_batch_size,
            num_threads=self.num_workers_for_unlabeled,  # other workers do the labeled dataloading
            device_id=self.device_id,
        )

        self.semi_supervised_loader = LightningWrapper(
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,  # TODO: verify what "reseting" means
        )

    # TODO: could be separated from this class
    # TODO: return something?
    def computePCA_params(  # Should only call this if pca in loss name dict
        self,
        components_to_keep: int = 3,
        empirical_epsilon_percentile: float = 90.0,
    ) -> None:
        print("Computing PCA on the keypoints...")
        # Nick: Subset inherits from dataset, it doesn't have access to
        # dataset.keypoints
        if type(self.train_dataset) == torch.utils.data.dataset.Subset:
            indxs = torch.tensor(self.train_dataset.indices)
            regressionData = (
                super(type(self.dataset), self.dataset)
                if type(self.dataset) == HeatmapDataset
                else self.dataset
            )
            data_arr = torch.index_select(
                self.dataset.keypoints.detach().clone(), 0, indxs
            )
            if self.dataset.imgaug_transform:
                i = 0
                for idx in indxs:
                    vals = regressionData.__getitem__(idx)
                    data_arr[i] = vals[1].reshape(-1, 2)
                    i += 1
        else:
            data_arr = (
                self.train_dataset.keypoints.detach().clone()
            )  # won't work for random splitting
            if self.train_dataset.imgaug_transform:
                for i in range(len(data_arr)):
                    data_arr[i] = super(
                        type(self.train_dataset), self.train_dataset
                    ).__getitem__(i)[1]

        # TODO: format_mouse_data is specific to Rick's dataset, change when
        # TODO: we're scaling to more data sources
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
        pca_prints(pca, components_to_keep)  # print important params
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

        # compute the keypoints' projections on the discarded components, to
        # estimate the e.g., 90th percentile and determine epsilon
        # absolute value is important -- projections can be negative.
        discarded_eigs = self.loss_param_dict["pca"]["discarded_eigenvectors"]
        proj_discarded = torch.abs(
            torch.matmul(
                arr_for_pca.T,
                discarded_eigs.clone().detach().cpu().T,  # TODO: why cpu?
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

    def train_dataloader(self):
        loader = {
            "labeled": DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers_for_labeled,
            ),
            "unlabeled": self.unlabeled_dataloader(),
        }
        return loader

    # TODO: check if necessary
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers_for_labeled,
        )


@typechecked
def pca_prints(pca: PCA, components_to_keep: int) -> None:
    print("Results of running PCA on keypoints:")
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
