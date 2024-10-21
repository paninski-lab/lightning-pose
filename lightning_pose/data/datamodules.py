"""Data modules split a dataset into train, val, and test modules."""

import copy
from typing import List, Literal, Optional, Union

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split

from lightning_pose.data.dali import PrepareDALI
from lightning_pose.data.utils import (
    SemiSupervisedDataLoaderDict,
    compute_num_train_frames,
    split_sizes_from_probabilities,
)
from lightning_pose.utils.io import check_video_paths

# to ignore imports for sphix-autoapidoc
__all__ = [
    "BaseDataModule",
    "UnlabeledDataModule",
]


class BaseDataModule(pl.LightningDataModule):
    """Splits a labeled dataset into train, val, and test data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
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
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_dataset = None  # populated by self.setup()
        self.val_dataset = None  # populated by self.setup()
        self.test_dataset = None  # populated by self.setup()
        self.torch_seed = torch_seed

    def setup(self, stage: Optional[str] = None) -> None:  # stage arg needed for ptl

        datalen = self.dataset.__len__()
        print(f"Number of labeled images in the full dataset (train+val+test): {datalen}")

        # split data based on provided probabilities
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=self.train_probability,
            val_probability=self.val_probability,
            test_probability=self.test_probability,
        )

        if len(self.dataset.imgaug_transform) == 1:
            # no augmentations in the pipeline; subsets can share same underlying dataset
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                data_splits_list,
                generator=torch.Generator().manual_seed(self.torch_seed),
            )
        else:
            # augmentations in the pipeline; we want validation and test datasets that only resize
            # we can't simply change the imgaug pipeline in the datasets after they've been split
            # because the subsets actually point to the same underlying dataset, so we create
            # separate datasets here
            train_idxs, val_idxs, test_idxs = random_split(
                range(len(self.dataset)),
                data_splits_list,
                generator=torch.Generator().manual_seed(self.torch_seed),
            )

            self.train_dataset = Subset(copy.deepcopy(self.dataset), indices=list(train_idxs))
            self.val_dataset = Subset(copy.deepcopy(self.dataset), indices=list(val_idxs))
            self.test_dataset = Subset(copy.deepcopy(self.dataset), indices=list(test_idxs))

            # only use the final resize transform for the validation and test datasets
            resize_transform = iaa.Sequential([self.dataset.imgaug_transform[-1]])
            self.val_dataset.dataset.imgaug_transform = resize_transform
            self.test_dataset.dataset.imgaug_transform = resize_transform

        # further subsample training data if desired
        if self.train_frames is not None:
            n_frames = compute_num_train_frames(len(self.train_dataset), self.train_frames)

            if n_frames < len(self.train_dataset):
                # split the data a second time to reflect further subsampling from
                # train_frames
                self.train_dataset.indices = self.train_dataset.indices[:n_frames]

        print(
            f"Dataset splits -- "
            f"train: {len(self.train_dataset)}, "
            f"val: {len(self.val_dataset)}, "
            f"test: {len(self.test_dataset)}"
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.torch_seed),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def full_labeled_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )


class UnlabeledDataModule(BaseDataModule):
    """Data module that contains labeled and unlabled data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        video_paths_list: Union[List[str], str],
        dali_config: Union[dict, DictConfig],
        view_names: Optional[List[str]] = None,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[float] = None,
        torch_seed: int = 42,
        imgaug: Literal["default", "dlc", "dlc-top-down"] = "default",
    ) -> None:
        """Data module that contains labeled and unlabeled data loaders.

        Args:
            dataset: pytorch Dataset for labeled data
            video_paths_list: absolute paths of videos ("unlabeled" data)
            view_names: if fitting a non-mirrored multiview model, pass view names in order to
                correctly organize the video paths
            dali_config: see `dali` entry of default config file for keys
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
            torch_seed: control randomness of labeled data loading
            imgaug: type of image augmentation to apply to unlabeled frames

        """
        super().__init__(
            dataset=dataset,
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
        self.filenames = check_video_paths(self.video_paths_list, view_names=view_names)
        self.num_workers_for_unlabeled = 1  # WARNING!! do not increase above 1, weird behavior
        self.dali_config = dali_config
        self.unlabeled_dataloader = None  # initialized in setup_unlabeled
        self.imgaug = imgaug
        # TODO: Should these belong in a setup method that called by lightning,
        # rather than __init__? BaseDataModule already follows that pattern.
        super().setup()
        self.setup_unlabeled()

    def setup_unlabeled(self) -> None:
        """Sets up the unlabeled data loader."""
        dali_prep = PrepareDALI(
            train_stage="train",
            model_type="context" if self.dataset.do_context else "base",
            filenames=self.filenames,
            resize_dims=[self.dataset.height, self.dataset.width],
            dali_config=self.dali_config,
            imgaug=self.imgaug,
            num_threads=self.num_workers_for_unlabeled,
        )

        self.unlabeled_dataloader = dali_prep()

    def train_dataloader(self) -> CombinedLoader:
        loader = SemiSupervisedDataLoaderDict(
            labeled=super().train_dataloader(),
            unlabeled=self.unlabeled_dataloader,
        )
        return CombinedLoader(loader, mode="max_size_cycle")
