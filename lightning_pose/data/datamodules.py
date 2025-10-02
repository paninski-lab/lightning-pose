"""Data modules split a dataset into train, val, and test modules."""

import copy
import os
from typing import Literal

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split

from lightning_pose.data.dali import PrepareDALI
from lightning_pose.data.datatypes import SemiSupervisedDataLoaderDict
from lightning_pose.data.utils import (
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
        num_workers: int | None = None,
        train_probability: float = 0.8,
        val_probability: float | None = None,
        test_probability: float | None = None,
        train_frames: float | int | None = None,
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
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                self.num_workers = int(slurm_cpus)
            else:
                # Fallback to os.cpu_count()
                self.num_workers = os.cpu_count()
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_dataset = None  # populated by self.setup()
        self.val_dataset = None  # populated by self.setup()
        self.test_dataset = None  # populated by self.setup()
        self.torch_seed = torch_seed
        self._setup()

    def _setup(self) -> None:

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
            if self.dataset.imgaug_transform[-1].__str__().find("Resize") == 0:
                final_transform = iaa.Sequential([self.dataset.imgaug_transform[-1]])
            else:
                # if we're here it's because the dataset is a MultiviewHeatmapDataset that doesn't
                # resize by default in the pipeline; we enforce resizing here on val/test batches
                height = self.dataset.height
                width = self.dataset.width
                final_transform = iaa.Sequential([iaa.Resize({"height": height, "width": width})])

            self.val_dataset.dataset.imgaug_transform = final_transform
            if hasattr(self.val_dataset.dataset, "dataset"):
                # this will get triggered for multiview datasets
                print("val: updating children datasets with resize imgaug pipeline")
                for view_name, dset in self.val_dataset.dataset.dataset.items():
                    dset.imgaug_transform = final_transform

            self.test_dataset.dataset.imgaug_transform = final_transform
            if hasattr(self.test_dataset.dataset, "dataset"):
                # this will get triggered for multiview datasets
                print("test: updating children datasets with resize imgaug pipeline")
                for view_name, dset in self.test_dataset.dataset.dataset.items():
                    dset.imgaug_transform = final_transform

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
        video_paths_list: list[str] | str,
        dali_config: dict | DictConfig,
        view_names: list[str] | None = None,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int | None = None,
        train_probability: float = 0.8,
        val_probability: float | None = None,
        test_probability: float | None = None,
        train_frames: float | None = None,
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
        # CombinedLoader mode="max_size_cycle" works in concert with
        # `trainer.limit_train_batches`. Assuming unlabeled data is plentiful,
        # it will cycle through labeled data until limit_train_batches.
        # We set limit_train_batches such that it exhausts all labeled data
        # in an epoch, or it cycles for a minimum of 10 batches.
        #
        # The reason to have a minimum number of batches is so that when labeled data is
        # scarce, the model sees more unlabeled data per epoch instead of just stopping
        # (empirically better).
        return CombinedLoader(loader, mode="max_size_cycle")
