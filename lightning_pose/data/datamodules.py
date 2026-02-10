"""Data modules split a dataset into train, val, and test modules."""

import copy
import os
from typing import Literal

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split, WeightedRandomSampler, RandomSampler

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
        enable_weighted_sampler: bool = True,
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
            enable_weighted_sampler: If True, use a WeightedRandomSampler
            for the training dataloader to oversample examples with rarer keypoints.
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
        self.enable_weighted_sampler = enable_weighted_sampler
        self.train_sampler = None
        self._setup()
        
    def _calculate_train_sampler_weights(self, epsilon=1e-6):
        """Calculates weights for WeightedRandomSampler based on keypoint presence."""

        if not isinstance(self.train_dataset, Subset):
            print("Warning: Sampler weight calculation expects self.train_dataset to be a Subset. Skipping.")
            self.train_sampler = None
            return

        # Determine how to access keypoints based on dataset type
        underlying_dataset = self.train_dataset.dataset
        if hasattr(underlying_dataset, 'keypoints'): # BaseTrackingDataset or HeatmapDataset
            all_keypoints = underlying_dataset.keypoints
        elif hasattr(underlying_dataset, 'dataset') and isinstance(underlying_dataset.dataset, dict): # MultiviewHeatmapDataset
            # Using the first view as reference
            try:
                first_view_key = list(underlying_dataset.dataset.keys())[0]
                all_keypoints = underlying_dataset.dataset[first_view_key].keypoints
                print(f"Calculating sampler weights based on '{first_view_key}' view's keypoints for multiview.")
            except (IndexError, AttributeError):
                 print("Warning: Could not access keypoints from the first view of Multiview dataset. Skipping sampler.")
                 self.train_sampler = None
                 return
        else:
            print("Warning: Could not find keypoints attribute for sampler weight calculation. Skipping.")
            self.train_sampler = None
            return

        try:
            train_indices = self.train_dataset.indices
            # Ensure indices are valid for the keypoints tensor
            if max(train_indices) >= len(all_keypoints):
                 print(f"Warning: train_indices ({max(train_indices)}) out of bounds for all_keypoints ({len(all_keypoints)}). Skipping sampler.")
                 self.train_sampler = None
                 return
            train_keypoints = all_keypoints[train_indices]
        except IndexError as e:
            print(f"Error indexing keypoints with train_indices: {e}. Skipping sampler.")
            self.train_sampler = None
            return
        except Exception as e:
            print(f"Unexpected error accessing train keypoints: {e}. Skipping sampler.")
            self.train_sampler = None
            return

        # Check for NaNs (use x-coordinate)
        is_present = ~torch.isnan(train_keypoints[:, :, 0]) # Shape: [num_train_samples, num_keypoints]

        # Calculate frequency
        keypoint_counts = torch.sum(is_present, dim=0).float()
        num_train_samples = len(train_indices)
        if num_train_samples == 0:
            print("Warning: Zero samples in training set. Skipping sampler.")
            self.train_sampler = None
            return
        keypoint_frequencies = keypoint_counts / num_train_samples

        # Inverse frequency weights for keypoints
        inverse_frequencies = 1.0 / (keypoint_frequencies + epsilon)

        # Assign weight to each sample
        sample_weights = torch.sum(is_present * inverse_frequencies.unsqueeze(0), dim=1)

        # Handle cases where all keypoints might be NaN for a sample
        sample_weights[sample_weights == 0] = epsilon # Assign a tiny weight instead of zero

        # Create the sampler
        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print("Created WeightedRandomSampler for training data.")
        
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

        if self.enable_weighted_sampler:
            self._calculate_train_sampler_weights()
        else:
            self.train_sampler = None

        # print sampler status
        if self.train_sampler:
            print("Training sampler: WeightedRandomSampler enabled.")
        else:
            print("Training sampler: Standard shuffling enabled.")
            
        print(
            f"Dataset splits -- "
            f"train: {len(self.train_dataset)}, "
            f"val: {len(self.val_dataset)}, "
            f"test: {len(self.test_dataset)}"
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:

        if self.train_sampler is not None:
            sampler_arg = self.train_sampler
            shuffle_arg = None
            generator_arg = None
            print(f"DEBUG train_dataloader: Using sampler={type(sampler_arg)}")
        else:
            sampler_arg = None
            shuffle_arg = True
            generator_arg = torch.Generator().manual_seed(self.torch_seed)
            print(f"DEBUG train_dataloader: Using shuffle={shuffle_arg}, sampler=None")

        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            sampler=sampler_arg,
            shuffle=shuffle_arg,
            generator=generator_arg,
        )

        # (Optional debug prints after creation)
        print(f"DEBUG train_dataloader: DataLoader created.")
        if hasattr(loader, 'batch_sampler') and loader.batch_sampler is not None:
            print(f" -> batch_sampler type: {type(loader.batch_sampler)}")
            if hasattr(loader.batch_sampler, 'sampler'):
                print(f" -> underlying sampler type: {type(loader.batch_sampler.sampler)}")
            else:
                print(" -> batch_sampler has no 'sampler' attribute")
        else:
            print(f" -> No batch_sampler found on DataLoader.")
        print(f" -> shuffle attribute (post-init): {getattr(loader, 'shuffle', 'N/A')}")

        return loader


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
