"""Data modules split a dataset into train, val, and test modules."""

import os
from typing import Literal, Callable

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from lightning_pose.data.dali import PrepareDALI
from lightning_pose.data.datatypes import SemiSupervisedDataLoaderDict
from lightning_pose.utils.io import check_video_paths

# to ignore imports for sphix-autoapidoc
__all__ = [
    "BaseDataModule",
    "UnlabeledDataModule",
]


class BaseDataModule(pl.LightningDataModule):
    """Wraps labeled dataset splits and delegates loader creation to a factory."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        splits: tuple[Subset, Subset, Subset],
        dataloader_factory: Callable[[str], DataLoader]
    ) -> None:
        """Data module that uses an injected dataloader factory.

        Args:
            dataset: base dataset corresponding to provided splits
            splits: tuple of (train_subset, val_subset, test_subset)
            dataloader_factory: function mapping a stage string ("train"|"val"|"test")
                to a configured DataLoader for the corresponding split
        """
        super().__init__()
        self.dataset = dataset
        self.train_dataset, self.val_dataset, self.test_dataset = splits
        self._get_dataloader = dataloader_factory

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._get_dataloader("val")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self._get_dataloader("test")

    def full_labeled_dataloader(self) -> torch.utils.data.DataLoader:
        # Delegate to factory; expect it to support a 'full' stage for labeled data.
        return self._get_dataloader("full")


class UnlabeledDataModule(BaseDataModule):
    """Data module that contains labeled and unlabled data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        splits: tuple[Subset, Subset, Subset],
        dataloader_factory: Callable[[str], DataLoader],
        video_paths_list: list[str] | str,
        dali_config: dict | DictConfig,
        view_names: list[str] | None = None,
        imgaug: Literal["default", "dlc", "dlc-top-down"] = "default",
    ) -> None:
        """Data module that contains labeled and unlabeled data loaders.

        Args:
            dataset: pytorch Dataset for labeled data
            # Change: list[str] | list[list[str]] (singleview, multiview respectively)
            video_paths_list: absolute paths of videos ("unlabeled" data) # Replace with session list
            view_names: if fitting a non-mirrored multiview model, pass view names in order to
                correctly organize the video paths
            dali_config: see `dali` entry of default config file for keys
            imgaug: type of image augmentation to apply to unlabeled frames

        """
        super().__init__(
            dataset=dataset,
            splits=splits,
            dataloader_factory=dataloader_factory,
        )
        self.video_paths_list = video_paths_list
        # Replace with path_resolver(sessions,view_names) -> filenames.
        # Remove and use video_paths_list directly.
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
