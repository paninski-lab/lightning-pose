"""Data modules split a dataset into train, val, and test modules."""

from __future__ import annotations

import copy
import logging
import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lightning_pose.data.datasets import (
        BaseTrackingDataset,
        HeatmapDataset,
        MultiviewHeatmapDataset,
    )

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, Subset, random_split

from lightning_pose.data.datatypes import SemiSupervisedDataLoaderDict
from lightning_pose.data.samplers import TemperatureSampler
from lightning_pose.data.utils import (
    compute_num_train_frames,
    split_sizes_from_probabilities,
)
from lightning_pose.utils.io import check_video_paths

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


class BaseDataModule(pl.LightningDataModule):
    """Splits a labeled dataset into train, val, and test data loaders."""

    def __init__(
        self,
        dataset: BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int | None = None,
        train_probability: float = 0.8,
        val_probability: float | None = None,
        test_probability: float | None = None,
        train_frames: float | int | None = None,
        torch_seed: int = 42,
        sampling_temperature: float | None = None,
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
            sampling_temperature: multi-dataset sampling temperature over supervision
                mass (see :class:`~lightning_pose.data.samplers.TemperatureSampler`).
                None or 1 keeps the stock shuffled loader; values > 1 require the
                dataset to carry ``dataset_ids`` (i.e. ``data.dataset_names`` set).

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
                self.num_workers = os.cpu_count() or 0
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_dataset: Subset | None = None
        self.val_dataset: Subset | None = None
        self.test_dataset: Subset | None = None
        self.torch_seed = torch_seed
        self.sampling_temperature = sampling_temperature
        self.train_sampler = None
        self._setup()
        self._setup_train_sampler()

    def _setup(self) -> None:
        """Split the dataset into train, validation, and test subsets."""

        datalen = len(self.dataset)
        logger.info(f'number of labeled images in the full dataset (train+val+test): {datalen}')

        # select indices for each split; stratified by source dataset in multi-dataset mode
        train_idxs, val_idxs, test_idxs = self._split_indices()

        imgaug_hflip = getattr(self.dataset, 'imgaug_hflip', False)
        imgaug_per_dataset = getattr(self.dataset, 'imgaug_transform_per_dataset', None)
        # the imgaug pipeline always contains at least one element (the final resize transform);
        # len == 1 therefore means "no augmentations" and subsets can safely share the same
        # underlying dataset object. imgaug_hflip and imgaug_transform_per_dataset are checked
        # separately because they are applied outside the global pipeline in __getitem__ and
        # must also be stripped from val/test.
        if (
            len(self.dataset.imgaug_transform) == 1  # type: ignore[arg-type]
            and not imgaug_hflip
            and imgaug_per_dataset is None
        ):
            # no augmentations in the pipeline; subsets can share same underlying dataset
            self.train_dataset = Subset(self.dataset, indices=train_idxs)
            self.val_dataset = Subset(self.dataset, indices=val_idxs)
            self.test_dataset = Subset(self.dataset, indices=test_idxs)
        else:
            # augmentations in the pipeline; we want validation and test datasets that only resize
            # we can't simply change the imgaug pipeline in the datasets after they've been split
            # because the subsets actually point to the same underlying dataset, so we create
            # separate datasets here
            self.train_dataset = Subset(copy.deepcopy(self.dataset), indices=train_idxs)
            self.val_dataset = Subset(copy.deepcopy(self.dataset), indices=val_idxs)
            self.test_dataset = Subset(copy.deepcopy(self.dataset), indices=test_idxs)

            # only use the final resize transform for the validation and test datasets
            if self.dataset.imgaug_transform[-1].__str__().find("Resize") == 0:  # type: ignore[index]
                final_transform = iaa.Sequential([self.dataset.imgaug_transform[-1]])  # type: ignore[index]
            else:
                # if we're here it's because the dataset is a MultiviewHeatmapDataset that doesn't
                # resize by default in the pipeline; we enforce resizing here on val/test batches
                height = self.dataset.height
                width = self.dataset.width
                final_transform = iaa.Sequential([iaa.Resize({"height": height, "width": width})])

            self.val_dataset.dataset.imgaug_transform = final_transform  # type: ignore[union-attr]
            self.val_dataset.dataset.imgaug_hflip = False  # type: ignore[union-attr]
            self.val_dataset.dataset.imgaug_transform_per_dataset = None  # type: ignore[union-attr]
            if hasattr(self.val_dataset.dataset, "dataset"):
                # this will get triggered for multiview datasets
                logger.debug('val: updating children datasets with resize imgaug pipeline')
                for _view_name, dset in self.val_dataset.dataset.dataset.items():  # type: ignore[union-attr]
                    dset.imgaug_transform = final_transform
                    dset.imgaug_hflip = False

            self.test_dataset.dataset.imgaug_transform = final_transform  # type: ignore[union-attr]
            self.test_dataset.dataset.imgaug_hflip = False  # type: ignore[union-attr]
            self.test_dataset.dataset.imgaug_transform_per_dataset = None  # type: ignore[union-attr]
            if hasattr(self.test_dataset.dataset, "dataset"):
                # this will get triggered for multiview datasets
                logger.debug('test: updating children datasets with resize imgaug pipeline')
                for _view_name, dset in self.test_dataset.dataset.dataset.items():  # type: ignore[union-attr]
                    dset.imgaug_transform = final_transform
                    dset.imgaug_hflip = False

        # further subsample training data if desired
        if self.train_frames is not None:
            n_frames = compute_num_train_frames(len(self.train_dataset), self.train_frames)

            if n_frames < len(self.train_dataset):
                # split the data a second time to reflect further subsampling from
                # train_frames
                self.train_dataset.indices = self.train_dataset.indices[:n_frames]

        logger.info(
            f'dataset splits -- '
            f'train: {len(self.train_dataset)}, '
            f'val: {len(self.val_dataset)}, '
            f'test: {len(self.test_dataset)}'
        )

    def _split_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Select train/val/test indices, stratified by source dataset when available.

        When the underlying dataset carries no ``dataset_ids`` (stock Lightning Pose),
        this reproduces the historical global ``random_split`` exactly, including its
        use of a generator seeded with ``torch_seed`` — existing splits are unchanged.

        In multi-dataset mode, the split probabilities are instead applied within each
        source dataset (in registry-id order, deterministically), so every source is
        represented in train and val at the same proportions regardless of size. The
        concatenated train list is then reshuffled so that positional subsampling
        (``train_frames``) draws uniformly across sources rather than from whichever
        dataset happens to come first.

        Returns:
            tuple of (train, val, test) index lists into the full dataset.
        """
        generator = torch.Generator().manual_seed(self.torch_seed)
        dataset_ids = getattr(self.dataset, 'dataset_ids', None)

        if dataset_ids is None:
            data_splits_list = split_sizes_from_probabilities(
                len(self.dataset),
                train_probability=self.train_probability,
                val_probability=self.val_probability,
                test_probability=self.test_probability,
            )
            splits = random_split(
                range(len(self.dataset)),  # type: ignore[arg-type]
                data_splits_list,
                generator=generator,
            )
            return tuple(list(s) for s in splits)  # type: ignore[return-value]

        train_idxs: list[int] = []
        val_idxs: list[int] = []
        test_idxs: list[int] = []
        for dataset_id in torch.unique(dataset_ids, sorted=True):
            idxs = torch.nonzero(dataset_ids == dataset_id).flatten()
            perm = idxs[torch.randperm(len(idxs), generator=generator)]
            n_train, n_val, _n_test = split_sizes_from_probabilities(
                len(idxs),
                train_probability=self.train_probability,
                val_probability=self.val_probability,
                test_probability=self.test_probability,
            )
            train_idxs += perm[:n_train].tolist()
            val_idxs += perm[n_train:n_train + n_val].tolist()
            test_idxs += perm[n_train + n_val:].tolist()
            name = self.dataset.dataset_names[int(dataset_id)]  # type: ignore[index]
            logger.info(
                f'stratified split -- {name}: '
                f'train {n_train}, val {n_val}, test {len(idxs) - n_train - n_val}'
            )

        # reshuffle so positional subsampling (train_frames) is not biased by source order
        order = torch.randperm(len(train_idxs), generator=generator)
        train_idxs = [train_idxs[i] for i in order]
        return train_idxs, val_idxs, test_idxs

    def _setup_train_sampler(self) -> None:
        """Install a temperature sampler over the training subset when configured.

        No sampler is constructed for ``sampling_temperature`` None or 1: T=1 is
        frame-proportional sampling, which the stock ``shuffle=True`` loader already
        implements — leaving that path untouched keeps the T=1 cell of a temperature
        sweep exactly identical to a stock run rather than merely equivalent.
        """
        T = self.sampling_temperature
        if T is None or T == 1:
            return

        dataset_ids = getattr(self.dataset, 'dataset_ids', None)
        if dataset_ids is None:
            raise ValueError(
                f'sampling_temperature={T} requires a multi-dataset corpus: set '
                f'data.dataset_names so per-example dataset ids are available'
            )

        train_idxs = list(self.train_dataset.indices)
        train_ids = dataset_ids[train_idxs]

        # kbar_d: mean labeled (visible==2) keypoints per frame, from the train split only —
        # deriving it from the full CSV would leak validation composition into the sampler
        visibility = getattr(self.dataset, 'visibility', None)
        if visibility is not None:
            labeled_per_frame = (visibility[train_idxs] == 2).sum(dim=1).double()
        else:
            labeled_per_frame = (
                ~torch.isnan(self.dataset.keypoints[train_idxs, :, 0])
            ).sum(dim=1).double()

        num_datasets = len(self.dataset.dataset_names)  # type: ignore[union-attr]
        kbar = torch.zeros(num_datasets, dtype=torch.double)
        for d in range(num_datasets):
            mask = train_ids == d
            if mask.any():
                kbar[d] = labeled_per_frame[mask].mean()

        # yields positions within train_dataset (the Subset resolves them to full-dataset rows)
        self.train_sampler = TemperatureSampler(
            dataset_ids=train_ids,
            kbar=kbar,
            temperature=float(T),
            seed=self.torch_seed,
        )

    def split_manifest(self) -> dict | None:
        """Reproducibility manifest for multi-dataset runs; None in stock mode.

        Records the split seed, the registry, per-split per-dataset counts, and the
        exact image names in each split. Downstream consumers (the temperature
        sampler's ``k̄_d``, evaluation) must be reproducible from this manifest alone.
        """
        dataset_ids = getattr(self.dataset, 'dataset_ids', None)
        if dataset_ids is None:
            return None

        names = self.dataset.image_names
        dataset_names = self.dataset.dataset_names  # type: ignore[union-attr]

        def entry(subset: Subset) -> dict:
            idxs = list(subset.indices)
            counts = torch.bincount(dataset_ids[idxs], minlength=len(dataset_names))
            return {
                'n': len(idxs),
                'per_dataset': {n: int(c) for n, c in zip(dataset_names, counts)},
                'image_names': [names[i] for i in idxs],
            }

        return {
            'torch_seed': self.torch_seed,
            'dataset_names': list(dataset_names),
            'train': entry(self.train_dataset),
            'val': entry(self.val_dataset),
            'test': entry(self.test_dataset),
        }

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader with shuffling enabled.

        Returns:
            DataLoader wrapping the training subset.
        """
        if self.train_sampler is not None:
            # temperature sampling: the sampler owns example order (shuffle must be False);
            # it yields positions within train_dataset, whose indices the Subset resolves
            return DataLoader(
                self.train_dataset,  # type: ignore[arg-type]
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                sampler=self.train_sampler,
            )
        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.torch_seed),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the validation dataloader.

        Returns:
            DataLoader wrapping the validation subset.
        """
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the test dataloader.

        Returns:
            DataLoader wrapping the test subset.
        """
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def full_labeled_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a dataloader covering the entire labeled dataset (all splits combined).

        Returns:
            DataLoader over the full underlying dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
        )


class UnlabeledDataModule(BaseDataModule):
    """Data module that contains labeled and unlabled data loaders."""

    def __init__(
        self,
        dataset: BaseTrackingDataset | HeatmapDataset | MultiviewHeatmapDataset,
        video_paths_list: list[str] | str,
        dali_config: dict | DictConfig | ListConfig,
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
        from lightning_pose.data.dali import PrepareDALI  # avoids ImportError on cpu-only installs
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
        """Return a combined dataloader pairing labeled and unlabeled training data.

        Returns:
            ``CombinedLoader`` in ``max_size_cycle`` mode that cycles through labeled and
            unlabeled batches together.
        """
        assert self.unlabeled_dataloader is not None
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
