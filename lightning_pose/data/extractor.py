"""Helper class to extract labeled data from a data module."""

from typing import Literal

import imgaug.augmenters as iaa
import torch
from jaxtyping import Float
from lightning.pytorch.utilities import CombinedLoader

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import (
    BaseTrackingDataset,
    HeatmapDataset,
    MultiviewHeatmapDataset,
)

# to ignore imports for sphix-autoapidoc
__all__ = ["DataExtractor"]


class DataExtractor:
    """Helper class to extract all data from a data module."""

    def __init__(
        self,
        data_module: BaseDataModule | UnlabeledDataModule,
        cond: Literal["train", "test", "val"] = "train",
        extract_images: bool = False,
        remove_augmentations: bool = True,
    ) -> None:
        """Initialize DataExtractor.

        Args:
            data_module: data module containing the labeled dataset and splits.
            cond: which data split to extract (``"train"``, ``"val"``, or ``"test"``).
            extract_images: if True, also extract and return image tensors.
            remove_augmentations: if True, rebuild the dataset with only resize augmentation
                before extracting, to avoid contaminating PCA fits with augmented data.
        """
        self.cond = cond
        self.extract_images = extract_images
        self.remove_augmentations = remove_augmentations

        if self.remove_augmentations:
            assert isinstance(
                data_module.dataset, (BaseTrackingDataset, HeatmapDataset, MultiviewHeatmapDataset)
            )
            imgaug_curr = data_module.dataset.imgaug_transform
            assert isinstance(imgaug_curr, iaa.Sequential)
            if len(imgaug_curr) == 1 and isinstance(imgaug_curr[0], iaa.Resize):
                # current augmentation just resizes; keep this
                self.data_module = data_module
            else:
                # create a simple resize-only augmentation pipeline for PCA
                # use the same resize dimensions as the original dataset
                dataset_old = data_module.dataset
                image_resize_height = dataset_old.image_resize_height
                image_resize_width = dataset_old.image_resize_width
                imgaug_new = iaa.Sequential([
                    iaa.Resize({'height': image_resize_height, 'width': image_resize_width})
                ])

                if isinstance(dataset_old, HeatmapDataset):
                    dataset_new = HeatmapDataset(
                        root_directory=dataset_old.root_directory,
                        csv_path=dataset_old.csv_path,
                        image_resize_height=dataset_old.image_resize_height,
                        image_resize_width=dataset_old.image_resize_width,
                        imgaug_transform=imgaug_new,
                        downsample_factor=dataset_old.downsample_factor,
                        do_context=dataset_old.do_context,
                    )
                elif isinstance(dataset_old, BaseTrackingDataset):
                    dataset_new = BaseTrackingDataset(
                        root_directory=dataset_old.root_directory,
                        csv_path=dataset_old.csv_path,
                        image_resize_height=dataset_old.image_resize_height,
                        image_resize_width=dataset_old.image_resize_width,
                        imgaug_transform=imgaug_new,
                        do_context=dataset_old.do_context,
                    )
                elif isinstance(dataset_old, MultiviewHeatmapDataset):
                    dataset_new = MultiviewHeatmapDataset(
                        root_directory=dataset_old.root_directory,
                        csv_paths=dataset_old.csv_paths,
                        view_names=dataset_old.view_names,
                        image_resize_height=dataset_old.image_resize_height,
                        image_resize_width=dataset_old.image_resize_width,
                        imgaug_transform=imgaug_new,
                        do_context=dataset_old.do_context,
                    )
                else:
                    raise NotImplementedError

                # rebuild data_module with new dataset
                if isinstance(data_module, UnlabeledDataModule):
                    data_module_new = UnlabeledDataModule(
                        dataset=dataset_new,
                        video_paths_list=data_module.video_paths_list,
                        train_batch_size=data_module.train_batch_size,
                        val_batch_size=data_module.val_batch_size,
                        test_batch_size=data_module.test_batch_size,
                        num_workers=data_module.num_workers,
                        train_probability=data_module.train_probability,
                        val_probability=data_module.val_probability,
                        train_frames=data_module.train_frames,
                        dali_config=data_module.dali_config,
                        torch_seed=data_module.torch_seed,
                    )
                    # data_module_new.setup() happens internally
                elif isinstance(data_module, BaseDataModule):
                    data_module_new = BaseDataModule(
                        dataset=dataset_new,
                        train_batch_size=data_module.train_batch_size,
                        val_batch_size=data_module.val_batch_size,
                        test_batch_size=data_module.test_batch_size,
                        num_workers=data_module.num_workers,
                        train_probability=data_module.train_probability,
                        val_probability=data_module.val_probability,
                        train_frames=data_module.train_frames,
                        torch_seed=data_module.torch_seed,
                    )
                else:
                    raise NotImplementedError

                self.data_module = data_module_new

        else:
            self.data_module = data_module

    @property
    def dataset_length(self) -> int:
        """Number of examples in the selected data split.

        Returns:
            Length of the ``train``, ``val``, or ``test`` dataset depending on ``self.cond``.
        """
        name = f'{self.cond}_dataset'
        return len(getattr(self.data_module, name))

    def get_loader(self) -> torch.utils.data.DataLoader | CombinedLoader:
        """Return the dataloader for the selected split.

        Returns:
            DataLoader or ``CombinedLoader`` corresponding to ``self.cond``.

        Raises:
            ValueError: if ``self.cond`` is not ``"train"``, ``"val"``, or ``"test"``.
        """
        if self.cond == 'train':
            return self.data_module.train_dataloader()  # type: ignore[return-value]
        if self.cond == 'val':
            return self.data_module.val_dataloader()
        if self.cond == 'test':
            return self.data_module.test_dataloader()
        raise ValueError(f'cond must be "train", "val", or "test", got {self.cond!r}')

    @staticmethod
    def verify_labeled_loader(
        loader: torch.utils.data.DataLoader | CombinedLoader,
    ) -> torch.utils.data.DataLoader:
        """Extract and return the labeled DataLoader from a potentially combined loader.

        Args:
            loader: either a plain ``DataLoader`` or a ``CombinedLoader`` containing labeled and
                unlabeled sub-loaders.

        Returns:
            The labeled ``DataLoader``.
        """
        if isinstance(loader, torch.utils.data.DataLoader):
            return loader
        # CombinedLoader wraps labeled + unlabeled; extract only the labeled one
        return loader.iterables['labeled']  # type: ignore[index]

    def iterate_over_dataloader(
        self, loader: torch.utils.data.DataLoader
    ) -> tuple[
        torch.Tensor,
        (
            Float[torch.Tensor, "num_examples 3 image_width image_height"]
            | Float[torch.Tensor, "num_examples frames 3 image_width image_height"]
            | None
        ),
    ]:
        """Iterate over a dataloader and collect keypoints (and optionally images).

        Args:
            loader: labeled dataloader to iterate over.

        Returns:
            Tuple of:
                - concatenated keypoints tensor of shape ``(num_examples, num_targets)``.
                - concatenated image tensor or ``None`` if ``self.extract_images`` is False.
        """
        keypoints_list = []
        images_list = []
        for _ind, batch in enumerate(loader):
            keypoints_list.append(batch['keypoints'])
            if self.extract_images:
                images_list.append(batch['images'])
        concat_keypoints = torch.cat(keypoints_list, dim=0)
        if self.extract_images:
            concat_images = torch.cat(images_list, dim=0)
        else:
            concat_images = None
        assert concat_keypoints.shape == (
            self.dataset_length,
            keypoints_list[0].shape[1],
        )
        return concat_keypoints, concat_images

    def __call__(
        self,
    ) -> tuple[
        torch.Tensor,
        (
            Float[torch.Tensor, "num_examples 3 image_width image_height"]
            | Float[torch.Tensor, "num_examples frames 3 image_width image_height"]
            | None
        ),
    ]:
        """Extract all keypoints (and optionally images) from the selected data split.

        Returns:
            Tuple of:
                - concatenated keypoints tensor of shape ``(num_examples, num_targets)``.
                - concatenated image tensor or ``None`` if ``self.extract_images`` is False.
        """
        loader = self.get_loader()
        loader = self.verify_labeled_loader(loader)
        return self.iterate_over_dataloader(loader)
