from __future__ import annotations

from functools import wraps
from typing import Protocol, Callable, TypeAlias, Tuple, Union
from typing import TYPE_CHECKING

from lightning_pose.utils import scripts
from lightning_pose.utils.mega_factory import ModelComponentContainer

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig
    import imgaug.augmenters as iaa
    import lightning.pytorch as pl
    from torchtyping import TensorType

    from torch.utils.data import Subset

    from lightning_pose.losses.factory import LossFactory
    from lightning_pose.models import HeatmapTracker

    DataExtractorOutput: TypeAlias = Tuple[
        TensorType["num_examples", Any],
        Union[
            TensorType["num_examples", 3, "image_width", "image_height"],
            TensorType["num_examples", "frames", 3, "image_width", "image_height"],
            None,
        ],
    ]


def cached(func):
    """Cache a method's result on the instance itself."""

    @wraps(func)
    def wrapper(self):
        cache_attr = "_instance_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)
        key = func.__name__
        if key not in cache:
            cache[key] = func(self)
        return cache[key]

    return wrapper


class ModelComponentContainerImpl(Protocol):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    # Intentionally not cached since this can be mutated later.
    def get_imgaug_transform(self) -> iaa.Sequential:
        return scripts.get_imgaug_transform(cfg=self.cfg)

    @cached
    def get_dataset(self) -> torch.utils.data.Dataset:
        imgaug_transform = self.get_imgaug_transform()
        return scripts.get_dataset(
            cfg=self.cfg,
            data_dir=self.cfg.data.data_dir,
            imgaug_transform=imgaug_transform,
        )

    @cached
    def get_split_datasets(self) -> tuple[Subset, Subset, Subset]:
        dataset = self.get_dataset()
        return scripts.get_split_datasets(cfg=self.cfg, dataset=dataset)

    @cached
    def get_dataloader_factory(self) -> Callable[[str], torch.utils.data.DataLoader]:
        """stage -> dataloader"""
        splits = self.get_split_datasets()
        return scripts.get_dataloader_factory(cfg=self.cfg, splits=splits)

    @cached
    def get_predict_dali_dataloader_factory(
        self,
    ) -> Callable[[str], torch.utils.data.DataLoader]:
        """filename -> predict dataloader"""
        ...

    @cached
    def get_combined_dataloader_factory(
        self,
    ) -> Callable[[Subset], torch.utils.data.DataLoader]: ...

    @cached
    def get_data_module(self) -> pl.LightningDataModule:
        dataset = self.get_dataset()
        # Build splits and labeled dataloader factory, then construct data module
        splits = scripts.get_split_datasets(cfg=self.cfg, dataset=dataset)
        dataloader_factory = scripts.get_dataloader_factory(
            cfg=self.cfg, dataset=dataset, splits=splits
        )
        return scripts.get_data_module(
            cfg=self.cfg, dataset=dataset, dataloader_factory=dataloader_factory
        )

    @cached
    def get_loss_factories(self) -> dict[str, LossFactory | None]:
        data_module = self.get_data_module()
        return scripts.get_loss_factories(cfg=self.cfg, data_module=data_module)

    @cached
    def get_model(self) -> HeatmapTracker:
        data_module = self.get_data_module()
        loss_factories = self.get_loss_factories()
        return scripts.get_model(
            cfg=self.cfg, loss_factories=loss_factories, data_module=data_module
        )

    # Trainer dependencies

    def get_steps_for_epoch(self):
        return scripts.calculate_steps_per_epoch(self.get_data_module())

    def get_logger(self):
        return scripts.get_training_logger(cfg=self.cfg)

    def get_callbacks(self):
        return scripts.get_callbacks(cfg=self.cfg)

    def get_trainer(self) -> pl.Trainer:
        return scripts.get_trainer(
            cfg=self.cfg,
            steps_per_epoch=self.get_steps_for_epoch(),
            logger=self.get_logger(),
            callbacks=self.get_callbacks(),
        )


test: ModelComponentContainer = ModelComponentContainerImpl(None)
