from __future__ import annotations
from typing import Protocol, Callable, TypeAlias, Tuple, Union
from typing import TYPE_CHECKING


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


class ModelComponentContainer(Protocol):
    def __init__(self, cfg: DictConfig): ...

    def get_imgaug_transform(self) -> iaa.Sequential: ...

    def get_dataset(self) -> torch.utils.data.Dataset: ...

    def get_split_datasets(self) -> tuple[Subset, Subset, Subset]: ...

    def get_dataloader_factory(self) -> Callable[[str], torch.utils.data.DataLoader]:
        """stage -> dataloader"""
        ...

    def get_predict_dali_dataloader_factory(
        self,
    ) -> Callable[[str], torch.utils.data.DataLoader]:
        """filename -> predict dataloader"""
        ...

    def get_combined_dataloader_factory(
        self,
    ) -> Callable[[Subset], torch.utils.data.DataLoader]: ...

    def get_data_module(self) -> pl.LightningDataModule: ...

    def get_model(self) -> HeatmapTracker: ...
