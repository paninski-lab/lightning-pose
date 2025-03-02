"""Classes to streamline data typechecking."""

from typing import TypedDict, Union

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchtyping import TensorType

# to ignore imports for sphix-autoapidoc
__all__ = [
    "BaseLabeledExampleDict",
    "HeatmapLabeledExampleDict",
    "MultiviewLabeledExampleDict",
    "MultiviewHeatmapLabeledExampleDict",
    "BaseLabeledBatchDict",
    "HeatmapLabeledBatchDict",
    "MultiviewLabeledBatchDict",
    "MultiviewHeatmapLabeledBatchDict",
    "UnlabeledBatchDict",
    "MultiviewUnlabeledBatchDict",
    "SemiSupervisedBatchDict",
    "SemiSupervisedHeatmapBatchDict",
    "SemiSupervisedDataLoaderDict",
]


class BaseLabeledExampleDict(TypedDict):
    """Return type when calling __getitem__() on BaseTrackingDataset."""
    images: Union[
        TensorType["RGB":3, "image_height", "image_width", float],
        TensorType["frames", "RGB":3, "image_height", "image_width", float],
    ]
    keypoints: TensorType["num_targets", float]
    bbox: TensorType["xyhw":4, float]
    idxs: int


class HeatmapLabeledExampleDict(BaseLabeledExampleDict):
    """Return type when calling __getitem__() on HeatmapTrackingDataset."""
    heatmaps: TensorType["num_keypoints", "heatmap_height", "heatmap_width", float]


class MultiviewLabeledExampleDict(TypedDict):
    """Return type when calling __getitem__() on MultiviewDataset."""
    images: Union[
        TensorType["num_views", "RGB":3, "image_height", "image_width", float],
        TensorType["num_views", "frames", "RGB":3, "image_height", "image_width", float],
    ]
    keypoints: TensorType["num_targets", float]
    bbox: TensorType["num_views", "xyhw":4, float]
    idxs: int
    num_views: int
    concat_order: list[str]
    view_names: list[str]


class MultiviewHeatmapLabeledExampleDict(MultiviewLabeledExampleDict):
    """Return type when calling __getitem__() on MultiviewHeatmapDataset."""
    heatmaps: TensorType["num_keypoints", "heatmap_height", "heatmap_width", float]


class BaseLabeledBatchDict(TypedDict):
    """Batch type for base labeled data."""
    images: Union[
        TensorType["batch", "RGB":3, "image_height", "image_width", float],
        TensorType["batch", "frames", "RGB":3, "image_height", "image_width", float],
    ]
    keypoints: TensorType["batch", "num_targets", float]
    bbox: TensorType["batch", "xyhw":4, float]
    idxs: TensorType["batch", int]


class HeatmapLabeledBatchDict(BaseLabeledBatchDict):
    """Batch type for heatmap labeled data."""
    heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width", float]


class MultiviewLabeledBatchDict(TypedDict):
    """Batch type for multiview labeled data."""
    images: Union[
        TensorType["batch", "num_views", "RGB":3, "image_height", "image_width", float],
        TensorType["batch", "num_views", "frames", "RGB":3, "image_height", "image_width", float],
    ]
    keypoints: TensorType["batch", "num_targets", float]
    bbox: TensorType["batch", "num_views * xyhw", float]
    idxs: TensorType["batch", int]
    num_views: TensorType["batch", int]
    concat_order: list  # [Tuple[str]]
    view_names: list  # [Tuple[str]]


class MultiviewHeatmapLabeledBatchDict(MultiviewLabeledBatchDict):
    """Batch type for multiview heatmap labeled data."""
    heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width", float]


class UnlabeledBatchDict(TypedDict):
    """Batch type for unlabeled data."""
    frames: TensorType["seq_len", "RGB":3, "image_height", "image_width", float]
    transforms: Union[
        TensorType["seq_len", "h":2, "w":3, float],
        TensorType["h":2, "w":3, float],
        TensorType["seq_len", "null":1, float],
        TensorType["null":1, float],
        torch.Tensor,
    ]
    # transforms shapes
    # (seq_len, 2, 3): different transform for each sequence
    # (2, 3): same transform for all returned frames/keypoints
    # (seq_len, 1): no transforms
    # (1,): no transforms
    # torch.Tensor: necessary, getting error about torch.AnnotatedAlias that I don't understand

    bbox: TensorType["seq_len", "xyhw":4, float]
    is_multiview: bool = False  # helps with downstream logic since isinstance fails on TypedDicts


class MultiviewUnlabeledBatchDict(TypedDict):
    """Batch type for multiview unlabeled data."""
    frames: TensorType["seq_len", "num_views", "RGB":3, "image_height", "image_width", float]
    transforms: Union[
        TensorType["num_views", "h":2, "w":3, float],
        TensorType["num_views", "null":1, "null":1, float],
        torch.Tensor,
    ]
    bbox: TensorType["seq_len", "num_views * xyhw", float]
    is_multiview: bool = True  # helps with downstream logic since isinstance fails on TypedDicts


class SemiSupervisedBatchDict(TypedDict):
    """Batch type for base labeled+unlabeled data."""

    labeled: BaseLabeledBatchDict | MultiviewLabeledBatchDict
    unlabeled: UnlabeledBatchDict | MultiviewUnlabeledBatchDict


class SemiSupervisedHeatmapBatchDict(TypedDict):
    """Batch type for heatmap labeled+unlabeled data."""

    labeled: HeatmapLabeledBatchDict | MultiviewHeatmapLabeledBatchDict
    unlabeled: UnlabeledBatchDict | MultiviewUnlabeledBatchDict


class SemiSupervisedDataLoaderDict(TypedDict):
    """Return type when calling train/val/test_dataloader() on semi-supervised models."""

    labeled: torch.utils.data.DataLoader
    unlabeled: DALIGenericIterator
