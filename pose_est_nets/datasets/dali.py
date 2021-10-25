"""Data pipelines based on efficient video reading by nvidia dali package."""

import cv2
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.types as types
import os
import sklearn
import torch
from typeguard import typechecked
from typing import List, Optional, Union

# DALI Defaults
# statistics of imagenet dataset on which the resnet was trained
# see https://pytorch.org/vision/stable/models.html
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
_SEQUENCE_LENGTH_UNSUPERVISED = 16
_INITIAL_PREFETCH_SIZE = 16


# TODO: use the below in the lightning wrapper
@typechecked
def count_frames(video_full_path: str) -> int:
    """Count the number of frames in a video using opencv.

    Args:
        video_full_path: absolute path to a single video

    Returns:
        Number of frames in the video

    """
    video = cv2.VideoCapture(video_full_path)
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


# cannot typecheck due to way pipeline_def decorator consumes additional args
@pipeline_def
def video_pipe(
    filenames: Union[list[str], str],
    resize_dims: Optional[List[int]] = None,
    random_shuffle: bool = False,
    seed: int = 123456,
    sequence_length: int = 16,
    pad_sequences: bool = True,
    initial_fill: int = 16,
    normalization_mean: List[float] = _IMAGENET_MEAN,
    normalization_std: List[float] = _IMAGENET_STD,
    device: str = _DALI_DEVICE,
    name: str = "reader",
    # arguments consumed by decorator:
    # batch_size,
    # num_threads,
    # device_id
) -> Pipeline:
    """Generic video reader pipeline that loads videos, resizes, and normalizes.

    Args:
        filenames: list of absolute paths of video files to feed through
            pipeline
        resize_dims: [height, width] to resize raw frames
        random_shuffle: True to grab random batches of frames from videos;
            False to sequential read
        seed: random seed when `random_shuffle` is True
        sequence_length: number of frames to load per sequence
        pad_sequences: allows creation of incomplete sequences if there is an
            insufficient number of frames at the very end of the video
        initial_fill:
        normalization_mean: mean values in (0, 1) to subtract from each channel
        normalization_std: standard deviation values to subtract from each
            channel
        device: "cpu" | "gpu"
        name: pipeline name, used to string together DataNode elements

    Returns:
        pipeline object to be fed to DALIGenericIterator

    """
    video = fn.readers.video(
        device=device,
        filenames=filenames,
        random_shuffle=random_shuffle,
        seed=seed,
        sequence_length=sequence_length,
        pad_sequences=pad_sequences,
        initial_fill=initial_fill,
        normalized=False,
        name=name,
        dtype=types.DALIDataType.FLOAT,
    )
    if resize_dims:
        video = fn.resize(video, size=resize_dims)
    # videos range from 0-255. transform it to 0,1.
    # happens naturally in the torchvision transform to tensor.
    video = video / 255.0
    # permute dimensions and normalize to imagenet statistics
    transform = fn.crop_mirror_normalize(
        video,
        output_layout="FCHW",
        mean=normalization_mean,
        std=normalization_std,
    )
    return transform


class LightningWrapper(DALIGenericIterator):
    """wrapper around a DALI pipeline to get batches for ptl."""

    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)

    def __len__(self):  # just to avoid ptl err check
        # TODO: determine actual length of vid
        return 1  # num frames = len * batch_size

    def __next__(self):
        out = super().__next__()
        return torch.tensor(
            out[0]["x"][0, :, :, :, :],  # should be batch_size, W, H, 3.
            dtype=torch.float,
        )  # TODO: careful: valid for one sequence, i.e., batch size of 1.
