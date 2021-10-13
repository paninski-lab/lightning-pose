import torch
import sklearn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typeguard import typechecked
from typing import List, Optional
import cv2
import os
import numpy as np

# TODO: use the below in the lightning wrapper
def count_frames(video_full_path: str) -> int:
    """counts the number of frames in a video using opencv.

    Args:
        video_full_path (str): path to a single video

    Returns:
        int: number of frames in the video
    """
    video = cv2.VideoCapture(video_full_path)
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


## DALI Defaults
# statistics of imagenet dataset on which the resnet was trained
# see https://pytorch.org/vision/stable/models.html
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
_SEQUENCE_LENGTH_UNSUPERVISED = 16
_INITIAL_PREFETCH_SIZE = 16


@pipeline_def
def video_pipe(
    filenames: list,
    resize_dims: Optional[List[int]] = None,
    random_shuffle: Optional[bool] = False,
    normalization_mean: Optional[List[float]] = _IMAGENET_MEAN,
    normalization_std: Optional[List[float]] = _IMAGENET_STD,
    device: Optional[str] = _DALI_DEVICE,
    sequence_length: int = _SEQUENCE_LENGTH_UNSUPERVISED,
):  # TODO: what does it return? more typechecking
    video = fn.readers.video(
        device=device,
        filenames=filenames,
        sequence_length=sequence_length,
        random_shuffle=random_shuffle,
        initial_fill=_INITIAL_PREFETCH_SIZE,
        normalized=False,
        dtype=types.DALIDataType.FLOAT,
    )
    if resize_dims:
        video = fn.resize(video, size=resize_dims)
    video = (
        video / 255.0
    )  # original videos (at least Rick's) range from 0-255. transform it to 0,1. happens naturally in the torchvision transform to tensor.
    transform = fn.crop_mirror_normalize(
        video,
        output_layout="FCHW",
        mean=normalization_mean,
        std=normalization_std,
    )
    return transform


class LightningWrapper(DALIGenericIterator):
    """wrapper around a DALI pipeline to get batches for ptl.

    Args:
        DALIGenericIterator ([type]): [description]
    """

    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)

    def __len__(self):  # just to avoid ptl err check
        return 1  # num frames = len * batch_size; TODO: determine actual length of vid

    def __next__(self):
        out = super().__next__()
        return torch.tensor(
            out[0]["x"][
                0, :, :, :, :
            ],  # should be batch_size, W, H, 3. TODO: careful: valid for one sequence, i.e., batch size of 1.
            dtype=torch.float,  # , device="cuda"
        )
