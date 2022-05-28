"""Data pipelines based on efficient video reading by nvidia dali package."""

import cv2
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types
import torch
from typeguard import typechecked
from typing import List, Optional, Union
from torchtyping import TensorType, patch_typeguard


from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD

_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

patch_typeguard()  # use before @typechecked. TODO: new, make sure it doesn't mess things

# cannot typecheck due to way pipeline_def decorator consumes additional args
@pipeline_def
def video_pipe(
    filenames: Union[List[str], str],
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
    step: int = 1,
    pad_last_batch: bool = False,
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
        initial_fill: size of the buffer that is used for random shuffling
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
        step=step,
        pad_sequences=pad_sequences,
        initial_fill=initial_fill,
        normalized=False,
        name=name,
        dtype=types.DALIDataType.FLOAT,
        pad_last_batch=pad_last_batch, # Important for context loaders

    )
    if resize_dims:
        video = fn.resize(video, size=resize_dims)
    # video pixel range is [0, 255]; transform it to [0, 1].
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

    def __init__(self, *kargs, **kwargs):

        # TODO: change name to "num_iters"
        # collect number of batches computed outside of class
        self.num_batches = kwargs.pop("num_batches", 1)

        # call parent
        super().__init__(*kargs, **kwargs)

    def __len__(self):
        return self.num_batches

    def __next__(self):
        out = super().__next__()
        return torch.tensor(
            out[0]["x"][0, :, :, :, :],  # should be (sequence_length, 3, H, W)
            dtype=torch.float,
        )  # careful: only valid for one sequence, i.e., batch size of 1.

# TODO: see if can be easily merged.
class ContextLightningWrapper(DALIGenericIterator):
    """wrapper around a DALI pipeline to get batches for ptl."""

    def __init__(self, *kargs, **kwargs):

        # collect number of batches computed outside of class
        self.num_batches = kwargs.pop("num_batches", 1)

        # call parent
        super().__init__(*kargs, **kwargs)

    def __len__(self):
        # TODO: careful here, this needs to be something different now
        return self.num_batches

    def __next__(self):
        out = super().__next__()
        return out[0]["x"]
    

@typechecked
class LitDaliWrapper(DALIGenericIterator):
    """wrapper around a DALI pipeline to get batches for ptl."""

    def __init__(self, *kargs, num_iters: int = 1, do_context: bool = False, **kwargs):
        """ wrapper around DALIGenericIterator to get batches for pl.
        Args: 
            num_iters: number of enumerations of dataloader (should be computed outside for now; should be fixed by lightning/dali teams)
            do_context: whether model/loader use 5-frame context or not
            """
        # TODO: add a case where we 
        self.num_iters = num_iters
        self.do_context = do_context

        # call parent
        super().__init__(*kargs, **kwargs)

    def __len__(self):
        return self.num_iters
    
    def _modify_output(self, out) -> Union[TensorType["sequence_length", 3, "image_width", "image_height"], TensorType["batch", 5, 3, "image_width", "image_height"]]:
        """ modify output to be torch tensor. 
        looks different for context and non-context."""
        if self.do_context == False:
            return torch.tensor(
            out[0]["x"][0, :, :, :, :],  # should be (sequence_length, 3, H, W)
            dtype=torch.float,
        )  # careful: only valid for one sequence, i.e., batch size of 1.
        else:
            return out[0]["x"]

    def __next__(self):
        out = super().__next__()
        return self._modify_output(out)