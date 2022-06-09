"""Data pipelines based on efficient video reading by nvidia dali package."""

import cv2
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types
import torch
import numpy as np
from typeguard import typechecked
from typing import List, Dict, Optional, Union, Literal
from torchtyping import TensorType, patch_typeguard


from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.utils import count_frames

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

    def __init__(self, *kargs, eval_mode: Literal["train", "predict"], num_iters: int = 1, do_context: bool = False, context_sequences_successive: bool = False, **kwargs):
        """ wrapper around DALIGenericIterator to get batches for pl.
        Args: 
            num_iters: number of enumerations of dataloader (should be computed outside for now; should be fixed by lightning/dali teams)
            do_context: whether model/loader use 5-frame context or not
            """
        # TODO: add a case where we 
        self.num_iters = num_iters
        self.do_context = do_context
        self.eval_mode = eval_mode
        self.context_sequences_successive = context_sequences_successive

        # call parent
        super().__init__(*kargs, **kwargs)

    def __len__(self):
        return self.num_iters
    
    def _modify_output(self, out) -> Union[TensorType["sequence_length", 3, "image_height", "image_width"], TensorType["batch", 5, 3, "image_height", "image_width"]]:
        """ modify output to be torch tensor. 
        looks different for context and non-context."""
        if self.do_context == False:
            return torch.tensor(
            out[0]["x"][0, :, :, :, :],  # should be (sequence_length, 3, H, W)
            dtype=torch.float,
        )  # careful: only valid for one sequence, i.e., batch size of 1.
        else:
            if self.eval_mode == "predict":
                return out[0]["x"]
            else: # train with context. pipeline is like for "base" model. but we reshape images. assume batch_size=1
                if self.context_sequences_successive:
                    raw_out = torch.tensor(
                    out[0]["x"][0, :, :, :, :],  # should be (sequence_length, 3, H, W)
                    dtype=torch.float)
                    # reshape to (5, 3, H, W)
                    return get_context_from_seq(img_seq=raw_out, context_length=5)
                else: # grabbing independent 5-frame sequences
                    return out[0]["x"]

    def __next__(self):
        out = super().__next__()
        return self._modify_output(out)

def get_context_from_seq(
    img_seq: TensorType["sequence_length", 3, "image_height", "image_width"],
    context_length: int,
) -> TensorType["sequence_length", "context_length", "rgb": 3, "image_height", "image_width"]:
    pass
    # our goal is to extract 5-frame sequences from this sequence
    img_shape = img_seq.shape[1:] # e.g., (3, H, W)
    seq_len = img_seq.shape[0] # how many images in batch
    train_seq = torch.zeros(
        (seq_len, context_length, *img_shape), device=img_seq.device
    )
    # define pads: start pad repeats the zeroth image twice. end pad repeats the last image twice.
    # this is to give padding for the first and last frames of the sequence
    pad_start = torch.tile(img_seq[0].unsqueeze(0), (2, 1, 1, 1))
    pad_end = torch.tile(img_seq[-1].unsqueeze(0), (2, 1, 1, 1))
    # pad the sequence
    padded_seq = torch.cat((pad_start, img_seq, pad_end), dim=0)
    # padded_seq = torch.cat((two_pad, img_seq, two_pad), dim=0)
    for i in range(seq_len):
        # extract 5-frame sequences from the padded sequence
        train_seq[i] = padded_seq[i : i + context_length]
    return train_seq

class PrepareDALI(object):
    
    """
    All the DALI stuff in one place.
    TODO: make sure the order of args when you mix is valid.
    needs to know about context
    TODO: consider changing LightningWrapper args from num_batches to num_iter
    Big picture: this will initialize the pipes and dataloaders which will be called only in Trainer.predict().
    Another option -- make this valid for Trainer.train() as well, so the unlabeled stuff will be initialized here.
    Thoughts: define a dict with args for pipe and data loader, per condition.
    """
    def __init__(self, train_stage: Literal["predict", "train"], model_type: Literal["base", "context"], filenames: List[str], context_sequences_successive: bool = False, dali_params: Optional[dict] = None):
        self.train_stage = train_stage
        self.model_type = model_type
        self.dali_params = dali_params
        self.filenames = filenames
        self.frame_count = count_frames(self.filenames)
        self.context_sequences_successive = context_sequences_successive
        self._pipe_dict: dict = self._setup_pipe_dict(self.filenames)

    @property
    def num_iters(self) -> int:
        # count frames
        # "how many times should we enumerate the data loader?""
          # sum across vids
        pipe_dict = self._pipe_dict[self.train_stage][self.model_type]
        if self.model_type == "base":
            return int(np.ceil(self.frame_count / (pipe_dict["sequence_length"])))
        elif self.model_type == "context":
            if pipe_dict["step"] == 1: # context_sequences_successive
                return int(np.ceil(self.frame_count / (pipe_dict["batch_size"])))
            elif pipe_dict["step"] == pipe_dict["sequence_length"]: # context_sequences_successive = False
                # taking the floor because during training we don't care about missing the last non-full batch. we prefer having fewer batches but valid.
                return int(np.floor(self.frame_count / (pipe_dict["batch_size"] * pipe_dict["sequence_length"])))
            else:
                raise NotImplementedError
    
    def _setup_pipe_dict(self, filenames: List[str]) -> Dict[str, dict]:
        """all of the pipe() args in one place"""
        dict_args = {}
        dict_args["predict"] = {"context": None, "base": None}
        dict_args["train"] = {"context": None, "base": None}
        
        # base (vanilla single-frame model), 
        # train pipe args 
        dict_args["train"]["base"] = {"filenames": filenames, "resize_dims": [256, 256], 
        "sequence_length": 16, "step": 16, "batch_size": 1, 
        "seed": 123456, "num_threads": 4, "device_id": 0, 
        "random_shuffle": True, "device": "gpu"}

        # base (vanilla model), predict pipe args 
        dict_args["predict"]["base"] = {"filenames": filenames, "resize_dims": [256, 256], 
        "sequence_length": 16, "step": 16, "batch_size": 1, 
        "seed": 123456, "num_threads": 4, "device_id": 0, 
        "random_shuffle": False, "device": "gpu", "name": "reader", 
        "pad_sequences": True}

        # context (five-frame) model
        # predict pipe args
        dict_args["predict"]["context"] = {"filenames": filenames, "resize_dims": [256, 256], 
        "sequence_length": 5, "step": 1, "batch_size": 4, 
        "num_threads": 4, 
        "device_id": 0, "random_shuffle": False, 
        "device": "gpu", "name": "reader", "seed": 123456,
        "pad_sequences": True, "pad_last_batch": True}

        # train pipe args
        if self.context_sequences_successive:
            # grab a sequence of 8 frames and reshape it internally (sequence length will effectively be multiplied by 5)
            dict_args["train"]["context"] = {"filenames": filenames, "resize_dims": [256, 256], 
            "sequence_length": 8, "step": 8, "batch_size": 1, 
            "seed": 123456, "num_threads": 4, "device_id": 0, 
            "random_shuffle": True, "device": "gpu"}
        else:
            dict_args["train"]["context"] = {"filenames": filenames, "resize_dims": [256, 256], 
            "sequence_length": 5, "step": 5, "batch_size": 8, 
            "num_threads": 4, 
            "device_id": 0, "random_shuffle": True, 
            "device": "gpu", "name": "reader", "seed": 123456,
            "pad_sequences": True, "pad_last_batch": False}
            # our floor above should prevent us from getting to the very final batch.
        
        return dict_args
    
    def _get_dali_pipe(self):
        """
        Return a DALI pipe with predefined args.
        """

        pipe_args = self._pipe_dict[self.train_stage][self.model_type]
        pipe = video_pipe(**pipe_args)
        return pipe
    
    def _setup_dali_iterator_args(self) -> LitDaliWrapper:
        """ builds args for Lightning iterator"""
        dict_args = {}
        dict_args["predict"] = {"context": None, "base": None}
        dict_args["train"] = {"context": None, "base": None}

        # base models (single-frame)
        dict_args["train"]["base"] = {"num_iters": self.num_iters, "eval_mode": "train", "do_context": False, "output_map": ["x"], "last_batch_policy": LastBatchPolicy.PARTIAL, "auto_reset": True} # taken from datamodules.py. 
        dict_args["predict"]["base"] = {"num_iters": self.num_iters, "eval_mode": "predict", "do_context": False, "output_map": ["x"], "last_batch_policy": LastBatchPolicy.FILL, "last_batch_padded": False, "auto_reset": False, "reader_name": "reader"}

        # 5-frame context models
        dict_args["train"]["context"] = {"num_iters": self.num_iters, "eval_mode": "train", "do_context": True, "output_map": ["x"], "last_batch_policy": LastBatchPolicy.PARTIAL, "auto_reset": True} # taken from datamodules.py. only difference is that we need to do context
        dict_args["predict"]["context"] = {"num_iters": self.num_iters, "eval_mode": "predict", "do_context": True, "output_map": ["x"], "last_batch_policy": LastBatchPolicy.PARTIAL, "last_batch_padded": False, "auto_reset": False, "reader_name": "reader"}

        return dict_args
    
    def __call__(self) -> LitDaliWrapper:
        """
        Returns a LightningWrapper object.
        """
        pipe = self._get_dali_pipe()
        args = self._setup_dali_iterator_args()
        return LitDaliWrapper(pipe, **args[self.train_stage][self.model_type])
    