"""Data pipelines based on efficient video reading by nvidia dali package."""

import os
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from omegaconf import DictConfig

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.utils import MultiviewUnlabeledBatchDict, UnlabeledBatchDict, count_frames

# to ignore imports for sphix-autoapidoc
__all__ = [
    "video_pipe",
    "LitDaliWrapper",
    "PrepareDALI",
]


# cannot typecheck due to way pipeline_def decorator consumes additional args
@pipeline_def
def video_pipe(
    filenames: Union[List[str], str],
    resize_dims: Optional[List[int]] = None,
    random_shuffle: bool = False,
    sequence_length: int = 16,
    pad_sequences: bool = True,
    initial_fill: int = 16,
    normalization_mean: List[float] = _IMAGENET_MEAN,
    normalization_std: List[float] = _IMAGENET_STD,
    name: str = "reader",
    step: int = 1,
    pad_last_batch: bool = False,
    imgaug: str = "default",
    skip_vfr_check: bool = True,
    # arguments consumed by decorator:
    # batch_size,
    # num_threads,
    # device_id
) -> tuple:
    """Generic video reader pipeline that loads videos, resizes, augments, and normalizes.

    Args:
        filenames: list of absolute paths of video files to feed through pipeline
        resize_dims: [height, width] to resize raw frames
        random_shuffle: True to grab random batches of frames from videos; False to sequential read
        seed: random seed when `random_shuffle` is True
        sequence_length: number of frames to load per sequence
        pad_sequences: allows creation of incomplete sequences if there is an insufficient number
            of frames at the very end of the video
        initial_fill: size of the buffer that is used for random shuffling
        normalization_mean: mean values in (0, 1) to subtract from each channel
        normalization_std: standard deviation values to subtract from each channel
        name: pipeline name, used to string together DataNode elements
        step: number of frames to advance on each read
        pad_last_batch:
        imgaug: string identifying which imgaug pipeline to use; "default", "dlc", "dlc-top-down"
        skip_vfr_check: don't check for variable frame rates, can throw errors with small diffs

    Returns:
        pipeline object to be fed to DALIGenericIterator
        data augmentation matrix (returned so that geometric transforms can be undone)
        size of video frames, used for bbox

    """
    # turn all inputs into a list of list of strings to be most general
    # first list: over views (might only be one)
    # second list: over videos/sessions
    if isinstance(filenames, list) and isinstance(filenames[0], str):
        filenames = [filenames]
    elif isinstance(filenames, str):
        filenames = [[filenames]]

    assert isinstance(filenames, list) and isinstance(filenames[0], list)

    # loop over views (can be only one)
    frames_list = []
    transform_list = []
    orig_size_list = []
    for f, filename_list in enumerate(filenames):
        video = fn.readers.video(
            device="gpu",
            filenames=filename_list,
            random_shuffle=random_shuffle,
            sequence_length=sequence_length,
            step=step,
            pad_sequences=pad_sequences,
            initial_fill=initial_fill,
            normalized=False,
            name=f"{name}_{f}",
            dtype=types.DALIDataType.FLOAT,
            pad_last_batch=pad_last_batch,  # Important for context loaders
            file_list_include_preceding_frame=True,  # to get rid of dali warnings
            skip_vfr_check=skip_vfr_check,
        )
        orig_size = fn.shapes(video)
        if resize_dims:
            video = fn.resize(video, size=resize_dims)
        if imgaug == "dlc" or imgaug == "dlc-top-down":
            size = (resize_dims[0] / 2, resize_dims[1] / 2)
            center = size  # / 2
            # rotate + scale
            angle = fn.random.uniform(range=(-10, 10))
            transform = fn.transforms.rotation(angle=angle, center=center)
            scale = fn.random.uniform(range=(0.8, 1.2), shape=2)
            transform = fn.transforms.scale(transform, scale=scale, center=center)
            video = fn.warp_affine(video, matrix=transform, fill_value=0, inverse_map=False)
            # brightness contrast:
            contrast = fn.random.uniform(range=(0.75, 1.25))
            brightness = fn.random.uniform(range=(0.75, 1.25))
            video = fn.brightness_contrast(video, brightness=brightness, contrast=contrast)
            # # shot noise
            factor = fn.random.uniform(range=(0.0, 10.0))
            video = fn.noise.shot(video, factor=factor)
            # jpeg compression
            # quality = fn.random.uniform(range=(50, 100), dtype=types.INT32)
            # video = fn.jpeg_compression_distortion(video, quality=quality)
        else:
            # choose arbitrary scalar (rather than a matrix) so that downstream operations know
            # there is no geometric transforms to undo
            transform = np.array([-1])
        # video pixel range is [0, 255]; transform it to [0, 1].
        # happens naturally in the torchvision transform to tensor.
        video = video / 255.0
        # permute dimensions and normalize to imagenet statistics
        frames = fn.crop_mirror_normalize(
            video,
            output_layout="FCHW",
            mean=normalization_mean,
            std=normalization_std,
        )
        frames_list.append(frames)
        transform_list.append(transform)
        orig_size_list.append(orig_size)

    return (*frames_list, *transform_list, *orig_size_list)


class LitDaliWrapper(DALIGenericIterator):
    """wrapper around a DALI pipeline to get batches for ptl."""

    def __init__(
        self,
        *args,
        eval_mode: Literal["train", "predict"],
        num_iters: int = 1,
        do_context: bool = False,
        **kwargs
    ) -> None:
        """Wrapper around DALIGenericIterator to get batches for pl.

        Args:
            eval_mode
            num_iters: number of enumerations of dataloader (should be computed outside for now;
                should be fixed by lightning/dali teams)
            do_context: whether model/loader use 5-frame context or not

        """
        self.num_iters = num_iters
        self.do_context = do_context
        self.eval_mode = eval_mode
        self.batch_sampler = 1  # hack to get around DALI-ptl issue
        # call parent
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self.num_iters

    @staticmethod
    def _dali_output_to_tensors(
        batch: list
    ) -> Union[UnlabeledBatchDict, MultiviewUnlabeledBatchDict]:

        # always batch_size=1

        if len(batch[0].keys()) == 3:  # single view pipeline

            # shape (sequence_length, 3, H, W)
            frames = batch[0]["frames"][0, :, :, :, :]
            # shape (1,) or (2, 3)
            transforms = batch[0]["transforms"][0]
            # get frame size
            if batch[0]["frame_size"][0, -1] == 3:  # order is seq_len,H,W,C
                height = batch[0]["frame_size"][0, 1]
                width = batch[0]["frame_size"][0, 2]
            else:  # order is seq_len,C,H,W
                height = batch[0]["frame_size"][0, 2]
                width = batch[0]["frame_size"][0, 3]
            bbox = torch.tensor([0, 0, height, width], device=frames.device).repeat(
                (frames.shape[0], 1))

            return UnlabeledBatchDict(
                frames=frames, transforms=transforms, bbox=bbox, is_multiview=False,
            )

        else:  # multiview pipeline

            # final shape: ("seq_len", "num_views", "RGB":3, "image_height", "image_width")
            frames = torch.stack(
                [batch[0][key][0, :, :, :, :] for key in batch[0].keys() if
                 "transforms" not in key and "frame_size" not in key],
                dim=1,
            )

            # final shape: ("num_views", "h":2, "w":3)
            transforms = torch.stack(
                [batch[0][key] for key in batch[0].keys() if "transforms" in key],
                dim=0,
            )

            # final shape: ("seq_len", "num_views * xyhw")
            bbox = torch.cat([
                torch.tensor([
                        0,
                        0,
                        batch[0][key][0, 1],
                        batch[0][key][0, 2],
                    ],
                    device=frames.device
                ) for key in batch[0].keys() if "frame_size" in key
            ], dim=0).repeat(frames.shape[0], 1)

            return MultiviewUnlabeledBatchDict(
                frames=frames, transforms=transforms, bbox=bbox, is_multiview=True,
            )

    def __next__(self) -> Union[UnlabeledBatchDict, MultiviewUnlabeledBatchDict]:
        batch = super().__next__()
        return self._dali_output_to_tensors(batch=batch)


class PrepareDALI(object):
    """All the DALI stuff in one place.

    Big picture: this will initialize the pipes and dataloaders for both training and prediction.

    """

    def __init__(
        self,
        train_stage: Literal["predict", "train"],
        model_type: Literal["base", "context"],
        filenames: Union[List[str], List[List[str]]],
        resize_dims: List[int],
        dali_config: Union[dict, DictConfig] = None,
        imgaug: Optional[str] = "default",
        num_threads: int = 1,
    ) -> None:

        # determine if we have a multiview pipeline
        if isinstance(filenames, list) and isinstance(filenames[0], list):
            self.multiview = True
        else:
            self.multiview = False

        # make sure `filenames` is a list of existing video files
        if isinstance(filenames, list) and isinstance(filenames[0], str):
            filenames = [filenames]
        for view_list in filenames:
            for vid in view_list:
                if not os.path.exists(vid) or not os.path.isfile(vid):
                    raise FileNotFoundError(f"{vid} is not a video file!")

        self.train_stage = train_stage
        self.model_type = model_type
        self.filenames = filenames
        self.resize_dims = resize_dims
        self.dali_config = dali_config
        self.num_threads = num_threads
        self.frame_count = count_frames(self.filenames)
        self._pipe_dict: dict = self._setup_pipe_dict(self.filenames, imgaug)

    @property
    def num_iters(self) -> int:
        # count frames
        # "how many times should we enumerate the data loader?"
        # sum across vids
        pipe_dict = self._pipe_dict[self.train_stage][self.model_type]
        if self.model_type == "base":
            return int(np.ceil(self.frame_count / (pipe_dict["sequence_length"])))
        elif self.model_type == "context":
            if pipe_dict["step"] == 1:  # 0-5, 1-6, 2-7, 3-8, 4-9 ...
                return int(np.ceil(self.frame_count / (pipe_dict["batch_size"])))
            elif pipe_dict["step"] == pipe_dict["sequence_length"]:
                # taking the floor because during training we don't care about missing the last
                # non-full batch. we prefer having fewer batches but valid.
                return int(
                    np.floor(
                        self.frame_count / (pipe_dict["batch_size"] * pipe_dict["sequence_length"])
                    )
                )
            elif (pipe_dict["batch_size"] == 1) and (
                pipe_dict["step"] == (pipe_dict["sequence_length"] - 4)
            ):
                # the case of prediction with a single sequence at a time and internal model
                # reshapes
                if pipe_dict["step"] <= 0:
                    raise ValueError(
                        "step cannot be 0, please modify "
                        "cfg.dali.context.predict.sequence_length to be > 4"
                    )
                # remove the first sequence
                data_except_first_batch = self.frame_count - pipe_dict["sequence_length"]
                # calculate how many "step"s are needed to get at least to the end
                # count back the first sequence
                num_iters = int(np.ceil(data_except_first_batch / pipe_dict["step"])) + 1
                return num_iters
            else:
                raise NotImplementedError

    def _setup_pipe_dict(
        self,
        filenames: Union[List[str], List[List[str]]],
        imgaug: str,
    ) -> Dict[str, dict]:
        """All of the pipeline args in one place."""
        # When running with multiple GPUs, the LOCAL_RANK variable correctly
        # contains the DDP Local Rank, which is also the cuda device index.
        device_id = int(os.environ.get("LOCAL_RANK", "0"))

        dict_args = {
            "predict": {"context": {}, "base": {}},
            "train": {"context": {}, "base": {}},
        }
        gen_cfg = self.dali_config["general"]

        # base (vanilla single-frame model), train pipe args
        base_train_cfg = self.dali_config["base"]["train"]
        dict_args["train"]["base"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": base_train_cfg["sequence_length"],
            "step": base_train_cfg["sequence_length"],
            "batch_size": 1,
            # Multi-GPU strategy is to have each GPU randomize differently.
            "seed": gen_cfg["seed"] + device_id,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": True,
            "imgaug": imgaug,
        }

        # base (vanilla single-frame model), predict pipe args
        base_pred_cfg = self.dali_config["base"]["predict"]
        dict_args["predict"]["base"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": base_pred_cfg["sequence_length"],
            "step": base_pred_cfg["sequence_length"],
            "batch_size": 1,
            # Multi-GPU strategy is to have each GPU randomize differently.
            "seed": gen_cfg["seed"] + device_id,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": False,
            "name": "reader",
            "pad_sequences": True,
            "imgaug": "default",  # no imgaug when predicting
        }

        # context (five-frame) model, predict pipe args
        context_pred_cfg = self.dali_config["context"]["predict"]
        dict_args["predict"]["context"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": context_pred_cfg["sequence_length"],
            "step": context_pred_cfg["sequence_length"] - 4,
            "batch_size": 1,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": False,
            "name": "reader",
            # Multi-GPU strategy is to have each GPU randomize differently.
            "seed": gen_cfg["seed"] + device_id,
            "pad_sequences": True,
            # "pad_last_batch": True,
            "imgaug": "default",  # no imgaug when predicting
        }

        # context (five-frame) model, train pipe args
        # grab a single sequence of frames, will resize into 5-frame chunks at the
        # representation level inside BaseFeatureExtractor
        # note: reusing the batch size argument
        context_train_cfg = self.dali_config["context"]["train"]
        dict_args["train"]["context"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": context_train_cfg["batch_size"],
            "step": context_train_cfg["batch_size"],
            "batch_size": 1,
            # Multi-GPU strategy is to have each GPU randomize differently.
            "seed": gen_cfg["seed"] + device_id,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": True,
            "imgaug": imgaug,
        }
        # our floor above should prevent us from getting to the very final batch.

        return dict_args

    def _get_dali_pipe(self):
        """
        Return a DALI pipe with predefined args.
        """

        pipe_args = self._pipe_dict[self.train_stage][self.model_type]
        pipe = video_pipe(**pipe_args)
        return pipe

    def _setup_dali_iterator_args(self) -> dict:
        """Builds args for Lightning iterator.

        If you want to extract more outputs from DALI, e.g., optical flow, you should also add
        this in the "output_map" arg

        """
        dict_args = {
            "predict": {"context": {}, "base": {}},
            "train": {"context": {}, "base": {}},
        }

        if self.multiview:
            # video pipeline returns a big tuple: frame, transforms, and frame size for each view
            output_map = \
                [f"frames_{i}" for i in range(len(self.filenames))] \
                + [f"transforms_{i}" for i in range(len(self.filenames))] \
                + [f"frame_size_{i}" for i in range(len(self.filenames))]
        else:
            output_map = ["frames", "transforms", "frame_size"]

        # base models (single-frame)
        dict_args["train"]["base"] = {
            "num_iters": self.num_iters,
            "eval_mode": "train",
            "do_context": False,
            "output_map": output_map,
            "last_batch_policy": LastBatchPolicy.PARTIAL,
            "auto_reset": True,
        }
        dict_args["predict"]["base"] = {
            "num_iters": self.num_iters,
            "eval_mode": "predict",
            "do_context": False,
            "output_map": output_map,
            "last_batch_policy": LastBatchPolicy.FILL,
            "last_batch_padded": False,
            "auto_reset": False,
            # if we have multiple readers, if we select only 1 here there's an error
            "reader_name": "reader_0" if not self.multiview else None,
        }

        # 5-frame context models
        dict_args["train"]["context"] = {
            "num_iters": self.num_iters,
            "eval_mode": "train",
            "do_context": True,
            "output_map": output_map,
            "last_batch_policy": LastBatchPolicy.PARTIAL,
            "auto_reset": True,
        }  # taken from datamodules.py. only difference is that we need to do context
        dict_args["predict"]["context"] = {
            "num_iters": self.num_iters,
            "eval_mode": "predict",
            "do_context": True,
            "output_map": output_map,
            "last_batch_policy": LastBatchPolicy.FILL,  # LastBatchPolicy.PARTIAL,
            "last_batch_padded": False,
            "auto_reset": False,
            # if we have multiple readers, if we select only 1 here there's an error
            "reader_name": "reader_0" if not self.multiview else None,
        }

        return dict_args

    def __call__(self) -> LitDaliWrapper:
        """
        Returns a LightningWrapper object.
        """
        pipe = self._get_dali_pipe()
        args = self._setup_dali_iterator_args()
        return LitDaliWrapper(pipe, **args[self.train_stage][self.model_type])
