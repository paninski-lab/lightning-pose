"""Data pipelines based on efficient video reading by nvidia dali package."""

import os
from typing import Any, Literal

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pandas as pd
import torch
import torch.nn.functional as F
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from omegaconf import DictConfig, ListConfig

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.datatypes import (
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import count_frames

# to ignore imports for sphix-autoapidoc
__all__ = [
    "video_pipe",
    "LitDaliWrapper",
    "PrepareDALI",
]


# cannot typecheck due to way pipeline_def decorator consumes additional args
@pipeline_def
def video_pipe(
    filenames: list[str] | str | list[list[str]],
    resize_dims: list[int] | None = None,
    random_shuffle: bool = False,
    sequence_length: int = 16,
    pad_sequences: bool = True,
    initial_fill: int = 16,
    normalization_mean: list[float] = _IMAGENET_MEAN,
    normalization_std: list[float] = _IMAGENET_STD,
    name: str = "reader",
    step: int = 1,
    pad_last_batch: bool = False,
    imgaug: str = "default",
    skip_vfr_check: bool = True,
    reader_seed: int = 123456,
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
        reader_seed: seed shared by all per-view video readers. For multiview pipelines this must
            be identical across views so that every reader shuffles to the same sequence, keeping
            the views frame-synchronized (same session, same timepoint) within a batch.

    Returns:
        pipeline object to be fed to DALIGenericIterator
        data augmentation matrix (returned so that geometric transforms can be undone)
        size of video frames, used for bbox

    """
    # turn all inputs into a list of list of strings to be most general
    # first list: over views (might only be one)
    # second list: over videos/sessions
    filenames_2d: list[list[str]]
    if isinstance(filenames, str):
        filenames_2d = [[filenames]]
    elif isinstance(filenames[0], str):
        filenames_2d = [filenames]  # type: ignore[list-item]
    else:
        filenames_2d = filenames  # type: ignore[assignment]

    # loop over views (can be only one)
    frames_list = []
    transform_list = []
    orig_size_list = []
    for f, filename_list in enumerate(filenames_2d):
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
            # explicit, identical seed across views keeps multiview readers synchronized;
            # without it DALI auto-assigns a different seed per reader and the views desync
            seed=reader_seed,
        )
        orig_size = video.shape(device='gpu')  # type: ignore[union-attr]
        if resize_dims:
            video = fn.resize(video, size=resize_dims)  # type: ignore[arg-type]
        if imgaug == "dlc" or imgaug == "dlc-top-down":
            assert resize_dims is not None
            size = (resize_dims[0] / 2, resize_dims[1] / 2)
            center = size  # / 2
            # rotate + scale
            angle = fn.random.uniform(range=(-10, 10))
            transform = fn.transforms.rotation(angle=angle, center=center)
            scale = fn.random.uniform(range=(0.8, 1.2), shape=2)
            transform = fn.transforms.scale(transform, scale=scale, center=center)
            video = fn.warp_affine(
                video,  # type: ignore[arg-type]
                matrix=transform, fill_value=0, inverse_map=False,
            )
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
        video = video / 255.0  # type: ignore[operator]
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
        *args: Any,
        eval_mode: Literal["train", "predict"],
        num_iters: int = 1,
        do_context: bool = False,
        bbox_df: pd.DataFrame | None = None,
        resize_dims: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Wrapper around DALIGenericIterator to get batches for pl.

        Args:
            eval_mode: ``"train"`` or ``"predict"``.
            num_iters: number of enumerations of dataloader (should be computed outside for now;
                should be fixed by lightning/dali teams)
            do_context: whether model/loader use 5-frame context or not
            bbox_df: optional DataFrame with columns ``["x", "y", "h", "w"]``, one row per
                frame. When provided, each batch's frames are cropped per-frame and resized
                to ``resize_dims`` before being returned, and the ``bbox`` field of the
                batch dict is populated with the actual bbox coordinates.
            resize_dims: target ``[height, width]`` for post-crop resize; required when
                ``bbox_df`` is not None.

        """
        self.num_iters = num_iters
        self.do_context = do_context
        self.eval_mode = eval_mode
        self.batch_sampler = 1  # hack to get around DALI-ptl issue
        self.bbox_df = bbox_df
        self.resize_dims = resize_dims
        self._frame_idx = 0
        # call parent
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        """Return the number of iterations (batches) in this dataloader."""
        return self.num_iters

    @staticmethod
    def _dali_output_to_tensors(
        batch: list
    ) -> UnlabeledBatchDict | MultiviewUnlabeledBatchDict:
        """Convert a raw DALI batch output to a typed batch dictionary.

        Args:
            batch: raw output list from DALI's ``DALIGenericIterator.__next__``.

        Returns:
            An ``UnlabeledBatchDict`` for single-view pipelines or a
            ``MultiviewUnlabeledBatchDict`` for multi-view pipelines.
        """
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

    def _apply_bbox_crop(self, batch_dict: UnlabeledBatchDict) -> UnlabeledBatchDict:
        """Crop frames to per-frame bboxes and resize to the model's input dimensions.

        Args:
            batch_dict: single-view unlabeled batch with full-resolution frames from DALI.

        Returns:
            new UnlabeledBatchDict with cropped+resized frames and real bbox values.
        """
        frames = batch_dict['frames']  # (seq_len, 3, H, W)
        seq_len = frames.shape[0]
        step = seq_len - 4 if self.do_context else seq_len

        # slice bbox rows for this batch; pad last partial batch with the final row
        rows = self.bbox_df.iloc[self._frame_idx:self._frame_idx + seq_len]
        if len(rows) < seq_len:
            last_row = self.bbox_df.iloc[[-1]]
            rows = pd.concat(
                [rows] + [last_row] * (seq_len - len(rows)),
                ignore_index=True,
            )

        cropped_frames = []
        bboxes = []
        for i in range(seq_len):
            row = rows.iloc[i]
            x, y, h, w = int(row['x']), int(row['y']), int(row['h']), int(row['w'])
            frame_cropped = frames[i, :, y:y + h, x:x + w]
            frame_resized = F.interpolate(
                frame_cropped.unsqueeze(0),
                size=self.resize_dims,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            cropped_frames.append(frame_resized)
            bboxes.append(
                torch.tensor([x, y, h, w], dtype=torch.float32, device=frames.device)
            )

        self._frame_idx += step

        return UnlabeledBatchDict(
            frames=torch.stack(cropped_frames),
            transforms=batch_dict['transforms'],
            bbox=torch.stack(bboxes),
            is_multiview=False,
        )

    def __next__(self) -> UnlabeledBatchDict | MultiviewUnlabeledBatchDict:
        """Fetch the next batch, applying per-frame bbox crop+resize when configured."""
        batch = super().__next__()
        batch_dict = self._dali_output_to_tensors(batch=batch)
        if self.bbox_df is not None and not batch_dict['is_multiview']:
            return self._apply_bbox_crop(batch_dict)  # type: ignore[arg-type]
        return batch_dict


class PrepareDALI:
    """All the DALI stuff in one place.

    Big picture: this will initialize the pipes and dataloaders for both training and prediction.

    """

    def __init__(
        self,
        train_stage: Literal["predict", "train"],
        model_type: Literal["base", "context"],
        filenames: list[str] | list[list[str]],
        resize_dims: list[int],
        dali_config: dict | DictConfig | ListConfig | None = None,
        imgaug: str | None = "default",
        num_threads: int = 1,
        bbox_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize DALI pipelines and dataloaders for training or prediction.

        Args:
            train_stage: whether to set up pipelines for ``"train"`` or ``"predict"``.
            model_type: ``"base"`` for standard single-frame models, ``"context"`` for
                MHCRNN models that consume a temporal window.
            filenames: for single-view models, a flat list of video file paths; for
                multi-view models, a list of per-view lists of video file paths.
            resize_dims: ``[height, width]`` to resize frames to before feeding the model.
                Also used as the post-crop resize target when ``bbox_df`` is provided.
            dali_config: DALI-specific config dict; falls back to package defaults when None.
            imgaug: name of the augmentation pipeline to apply during training (e.g.
                ``"dlc"``); pass ``"default"`` for resize-only or ``None`` to disable.
            num_threads: number of CPU threads used by DALI pipelines.
            bbox_df: optional DataFrame with columns ``["x", "y", "h", "w"]``, one row per
                frame. When provided, the predict pipeline loads full-resolution frames
                (DALI resize is disabled) and ``LitDaliWrapper`` crops each frame to its
                bbox before resizing to ``resize_dims``.

        Raises:
            FileNotFoundError: if any path in ``filenames`` does not exist or is not a file.
            ValueError: for multiview inputs, if views have differing numbers of sessions or if a
                session has differing frame counts across views (which would desynchronize the
                per-view readers).
        """
        # determine if we have a multiview pipeline
        if isinstance(filenames, list) and isinstance(filenames[0], list):
            self.multiview = True
        else:
            self.multiview = False

        # make sure `filenames` is a list of existing video files
        filenames_2d: list[list[str]]
        if isinstance(filenames[0], str):
            filenames_2d = [filenames]  # type: ignore[list-item]
        else:
            filenames_2d = filenames  # type: ignore[assignment]
        for view_list in filenames_2d:
            for vid in view_list:
                if not os.path.exists(vid) or not os.path.isfile(vid):
                    raise FileNotFoundError(f"{vid} is not a video file!")

        # frame counts of view 0, per session; reused below for `self.frame_count`
        view0_frame_counts = list(map(count_frames, filenames_2d[0]))

        # For multiview, the per-view DALI readers share a seed so they shuffle to the same
        # sequence index. That only keeps the views frame-synchronized if every session has the
        # same number of frames across all views (and the same number of sessions per view).
        # Otherwise the readers silently drift apart, so fail loudly here instead.
        if self.multiview:
            num_sessions = len(filenames_2d[0])
            for view_idx, view_list in enumerate(filenames_2d):
                if len(view_list) != num_sessions:
                    raise ValueError(
                        f"View {view_idx} has {len(view_list)} video(s) but view 0 has "
                        f"{num_sessions}; all views must have the same number of sessions for "
                        "synchronized multiview loading."
                    )
            for session_idx in range(num_sessions):
                counts = [view0_frame_counts[session_idx]] + [
                    count_frames(filenames_2d[view_idx][session_idx])
                    for view_idx in range(1, len(filenames_2d))
                ]
                if len(set(counts)) != 1:
                    details = ", ".join(
                        f"{filenames_2d[view_idx][session_idx]}={counts[view_idx]}"
                        for view_idx in range(len(filenames_2d))
                    )
                    raise ValueError(
                        "Mismatched frame counts across views for the same session; multiview "
                        "video readers would desynchronize. Frame counts: "
                        f"{details}"
                    )

        self.train_stage = train_stage
        self.model_type = model_type
        self.filenames = filenames_2d
        self.resize_dims = resize_dims
        self.dali_config = dali_config
        self.num_threads = num_threads
        self.bbox_df = bbox_df
        self.frame_count = sum(view0_frame_counts)
        self._pipe_dict: dict = self._setup_pipe_dict(self.filenames, imgaug)

    @property
    def num_iters(self) -> int:
        """Number of dataloader iterations required to process all frames.

        Returns:
            Integer count of how many times the dataloader must be enumerated to exhaust all video
            frames for the current ``train_stage`` and ``model_type`` configuration.
        """
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
        else:
            raise ValueError(f'unknown model_type: {self.model_type}')

    def _setup_pipe_dict(
        self,
        filenames: list[str] | list[list[str]],
        imgaug: str | None,
    ) -> dict[str, dict]:
        """All of the pipeline args in one place."""
        assert self.dali_config is not None
        # When running with multiple GPUs, the LOCAL_RANK variable correctly
        # contains the DDP Local Rank, which is also the cuda device index.
        device_id = int(os.environ.get("LOCAL_RANK", "0"))

        dict_args = {
            "predict": {"context": {}, "base": {}},
            "train": {"context": {}, "base": {}},
        }
        gen_cfg = self.dali_config.get("general", {"seed": 123456})  # type: ignore[arg-type]
        # Multi-GPU strategy is to have each GPU randomize differently. The same value seeds every
        # per-view reader (`reader_seed`) so that multiview readers shuffle identically and stay
        # frame-synchronized within a pipeline.
        pipeline_seed = gen_cfg["seed"] + device_id

        # base (vanilla single-frame model), train pipe args
        base_train_cfg = self.dali_config["base"]["train"]  # type: ignore[arg-type]
        dict_args["train"]["base"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": base_train_cfg["sequence_length"],
            "step": base_train_cfg["sequence_length"],
            "batch_size": 1,
            "seed": pipeline_seed,
            "reader_seed": pipeline_seed,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": True,
            "imgaug": imgaug,
        }

        # base (vanilla single-frame model), predict pipe args
        base_pred_cfg = self.dali_config["base"]["predict"]  # type: ignore[arg-type]
        dict_args["predict"]["base"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": base_pred_cfg["sequence_length"],
            "step": base_pred_cfg["sequence_length"],
            "batch_size": 1,
            "seed": pipeline_seed,
            "reader_seed": pipeline_seed,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": False,
            "name": "reader",
            "pad_sequences": True,
            "imgaug": "default",  # no imgaug when predicting
        }

        # context (five-frame) model, predict pipe args
        context_pred_cfg = self.dali_config["context"]["predict"]  # type: ignore[index]
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
            "seed": pipeline_seed,
            "reader_seed": pipeline_seed,
            "pad_sequences": True,
            # "pad_last_batch": True,
            "imgaug": "default",  # no imgaug when predicting
        }

        # context (five-frame) model, train pipe args
        # grab a single sequence of frames, will resize into 5-frame chunks at the
        # representation level inside BaseFeatureExtractor
        # note: reusing the batch size argument
        context_train_cfg = self.dali_config["context"]["train"]  # type: ignore[index]
        dict_args["train"]["context"] = {
            "filenames": filenames,
            "resize_dims": self.resize_dims,
            "sequence_length": context_train_cfg["batch_size"],
            "step": context_train_cfg["batch_size"],
            "batch_size": 1,
            "seed": pipeline_seed,
            "reader_seed": pipeline_seed,
            "num_threads": self.num_threads,
            "device_id": device_id,
            "random_shuffle": True,
            "imgaug": imgaug,
        }
        # our floor above should prevent us from getting to the very final batch.

        # when per-frame bbox cropping is enabled, DALI must deliver full-resolution frames so
        # that LitDaliWrapper can crop them; the post-crop resize happens in PyTorch instead
        if self.bbox_df is not None:
            dict_args['predict']['base']['resize_dims'] = None
            dict_args['predict']['context']['resize_dims'] = None

        return dict_args

    def _get_dali_pipe(self) -> Any:
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
        """Return a LitDaliWrapper configured for the current train stage and model type."""
        pipe = self._get_dali_pipe()
        args = self._setup_dali_iterator_args()
        iterator_args = args[self.train_stage][self.model_type]
        if self.bbox_df is not None:
            iterator_args['bbox_df'] = self.bbox_df
            iterator_args['resize_dims'] = self.resize_dims
        return LitDaliWrapper(pipe, **iterator_args)
