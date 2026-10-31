"""Logic shared by the prediction-only video readers (``opencv.py`` and ``pynvvc.py``).

``PreparePredictBase`` and ``LitPredictWrapperBase`` hold everything that does not depend on how
frames are decoded: input validation, windowing parameters, ``num_iters``, ImageNet
normalization, per-view bbox cropping and batch-dict assembly. A backend subclasses both and
supplies only its decoder: ``_read_views`` on the wrapper (return one raw decoded window per
view) and ``__call__`` on the prepare class (build the wrapper).

``normalize_frames`` and ``build_batch_dict`` are plain functions so that
``dali.LitDaliWrapper``, which has its own iterator base class, can use them too.

This module imports neither DALI nor PyNvVideoCodec, so it is safe to import on any machine.
"""

import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.bboxes import as_bbox_df_list, crop_views, full_frame_bbox
from lightning_pose.data.datatypes import (
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import count_frames

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def normalize_frames(
    frames: torch.Tensor,
    decode_resize_dims: list[int] | None,
) -> torch.Tensor:
    """Scale uint8 frames to [0, 1], optionally resize, then ImageNet-normalize.

    Same order of operations as DALI's ``fn.resize`` -> ``fn.crop_mirror_normalize`` in
    ``dali.py:video_pipe``.

    Args:
        frames: uint8 RGB frames, shape (seq_len, C, H, W).
        decode_resize_dims: ``[height, width]`` to resize to, or None to keep the decoded size
            (bbox-crop mode, where full-resolution frames are needed for cropping).

    Returns:
        float tensor of shape (seq_len, C, H', W').
    """
    frames = frames.float() / 255.0
    if decode_resize_dims is not None:
        frames = F.interpolate(
            frames, size=decode_resize_dims, mode='bilinear', align_corners=False,
        )
    mean = torch.tensor(_IMAGENET_MEAN, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean) / std


def build_batch_dict(
    frames_list: list[torch.Tensor],
    bbox_list: list[torch.Tensor],
    transforms: torch.Tensor,
    is_multiview: bool,
) -> UnlabeledBatchDict | MultiviewUnlabeledBatchDict:
    """Assemble per-view frames and bboxes into the batch dict the models consume.

    Args:
        frames_list: one tensor of shape (seq_len, C, H, W) per view; all the same shape.
        bbox_list: one tensor of shape (seq_len, 4) per view, in [x, y, h, w] order.
        transforms: the batch's ``transforms`` field, passed through unchanged.
        is_multiview: whether to build a multiview batch (stacked views) or a single-view one.

    Returns:
        ``MultiviewUnlabeledBatchDict`` with frames of shape (seq_len, num_views, C, H, W) and
        bbox of shape (seq_len, num_views * 4), or ``UnlabeledBatchDict`` for single-view.
    """
    if not is_multiview:
        return UnlabeledBatchDict(
            frames=frames_list[0],
            transforms=transforms,
            bbox=bbox_list[0],
            is_multiview=False,
        )
    return MultiviewUnlabeledBatchDict(
        frames=torch.stack(frames_list, dim=1),
        transforms=transforms,
        bbox=torch.cat(bbox_list, dim=1),
        is_multiview=True,
    )


class PreparePredictBase(ABC):
    """Validation and windowing setup shared by ``PrepareOpenCV`` and ``PreparePynvvc``.

    Construction validates inputs and precomputes windowing parameters (raising early, before
    any decoder is allocated); calling the instance builds the backend's wrapper. Subclasses set
    ``backend_name`` (used in error messages) and implement ``__call__``.

    Sequence-length source: reuses ``dali_config["base"]["predict"]["sequence_length"]`` /
    ``dali_config["context"]["predict"]["sequence_length"]`` (i.e. the existing ``cfg.dali``
    section) -- see ``lightning_pose.data.video``'s package docstring for why this parameter
    name is kept even though it isn't DALI-specific.
    """

    backend_name: str

    def __init__(
        self,
        model_type: Literal['base', 'context'],
        filenames: list[str] | list[list[str]],
        resize_dims: list[int] | None,
        dali_config: dict | DictConfig | ListConfig,
        bbox_df: pd.DataFrame | list[pd.DataFrame] | None = None,
    ) -> None:
        """
        Args:
            model_type: ``"base"`` for standard single-frame models, ``"context"``
                for MHCRNN models that consume a temporal window.
            filenames: for single-view, a flat list containing one video path; for
                multi-view, one single-element list per view.
            resize_dims: ``[height, width]`` to resize frames to before feeding the
                model. Also used as the post-crop resize target when ``bbox_df`` is
                provided. ``None`` (without ``bbox_df``) delivers full-resolution frames.
            dali_config: the ``cfg.dali`` dict; only the ``predict.sequence_length`` values
                (base and context) are used.
            bbox_df: optional DataFrame with columns ``["x", "y", "h", "w"]``, one row per
                frame (for multiview, a list with one such DataFrame per view, in the same
                order as ``filenames``). When provided, frames are decoded at full resolution
                and the wrapper crops each to its bbox before resizing.

        Raises:
            FileNotFoundError: if any path in ``filenames`` does not exist or is not
                a file.
            NotImplementedError: if any view supplies more than one video (multi-
                session batching is a DALI-train-only concept, not needed here).
            ValueError: for multiview inputs, if views have differing frame counts
                (which would desynchronize the per-view sequential reads), if the number
                of ``bbox_df`` DataFrames does not match the number of views, if
                ``bbox_df`` is set without ``resize_dims``, or for an unknown ``model_type``.
        """
        if bbox_df is not None and resize_dims is None:
            raise ValueError('resize_dims is required when bbox_df is provided.')
        if isinstance(filenames, list) and isinstance(filenames[0], list):
            self.multiview = True
        else:
            self.multiview = False

        filenames_2d: list[list[str]]
        if isinstance(filenames[0], str):
            filenames_2d = [filenames]  # type: ignore[list-item]
        else:
            filenames_2d = filenames  # type: ignore[assignment]

        for view_list in filenames_2d:
            if len(view_list) != 1:
                raise NotImplementedError(
                    f'{self.backend_name} predict backend supports exactly one video per view '
                    f'(no multi-session batching); got {len(view_list)} videos for one view.'
                )
            vid = view_list[0]
            if not os.path.exists(vid) or not os.path.isfile(vid):
                raise FileNotFoundError(f'{vid} is not a video file!')

        view0_frame_count = count_frames(filenames_2d[0][0])
        if self.multiview:
            for view_idx, view_list in enumerate(filenames_2d[1:], start=1):
                frame_count = count_frames(view_list[0])
                if frame_count != view0_frame_count:
                    raise ValueError(
                        f'Mismatched frame counts across views; multiview {self.backend_name} '
                        f'reading would desynchronize. view 0={view0_frame_count}, '
                        f'view {view_idx}={frame_count}'
                    )

        self.model_type = model_type
        self.filenames = filenames_2d
        self.resize_dims = resize_dims
        # fail early on a bbox_df/view count mismatch, before any decoder is allocated
        as_bbox_df_list(bbox_df, len(filenames_2d))
        self.bbox_df = bbox_df
        self.frame_count = view0_frame_count

        if model_type == 'base':
            predict_cfg = dali_config['base']['predict']  # type: ignore[index]
            self.sequence_length = predict_cfg['sequence_length']
            self.step = self.sequence_length
            self.do_context = False
        elif model_type == 'context':
            predict_cfg = dali_config['context']['predict']  # type: ignore[index]
            self.sequence_length = predict_cfg['sequence_length']
            self.step = self.sequence_length - 4
            self.do_context = True
        else:
            raise ValueError(f'unknown model_type: {model_type}')

        # bbox-crop mode needs full-resolution decoded frames to crop from; the
        # post-crop resize to resize_dims happens in crop_and_resize_frames instead.
        self._decode_resize_dims: list[int] | None = (
            None if self.bbox_df is not None else resize_dims
        )

    @property
    def num_iters(self) -> int:
        """Number of dataloader iterations required to process all frames.

        Mirrors the branch of ``PrepareDALI.num_iters`` that applies to prediction (single
        sequence at a time, batch_size=1): for context models this is the
        "step == sequence_length - 4" case.
        """
        if self.model_type == 'base':
            return int(np.ceil(self.frame_count / self.sequence_length))
        else:  # context
            if self.step <= 0:
                raise ValueError(
                    'step cannot be 0, please modify '
                    'cfg.dali.context.predict.sequence_length to be > 4'
                )
            data_except_first_batch = self.frame_count - self.sequence_length
            return int(np.ceil(data_except_first_batch / self.step)) + 1

    @abstractmethod
    def __call__(self) -> 'LitPredictWrapperBase':
        """Build and return a ready-to-iterate wrapper for this backend."""


class LitPredictWrapperBase(ABC):
    """Iterator logic shared by ``LitOpenCVWrapper`` and ``LitPynvvcWrapper``.

    Subclasses implement ``_read_views`` (decode the next window of every view); this class
    handles iteration bookkeeping, normalization, per-view bbox cropping and batch assembly.

    Predict-only: no ``random_shuffle``, no ``imgaug``. See
    ``lightning_pose.data.video.dali.LitDaliWrapper`` for the train-time DALI path.
    """

    def __init__(
        self,
        resize_dims: list[int] | None,
        decode_resize_dims: list[int] | None,
        sequence_length: int,
        step: int,
        do_context: bool,
        num_iters: int,
        multiview: bool,
        bbox_df: pd.DataFrame | list[pd.DataFrame] | None = None,
    ) -> None:
        """
        Args:
            resize_dims: ``[height, width]`` the model expects as input. Used as the
                post-crop resize target when ``bbox_df`` is set.
            decode_resize_dims: ``[height, width]`` to resize decoded frames to
                immediately after decode. ``None`` in bbox-crop mode (full-resolution
                frames are needed for cropping); equal to ``resize_dims`` otherwise.
            sequence_length: number of frames per batch/window.
            step: frames to advance the read cursor between windows. Equals
                ``sequence_length`` for non-overlapping ("base") windows, or
                ``sequence_length - 4`` for context models' overlapping windows.
            do_context: whether this is a 5-frame-context (MHCRNN-style) model.
            num_iters: total number of batches this video will produce; precomputed
                by the ``Prepare*`` class so Lightning can report progress.
            multiview: whether this is a multiview prediction (one decoder per view,
                read in lockstep since predict never shuffles).
            bbox_df: optional per-frame bbox DataFrame (columns x, y, h, w); for multiview,
                a list with one DataFrame per view, in the same order as the views.
        """
        self.resize_dims = resize_dims
        self.decode_resize_dims = decode_resize_dims
        self.sequence_length = sequence_length
        self.step = step
        self.do_context = do_context
        self.num_iters = num_iters
        self.multiview = multiview
        self.bbox_df = bbox_df

        self._iters_done = 0
        # cursor into bbox_df; advances by the context-adjusted step per batch, same
        # convention as LitDaliWrapper._frame_idx
        self._frame_idx = 0

    def __len__(self) -> int:
        """Return the number of iterations (batches) in this dataloader."""
        return self.num_iters

    def __iter__(self) -> 'LitPredictWrapperBase':
        return self

    @abstractmethod
    def _read_views(self) -> list[torch.Tensor]:
        """Decode the next window of every view.

        Returns:
            one uint8 RGB tensor of shape (seq_len, C, H, W) per view, at the video's native
            resolution (views may differ in size).
        """

    def _resize_normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize one view's decoded frames; see :func:`normalize_frames`."""
        return normalize_frames(frames, self.decode_resize_dims)

    def __next__(self) -> UnlabeledBatchDict | MultiviewUnlabeledBatchDict:
        """Fetch the next batch, applying per-frame bbox crop+resize when configured."""
        if self._iters_done >= self.num_iters:
            raise StopIteration
        self._iters_done += 1

        per_view_frames = self._read_views()
        processed = [self._resize_normalize(f) for f in per_view_frames]
        seq_len = self.sequence_length

        # bbox reflects the true, pre-resize decoded frame size (not decode_resize_dims), so
        # norm_to_frame correctly rescales model-space predictions back into the original
        # video's pixel coordinates -- mirrors DALI's separate "frame_size" output, which is
        # captured before fn.resize.
        bbox_dfs = as_bbox_df_list(self.bbox_df, len(processed))
        if bbox_dfs is None:
            bbox_list = [
                full_frame_bbox(pv.shape[-2], pv.shape[-1], seq_len, pv.device)
                for pv in per_view_frames
            ]
        else:
            assert self.resize_dims is not None  # required whenever bbox_df is set
            # views are cropped independently, so they may have different native resolutions
            processed, bbox_list = crop_views(
                processed, bbox_dfs, self._frame_idx, self.resize_dims,
            )
            self._frame_idx += seq_len - 4 if self.do_context else seq_len

        return build_batch_dict(
            processed,
            bbox_list,
            transforms=(
                torch.tensor([-1.0]).repeat(len(processed), 1, 1)
                if self.multiview else torch.tensor([-1.0])
            ),
            is_multiview=self.multiview,
        )
