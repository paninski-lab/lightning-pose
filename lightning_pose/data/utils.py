"""Dataset/data module utilities."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
from jaxtyping import Float

logger = logging.getLogger(__name__)

# to ignore imports for sphix-autoapidoc
__all__ = [
    'split_sizes_from_probabilities',
    'clean_any_nans',
    'count_frames',
    'compute_num_train_frames',
    'undo_affine_transform',
    'undo_affine_transform_batch',
]


def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: float | None = None,
    test_probability: float | None = None,
) -> list[int]:
    """Returns the number of examples for train, val and test given split probs.

    Args:
        total_number: total number of examples in dataset
        train_probability: fraction of examples used for training
        val_probability: fraction of examples used for validation
        test_probability: fraction of examples used for test. Defaults to None. Can be computed
            as the remaining examples.

    Returns:
        [num training examples, num validation examples, num test examples]

    """

    if test_probability is None and val_probability is None:
        remaining_probability = 1.0 - train_probability
        # round each to 5 decimal places (issue with floating point precision)
        val_probability = round(remaining_probability / 2, 5)
        test_probability = round(remaining_probability / 2, 5)
    elif test_probability is None:
        assert val_probability is not None
        test_probability = 1.0 - train_probability - val_probability

    # probabilities should add to one
    assert val_probability is not None
    assert test_probability + train_probability + val_probability == 1.0

    # compute numbers from probabilities
    train_number = int(np.floor(train_probability * total_number))
    val_number = int(np.floor(val_probability * total_number))

    # if we lose extra examples by flooring, send these to train_number or test_number, depending
    leftover = total_number - train_number - val_number
    if leftover < 5:
        # very few samples, let's bulk up train
        train_number += leftover
        test_number = 0
    else:
        test_number = leftover

    # make sure that we have at least one validation sample
    if val_number == 0:
        train_number -= 1
        val_number += 1
        if train_number < 1:
            raise ValueError("Must have at least two labeled frames, one train and one validation")

    # assert that we're using all datapoints
    assert train_number + test_number + val_number == total_number

    return [train_number, val_number, test_number]


def clean_any_nans(data: torch.Tensor, dim: int) -> torch.Tensor:
    """Remove samples from a data array that contain nans."""
    # currently supports only 2D arrays
    nan_bool = (
        torch.sum(torch.isnan(data), dim=dim) > 0
    )  # e.g., when dim == 0, those columns (keypoints) that have >0 nans
    if dim == 0:
        return data[:, ~nan_bool]
    elif dim == 1:
        return data[~nan_bool]
    raise ValueError(f"dim must be 0 or 1, got {dim}")


def count_frames(video_file: str) -> int:
    """
    Simple function to count the number of frames in a video.
    """
    assert os.path.isfile(video_file)

    import cv2

    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return num_frames


def compute_num_train_frames(
    len_train_dataset: int,
    train_frames: int | float | None = None,
) -> int:
    """Quickly compute number of training frames for a given dataset.

    Args:
        len_train_dataset: total number of frames in training dataset
        train_frames:
            <=1 - fraction of total train frames used for training
            >1 - number of total train frames used for training

    Returns:
        total number of train frames

    """
    if train_frames is None:
        n_train_frames = len_train_dataset
    else:
        if train_frames >= len_train_dataset:
            # take max number of train frames
            logger.warning('requested training frames exceeds training set size; using all')
            n_train_frames = len_train_dataset
        elif train_frames == 1:
            # assume this is a fraction; use full dataset
            n_train_frames = len_train_dataset
        elif train_frames > 1:
            # take this number of train frames
            n_train_frames = int(train_frames)
        elif train_frames > 0:
            # take this fraction of train frames
            n_train_frames = int(train_frames * len_train_dataset)
        else:
            raise ValueError("train_frames must be >0")

    return n_train_frames



def undo_affine_transform(
    keypoints: Float[torch.Tensor, "seq_len num_keypoints 2"],
    transform: Float[torch.Tensor, "seq_len 2 3"] | Float[torch.Tensor, "2 3"],
) -> Float[torch.Tensor, "seq_len num_keypoints 2"]:
    """Undo an affine transform given a tensor of keypoints and the tranform matrix."""

    # add 1s to get keypoints in projective geometry coords
    ones = torch.ones(
        (keypoints.shape[0], keypoints.shape[1], 1),
        dtype=keypoints.dtype,
        device=keypoints.device,
        requires_grad=True,
    )
    kps_aff = torch.cat([keypoints, ones], dim=2)

    mat = torch.clone(transform).detach()
    if len(transform.shape) == 2:
        # single transform for all frames; add batch dim
        mat = mat.unsqueeze(0)

    # create inverse matrices
    mats_inv_torch = []
    for idx in range(mat.shape[0]):
        mat_inv_ = torch.linalg.inv(mat[idx, :, :2])
        mat_inv = torch.concat(
            [mat_inv_, torch.matmul(-mat_inv_, mat[idx, :, -1, None])], dim=1
        )
        mats_inv_torch.append(
            torch.transpose(mat_inv, 1, 0).detach().clone().to(
                dtype=keypoints.dtype, device=keypoints.device,
            ).requires_grad_(True)
        )

    # make a single block of inverse matrices
    if len(mats_inv_torch) == 1:
        # replicate this inverse matrix for each element of the batch
        mat_inv_torch = torch.tile(
            mats_inv_torch[0].unsqueeze(0), dims=(keypoints.shape[0], 1, 1)
        )
    else:
        # different transformation for each element of the batch
        mat_inv_torch = torch.stack(mats_inv_torch, dim=0)

    # apply inverse matrix to each element individually using batch matrix multiply
    kps_noaug = torch.bmm(kps_aff, mat_inv_torch)

    return kps_noaug


def undo_affine_transform_batch(
    keypoints_augmented: Float[torch.Tensor, "seq_len num_keypointsx2"],
    transforms: (
        Float[torch.Tensor, "seq_len h w"]
        | Float[torch.Tensor, "h w"]
        | Float[torch.Tensor, "seq_len 1"]
        | Float[torch.Tensor, 1]
        | Float[torch.Tensor, "num_views h w"]
        | Float[torch.Tensor, "num_views 1 1"]
    ),
    is_multiview: bool = False,
) -> Float[torch.Tensor, "seq_len num_keypointsx2"]:
    """Potentially undo an affine transform given a tensor of keypoints and the tranform matrix."""

    # undo augmentation if needed
    if transforms.shape[-1] == 3:
        # initial shape is (seq_len, n_keypoints * 2)
        # reshape to (seq_len, n_keypoints, 2)
        pred_kps = torch.reshape(
            keypoints_augmented,
            (keypoints_augmented.shape[0], -1, 2)
        )
        # undo
        if not is_multiview:
            # single affine transform for the whole batch
            pred_kps = undo_affine_transform(pred_kps, transforms)
        else:
            # each view has its own affine transform that we need to undo
            num_views = transforms.shape[0]
            kps_per_view = int(pred_kps.shape[1] / num_views)
            for v in range(num_views):
                idx_beg = v * kps_per_view
                idx_end = (v + 1) * kps_per_view
                # undo
                pred_kps[:, idx_beg:idx_end] = undo_affine_transform(
                    pred_kps[:, idx_beg:idx_end],
                    transforms[v]
                )
        # reshape to (seq_len, n_keypoints * 2)
        keypoints_unaugmented = torch.reshape(pred_kps, (pred_kps.shape[0], -1))
    else:
        keypoints_unaugmented = keypoints_augmented

    return keypoints_unaugmented
