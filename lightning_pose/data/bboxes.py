"""Bounding-box coordinate transformation utilities.

Three coordinate spaces are used throughout this module:

- **frame**: absolute pixel position in the original full-resolution camera frame.
- **norm**: [0, 1] coordinates normalised relative to the bounding box
  (top-left corner of bbox = (0, 0), bottom-right = (1, 1)).  This is the
  implicit intermediate space that connects the other two.
- **model**: pixel position in the model's resized input image (after cropping
  the frame to the bounding box and scaling to the model's input resolution).

Tensor-level helpers (accept explicit keypoint + bbox tensors):

- :func:`norm_to_frame`  — norm → frame
- :func:`frame_to_model` — frame → model  (via norm internally)

Batch-level wrappers (accept a ``batch_dict`` that carries ``bbox`` and image
dimensions; handle both single-view and multi-view):

- :func:`model_to_frame_batch` — model → frame  (called after inference)
- :func:`frame_to_model_batch` — frame → model  (called before loss computation)
"""
from __future__ import annotations

import torch
from jaxtyping import Float

from lightning_pose.data.datatypes import (
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)

__all__ = [
    'norm_to_frame',
    'frame_to_model',
    'model_to_frame_batch',
    'frame_to_model_batch',
]


def norm_to_frame(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    bbox: Float[torch.Tensor, 'batch xyhw'],
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Transform keypoints from norm coordinates to frame coordinates."""
    if keypoints.shape[0] == bbox.shape[0]:
        # normal batch
        keypoints[:, :, 0] *= bbox[:, 3].unsqueeze(1)  # scale x by box width
        keypoints[:, :, 0] += bbox[:, 0].unsqueeze(1)  # add bbox x offset
        keypoints[:, :, 1] *= bbox[:, 2].unsqueeze(1)  # scale y by box height
        keypoints[:, :, 1] += bbox[:, 1].unsqueeze(1)  # add bbox y offset
    else:
        # context batch; we don't have predictions for first/last two frames
        keypoints[:, :, 0] *= bbox[2:-2, 3].unsqueeze(1)  # scale x by box width
        keypoints[:, :, 0] += bbox[2:-2, 0].unsqueeze(1)  # add bbox x offset
        keypoints[:, :, 1] *= bbox[2:-2, 2].unsqueeze(1)  # scale y by box height
        keypoints[:, :, 1] += bbox[2:-2, 1].unsqueeze(1)  # add bbox y offset
    return keypoints


def model_to_frame_batch(
    batch_dict: (
        HeatmapLabeledBatchDict
        | MultiviewHeatmapLabeledBatchDict
        | MultiviewUnlabeledBatchDict
        | UnlabeledBatchDict
    ),
    predicted_keypoints: Float[torch.Tensor, 'batch num_targets'],
    in_place: bool = True,
) -> Float[torch.Tensor, 'batch num_targets']:
    """Transform keypoints from model coordinates to frame coordinates.

    Reads image dimensions and bbox from ``batch_dict``; handles single-view
    and multi-view batches.
    """
    num_targets = predicted_keypoints.shape[1]
    num_keypoints = num_targets // 2
    # reshape from (batch, n_targets) back to (batch, n_key, 2), in x,y order
    if in_place:
        predicted_keypoints_ = predicted_keypoints.reshape((-1, num_keypoints, 2))
    else:
        predicted_keypoints_ = predicted_keypoints.clone().reshape((-1, num_keypoints, 2))
    # divide by image dims to get norm coordinates
    if 'images' in batch_dict.keys():
        img = batch_dict['images']  # type: ignore[typeddict-item]
        predicted_keypoints_[:, :, 0] /= img.shape[-1]  # -1 dim is width "x"
        predicted_keypoints_[:, :, 1] /= img.shape[-2]  # -2 dim is height "y"
    else:  # we have unlabeled dict, 'frames' instead of 'images'
        frames = batch_dict['frames']  # type: ignore[typeddict-item]
        predicted_keypoints_[:, :, 0] /= frames.shape[-1]  # -1 dim is width "x"
        predicted_keypoints_[:, :, 1] /= frames.shape[-2]  # -2 dim is height "y"
    # multiply and add by bbox dims (x,y,h,w)
    if (
        ('num_views' in batch_dict.keys() and int(batch_dict['num_views'].max()) > 1)  # type: ignore[typeddict-item]
        or batch_dict.get('is_multiview', False)
    ):
        # the first check is for labeled batches while is_multiview is for unlabeled batches
        # For MultiviewUnlabeledBatchDict, we need to infer num_views from bbox shape
        if 'num_views' in batch_dict.keys():
            unique = batch_dict['num_views'].unique()  # type: ignore[typeddict-item]
            if len(unique) != 1:
                raise ValueError(
                    f'each batch element must contain the same number of views; '
                    f'found elements with {unique} views'
                )
            num_views = int(unique)
        else:
            # infer from bbox shape: bbox has shape [seq_len, num_views * 4]
            num_views = batch_dict['bbox'].shape[1] // 4

        num_keypoints_per_view = num_keypoints // num_views

        for v in range(num_views):
            idx_beg = num_keypoints_per_view * v
            idx_end = idx_beg + num_keypoints_per_view
            bbox_slice = batch_dict['bbox'][:, 4 * v:4 * (v + 1)]

            predicted_keypoints_[:, idx_beg:idx_end, :] = norm_to_frame(
                predicted_keypoints_[:, idx_beg:idx_end, :],
                bbox_slice,
            )
    else:
        predicted_keypoints_ = norm_to_frame(predicted_keypoints_, batch_dict['bbox'])
    # return new keypoints, reshaped to (batch, num_targets)
    return predicted_keypoints_.reshape((-1, num_targets))


def frame_to_model_batch(
    batch_dict: MultiviewHeatmapLabeledBatchDict,
    frame_keypoints: Float[torch.Tensor, 'batch num_views num_keypoints 2'],
) -> Float[torch.Tensor, 'batch num_views num_keypoints 2']:
    """Transform keypoints from frame coordinates to model coordinates.

    Reads image dimensions and bbox from ``batch_dict``; handles multi-view
    batches by processing each view independently.
    """
    batch_size, num_views, num_keypoints, _ = frame_keypoints.shape

    model_height = batch_dict['images'].shape[-2]
    model_width = batch_dict['images'].shape[-1]

    model_keypoints = frame_keypoints.clone()

    for v in range(num_views):
        bbox_slice = batch_dict['bbox'][:, 4 * v:4 * (v + 1)]  # (batch, 4)
        model_keypoints[:, v, :, :] = frame_to_model(
            frame_keypoints[:, v, :, :],
            bbox_slice,
            model_width,
            model_height,
        )

    return model_keypoints


def frame_to_model(
    keypoints: Float[torch.Tensor, 'batch num_keypoints 2'],
    bbox: Float[torch.Tensor, 'batch 4'],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, 'batch num_keypoints 2']:
    """Convert keypoints from frame coordinates to model coordinates.

    Combines two transformations:

    1. frame → norm: subtract bbox offset, divide by bbox dimensions
    2. norm → model: multiply by model dimensions

    bbox format: [x, y, h, w] where x,y is top-left corner.
    """
    model_keypoints = keypoints.clone()

    if keypoints.shape[0] == bbox.shape[0]:
        # normal batch
        model_keypoints[:, :, 0] -= bbox[:, 0].unsqueeze(1)
        model_keypoints[:, :, 0] /= bbox[:, 3].unsqueeze(1)
        model_keypoints[:, :, 1] -= bbox[:, 1].unsqueeze(1)
        model_keypoints[:, :, 1] /= bbox[:, 2].unsqueeze(1)
        model_keypoints[:, :, 0] *= model_width
        model_keypoints[:, :, 1] *= model_height
    else:
        # context batch; we don't have predictions for first/last two frames
        model_keypoints[:, :, 0] -= bbox[2:-2, 0].unsqueeze(1)
        model_keypoints[:, :, 0] /= bbox[2:-2, 3].unsqueeze(1)
        model_keypoints[:, :, 1] -= bbox[2:-2, 1].unsqueeze(1)
        model_keypoints[:, :, 1] /= bbox[2:-2, 2].unsqueeze(1)
        model_keypoints[:, :, 0] *= model_width
        model_keypoints[:, :, 1] *= model_height

    return model_keypoints
