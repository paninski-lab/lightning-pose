"""Bounding-box coordinate transformation utilities."""
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
    'normalized_to_bbox',
    'convert_bbox_coords',
    'convert_original_to_model_coords',
    'original_to_model',
]


def normalized_to_bbox(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    bbox: Float[torch.Tensor, 'batch xyhw'],
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Transform keypoints from normalized coordinates to bbox coordinates."""
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


def convert_bbox_coords(
    batch_dict: (
        HeatmapLabeledBatchDict
        | MultiviewHeatmapLabeledBatchDict
        | MultiviewUnlabeledBatchDict
        | UnlabeledBatchDict
    ),
    predicted_keypoints: Float[torch.Tensor, 'batch num_targets'],
    in_place: bool = True,
) -> Float[torch.Tensor, 'batch num_targets']:
    """Transform keypoints from bbox coordinates to absolute frame coordinates."""
    num_targets = predicted_keypoints.shape[1]
    num_keypoints = num_targets // 2
    # reshape from (batch, n_targets) back to (batch, n_key, 2), in x,y order
    if in_place:
        predicted_keypoints_ = predicted_keypoints.reshape((-1, num_keypoints, 2))
    else:
        predicted_keypoints_ = predicted_keypoints.clone().reshape((-1, num_keypoints, 2))
    # divide by image dims to get 0-1 normalized coordinates
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

            predicted_keypoints_[:, idx_beg:idx_end, :] = normalized_to_bbox(
                predicted_keypoints_[:, idx_beg:idx_end, :],
                bbox_slice,
            )
    else:
        predicted_keypoints_ = normalized_to_bbox(predicted_keypoints_, batch_dict['bbox'])
    # return new keypoints, reshaped to (batch, num_targets)
    return predicted_keypoints_.reshape((-1, num_targets))


def convert_original_to_model_coords(
    batch_dict: MultiviewHeatmapLabeledBatchDict,
    original_keypoints: Float[torch.Tensor, 'batch num_views num_keypoints 2'],
) -> Float[torch.Tensor, 'batch num_views num_keypoints 2']:
    """Transform keypoints from original frame coordinates to model input coordinates."""

    batch_size, num_views, num_keypoints, _ = original_keypoints.shape

    model_height = batch_dict['images'].shape[-2]
    model_width = batch_dict['images'].shape[-1]

    model_keypoints = original_keypoints.clone()

    for v in range(num_views):
        bbox_slice = batch_dict['bbox'][:, 4 * v:4 * (v + 1)]  # (batch, 4)
        model_keypoints[:, v, :, :] = original_to_model(
            original_keypoints[:, v, :, :],
            bbox_slice,
            model_width,
            model_height,
        )

    return model_keypoints


def original_to_model(
    keypoints: Float[torch.Tensor, 'batch num_keypoints 2'],
    bbox: Float[torch.Tensor, 'batch 4'],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, 'batch num_keypoints 2']:
    """Convert keypoints from original image coordinates to model input coordinates.

    Combines two transformations:

    1. original → bbox: subtract offset, divide by bbox dimensions
    2. bbox → model: multiply by model dimensions

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
