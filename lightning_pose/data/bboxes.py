"""Bounding-box coordinate transformation utilities.

Three coordinate spaces are used throughout this module:

- **frame**: absolute pixel position in the original full-resolution camera frame.
- **norm**: [0, 1] coordinates normalised relative to the bounding box
  (top-left corner of bbox = (0, 0), bottom-right = (1, 1)).  This is the
  intermediate space between frame and model.
- **model**: pixel position in the model's resized input image (after cropping
  the frame to the bounding box and scaling to the model's input resolution).

Tensor-level helpers (accept explicit keypoint tensors, plus bbox and/or model
dimension arguments):

- :func:`frame_to_norm`  — frame → norm
- :func:`norm_to_frame`  — norm → frame  (mutates input in-place)
- :func:`model_to_norm`  — model → norm
- :func:`norm_to_model`  — norm → model
- :func:`frame_to_model` — frame → model  (= frame_to_norm ∘ norm_to_model)
- :func:`model_to_frame` — model → frame  (= model_to_norm ∘ norm_to_frame)

Batch-level wrappers (accept a ``batch_dict`` that carries ``bbox`` and image
dimensions; handle both single-view and multi-view):

- :func:`frame_to_model_batch` — frame → model  (called before loss computation)
- :func:`model_to_frame_batch` — model → frame  (called after inference)
"""

import torch
from jaxtyping import Float

from lightning_pose.data.datatypes import (
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def frame_to_norm(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    bbox: Float[torch.Tensor, 'batch xyhw'],
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Transform keypoints from frame coordinates to norm coordinates.

    Args:
        keypoints: keypoints in frame (pixel) coordinates, shape (batch, num_keypoints, 2)
        bbox: bounding boxes in [x, y, h, w] format, shape (batch, 4)

    Returns:
        keypoints in norm coordinates, shape (batch, num_keypoints, 2)
    """
    kp = keypoints.clone()
    if kp.shape[0] == bbox.shape[0]:
        # normal batch
        kp[:, :, 0] -= bbox[:, 0].unsqueeze(1)   # subtract bbox x offset
        kp[:, :, 0] /= bbox[:, 3].unsqueeze(1)   # divide by bbox width
        kp[:, :, 1] -= bbox[:, 1].unsqueeze(1)   # subtract bbox y offset
        kp[:, :, 1] /= bbox[:, 2].unsqueeze(1)   # divide by bbox height
    else:
        # context batch; we don't have predictions for first/last two frames
        kp[:, :, 0] -= bbox[2:-2, 0].unsqueeze(1)
        kp[:, :, 0] /= bbox[2:-2, 3].unsqueeze(1)
        kp[:, :, 1] -= bbox[2:-2, 1].unsqueeze(1)
        kp[:, :, 1] /= bbox[2:-2, 2].unsqueeze(1)
    return kp


def norm_to_frame(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    bbox: Float[torch.Tensor, 'batch xyhw'],
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Transform keypoints from norm coordinates to frame coordinates.

    Unlike all other helpers in this module, this function modifies ``keypoints``
    in-place and returns it.  Callers that need to preserve the original tensor
    should pass a clone.

    Args:
        keypoints: keypoints in norm coordinates, shape (batch, num_keypoints, 2);
            modified in-place
        bbox: bounding boxes in [x, y, h, w] format, shape (batch, 4)

    Returns:
        keypoints in frame (pixel) coordinates, shape (batch, num_keypoints, 2);
            same tensor as the input
    """
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


def model_to_norm(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Transform keypoints from model coordinates to norm coordinates.

    Args:
        keypoints: keypoints in model (pixel) coordinates, shape (batch, num_keypoints, 2)
        model_width: width of the model's input image in pixels
        model_height: height of the model's input image in pixels

    Returns:
        keypoints in norm coordinates, shape (batch, num_keypoints, 2)
    """
    kp = keypoints.clone()
    kp[:, :, 0] /= model_width
    kp[:, :, 1] /= model_height
    return kp


def norm_to_model(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Transform keypoints from norm coordinates to model coordinates.

    Args:
        keypoints: keypoints in norm coordinates, shape (batch, num_keypoints, 2)
        model_width: width of the model's input image in pixels
        model_height: height of the model's input image in pixels

    Returns:
        keypoints in model (pixel) coordinates, shape (batch, num_keypoints, 2)
    """
    kp = keypoints.clone()
    kp[:, :, 0] *= model_width
    kp[:, :, 1] *= model_height
    return kp


def frame_to_model(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    bbox: Float[torch.Tensor, 'batch xyhw'],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Convert keypoints from frame coordinates to model coordinates.

    Composes two transformations: frame → norm → model.

    Args:
        keypoints: keypoints in frame (pixel) coordinates, shape (batch, num_keypoints, 2)
        bbox: bounding boxes in [x, y, h, w] format, shape (batch, 4)
        model_width: width of the model's input image in pixels
        model_height: height of the model's input image in pixels

    Returns:
        keypoints in model (pixel) coordinates, shape (batch, num_keypoints, 2)
    """
    return norm_to_model(frame_to_norm(keypoints, bbox), model_width, model_height)


def model_to_frame(
    keypoints: Float[torch.Tensor, 'batch num_keypoints xy'],
    bbox: Float[torch.Tensor, 'batch xyhw'],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, 'batch num_keypoints xy']:
    """Convert keypoints from model coordinates to frame coordinates.

    Composes two transformations: model → norm → frame.

    Args:
        keypoints: keypoints in model (pixel) coordinates, shape (batch, num_keypoints, 2)
        bbox: bounding boxes in [x, y, h, w] format, shape (batch, 4)
        model_width: width of the model's input image in pixels
        model_height: height of the model's input image in pixels

    Returns:
        keypoints in frame (pixel) coordinates, shape (batch, num_keypoints, 2)
    """
    return norm_to_frame(model_to_norm(keypoints, model_width, model_height), bbox)


def frame_to_model_batch(
    batch_dict: MultiviewHeatmapLabeledBatchDict,
    frame_keypoints: Float[torch.Tensor, 'batch num_views num_keypoints xy'],
) -> Float[torch.Tensor, 'batch num_views num_keypoints xy']:
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


def model_to_frame_batch(
    batch_dict: (
        HeatmapLabeledBatchDict
        | MultiviewHeatmapLabeledBatchDict
        | MultiviewUnlabeledBatchDict
        | UnlabeledBatchDict
    ),
    model_keypoints: Float[torch.Tensor, 'batch num_targets'],
    in_place: bool = True,
) -> Float[torch.Tensor, 'batch num_targets']:
    """Transform keypoints from model coordinates to frame coordinates.

    Reads image dimensions and bbox from ``batch_dict``; handles single-view
    and multi-view batches.
    """
    num_targets = model_keypoints.shape[1]
    num_keypoints = num_targets // 2
    # reshape from (batch, n_targets) to (batch, n_keypoints, 2), in x,y order
    if in_place:
        model_keypoints_ = model_keypoints.reshape((-1, num_keypoints, 2))
    else:
        model_keypoints_ = model_keypoints.clone().reshape((-1, num_keypoints, 2))
    # extract model image dimensions
    if 'images' in batch_dict.keys():
        img = batch_dict['images']  # type: ignore[typeddict-item]
        model_width = img.shape[-1]   # -1 dim is width "x"
        model_height = img.shape[-2]  # -2 dim is height "y"
    else:  # we have unlabeled dict, 'frames' instead of 'images'
        frames = batch_dict['frames']  # type: ignore[typeddict-item]
        model_width = frames.shape[-1]   # -1 dim is width "x"
        model_height = frames.shape[-2]  # -2 dim is height "y"
    # divide by model dims to get norm coordinates (in-place)
    model_keypoints_[:, :, 0] /= model_width
    model_keypoints_[:, :, 1] /= model_height
    # multiply and add by bbox dims to get frame coordinates, per view
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
            model_keypoints_[:, idx_beg:idx_end, :] = norm_to_frame(
                model_keypoints_[:, idx_beg:idx_end, :],
                bbox_slice,
            )
    else:
        model_keypoints_ = norm_to_frame(model_keypoints_, batch_dict['bbox'])
    # return new keypoints, reshaped to (batch, num_targets)
    return model_keypoints_.reshape((-1, num_targets))
