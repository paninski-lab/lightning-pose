"""Dataset/data module utilities."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
from jaxtyping import Float, Int

from lightning_pose.data.datatypes import (
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)

logger = logging.getLogger(__name__)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "split_sizes_from_probabilities",
    "clean_any_nans",
    "count_frames",
    "compute_num_train_frames",
    "generate_heatmaps",
    "evaluate_heatmaps_at_location",
    "undo_affine_transform",
    "undo_affine_transform_batch",
    "normalized_to_bbox",
    "convert_bbox_coords",
    "convert_original_to_model_coords",
    "original_to_model",
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


def generate_heatmaps(
    keypoints: Float[torch.Tensor, "batch num_keypoints 2"],
    height: int,
    width: int,
    output_shape: tuple[int, int],
    sigma: float = 1.25,
    uniform_heatmaps: bool = False,
    keep_gradients: bool = False,
    visibility: Int[torch.Tensor, "batch num_keypoints"] | None = None,
) -> Float[torch.Tensor, "batch num_keypoints height width"]:
    """Generate 2D Gaussian heatmaps from mean and sigma.

    Args:
        keypoints: coordinates that serve as mean of gaussian bump
        height: height of reshaped image (pixels, e.g., 128, 256, 512...)
        width: width of reshaped image (pixels, e.g., 128, 256, 512...)
        output_shape: dimensions of downsampled heatmap, (height, width)
        sigma: control spread of gaussian
        uniform_heatmaps: output uniform heatmaps if missing ground truth label, rather than skip;
            ignored when ``visibility`` is provided
        keep_gradients: True to not detach gradients from keypoints before creating heatmaps
        visibility: optional per-keypoint visibility flags with values 0 (not labeled → zero
            heatmap), 1 (occluded → uniform heatmap), or 2 (visible → Gaussian heatmap). When
            None, falls back to NaN detection combined with the ``uniform_heatmaps`` flag.

    Returns:
        batch of 2D heatmaps

    """
    if keep_gradients:
        keypoints = keypoints.clone()
    else:
        keypoints = keypoints.detach().clone()
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    # nan_idxs = torch.isnan(keypoints)[:, :, 0]
    # Mark as invalid: NaN keypoints OR out-of-bounds keypoints
    nan_idxs = (
        torch.isnan(keypoints)[:, :, 0]  # Original NaN check
        | (keypoints[:, :, 0] < -1)  # x < -1
        | (keypoints[:, :, 0] > out_width + 1)  # x > width + 1
        | (keypoints[:, :, 1] < -1)  # y < -1
        | (keypoints[:, :, 1] > out_height + 1)  # y > height + 1
    )

    # Clamp keypoints to prevent extreme Gaussian computations
    # Use a reasonable buffer around the image bounds
    # keypoints[:, :, 0] = torch.clamp(keypoints[:, :, 0], -margin, out_width + margin)
    # keypoints[:, :, 1] = torch.clamp(keypoints[:, :, 1], -margin, out_height + margin)
    clamped_x = torch.clamp(keypoints[:, :, 0], -1, out_width + 1)
    clamped_y = torch.clamp(keypoints[:, :, 1], -1, out_height + 1)
    keypoints = torch.stack([clamped_x, clamped_y], dim=2)

    xv = torch.arange(out_width, device=keypoints.device)
    yv = torch.arange(out_height, device=keypoints.device)
    # note flipped order because of pytorch's ij and numpy's xy indexing for meshgrid
    xx, yy = torch.meshgrid(yv, xv, indexing="ij")
    # adds batch and num_keypoints dimensions to grids
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)
    # adds dimension corresponding to the first dimension of the 2d grid
    keypoints = keypoints.unsqueeze(2)
    # evaluates 2d gaussian with mean equal to the keypoint and var equal to sigma^2
    heatmaps = (yy - keypoints[:, :, :, :1]) ** 2  # also flipped order here
    heatmaps += (xx - keypoints[:, :, :, 1:]) ** 2  # also flipped order here
    heatmaps *= -1
    heatmaps /= 2 * sigma**2
    heatmaps = torch.exp(heatmaps)
    # normalize all heatmaps to one
    heatmaps = heatmaps / torch.sum(heatmaps, dim=(2, 3), keepdim=True)
    zero_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)
    uniform_heatmap = torch.ones(
        (out_height, out_width), device=keypoints.device
    ) / (out_height * out_width)

    if visibility is None:
        # backward-compat: NaN/OOB keypoints get a uniform or zero filler based on the flag
        # (all-zeros heatmaps are ignored in the supervised heatmap loss)
        heatmaps[nan_idxs] = uniform_heatmap if uniform_heatmaps else zero_heatmap
    else:
        heatmaps[visibility == 0] = zero_heatmap    # not labeled: ignore in loss
        heatmaps[visibility == 1] = uniform_heatmap  # occluded: encourage low confidence
        # visible keypoints with NaN/OOB coords fall back to zero (defensive)
        heatmaps[(visibility == 2) & nan_idxs] = zero_heatmap

    return heatmaps


def evaluate_heatmaps_at_location(
    heatmaps: Float[torch.Tensor, "batch num_keypoints heatmap_height heatmap_width"],
    locs: Float[torch.Tensor, "batch num_keypoints 2"],
    sigma: float = 1.25,  # sigma used for generating heatmaps
    num_stds: int = 2,  # num standard deviations of pixels to compute confidence
) -> Float[torch.Tensor, "batch num_keypoints"]:
    """Evaluate 4D heatmaps using a 3D location tensor (last dim is x, y coords). Since
    the model outputs heatmaps with a standard deviation of sigma, confidence will be
    spread across neighboring pixels. To account for this, confidence is computed by
    taking all pixels within two standard deviations of the predicted pixel."""
    pix_to_consider = int(np.floor(sigma * num_stds))  # get all pixels within num_stds.
    num_pad = pix_to_consider
    heatmaps_padded = torch.zeros(
        (
            heatmaps.shape[0],
            heatmaps.shape[1],
            heatmaps.shape[2] + num_pad * 2,
            heatmaps.shape[3] + num_pad * 2,
        ),
        device=heatmaps.device,
    )
    heatmaps_padded[:, :, num_pad:-num_pad, num_pad:-num_pad] = heatmaps
    i = torch.arange(heatmaps_padded.shape[0], device=heatmaps_padded.device).reshape(
        -1, 1, 1, 1
    )
    j = torch.arange(heatmaps_padded.shape[1], device=heatmaps_padded.device).reshape(
        1, -1, 1, 1
    )
    k = locs[:, :, None, 1, None].type(torch.int64) + num_pad
    m = locs[:, :, 0, None, None].type(torch.int64) + num_pad
    offsets = list(np.arange(-pix_to_consider, pix_to_consider + 1))
    vals_all = []
    for offset in offsets:
        k_offset = k + int(offset)
        for offset_2 in offsets:
            m_offset = m + int(offset_2)
            # get rid of singleton dims
            vals = heatmaps_padded[i, j, k_offset, m_offset].squeeze(-1).squeeze(-1)
            vals_all.append(vals)
    vals = torch.stack(vals_all, 0).sum(0)
    return vals


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


def normalized_to_bbox(
    keypoints: Float[torch.Tensor, "batch num_keypoints xy"],
    bbox: Float[torch.Tensor, "batch xyhw"]
) -> Float[torch.Tensor, "batch num_keypoints xy"]:
    """Transform keypoints from normalized coordinates to bbox coordinates"""
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
    predicted_keypoints: Float[torch.Tensor, "batch num_targets"],
    in_place: bool = True,
) -> Float[torch.Tensor, "batch num_targets"]:
    """Transform keypoints from bbox coordinates to absolute frame coordinates."""
    num_targets = predicted_keypoints.shape[1]
    num_keypoints = num_targets // 2
    # reshape from (batch, n_targets) back to (batch, n_key, 2), in x,y order
    if in_place:
        predicted_keypoints_ = predicted_keypoints.reshape((-1, num_keypoints, 2))
    else:
        predicted_keypoints_ = predicted_keypoints.clone().reshape((-1, num_keypoints, 2))
    # divide by image dims to get 0-1 normalized coordinates
    if "images" in batch_dict.keys():
        img = batch_dict["images"]  # type: ignore[typeddict-item]
        predicted_keypoints_[:, :, 0] /= img.shape[-1]  # -1 dim is width "x"
        predicted_keypoints_[:, :, 1] /= img.shape[-2]  # -2 dim is height "y"
    else:  # we have unlabeled dict, 'frames' instead of 'images'
        frames = batch_dict["frames"]  # type: ignore[typeddict-item]
        predicted_keypoints_[:, :, 0] /= frames.shape[-1]  # -1 dim is width "x"
        predicted_keypoints_[:, :, 1] /= frames.shape[-2]  # -2 dim is height "y"
    # multiply and add by bbox dims (x,y,h,w)
    if (
        ("num_views" in batch_dict.keys() and int(batch_dict["num_views"].max()) > 1)  # type: ignore[typeddict-item]
        or batch_dict.get("is_multiview", False)
    ):
        # the first check is for labeled batches while is_multiview is for unlabeled batches
        # For MultiviewUnlabeledBatchDict, we need to infer num_views from bbox shape
        if "num_views" in batch_dict.keys():
            unique = batch_dict["num_views"].unique()  # type: ignore[typeddict-item]
            if len(unique) != 1:
                raise ValueError(
                    f"each batch element must contain the same number of views; "
                    f"found elements with {unique} views"
                )
            num_views = int(unique)
        else:
            # Infer from bbox shape: bbox has shape [seq_len, num_views * 4]
            num_views = batch_dict["bbox"].shape[1] // 4

        num_keypoints_per_view = num_keypoints // num_views

        for v in range(num_views):
            idx_beg = num_keypoints_per_view * v
            idx_end = idx_beg + num_keypoints_per_view
            bbox_slice = batch_dict["bbox"][:, 4 * v:4 * (v + 1)]

            predicted_keypoints_[:, idx_beg:idx_end, :] = normalized_to_bbox(
                predicted_keypoints_[:, idx_beg:idx_end, :],
                bbox_slice,
            )
    else:
        predicted_keypoints_ = normalized_to_bbox(predicted_keypoints_, batch_dict["bbox"])
    # return new keypoints, reshaped to (batch, num_targets)
    return predicted_keypoints_.reshape((-1, num_targets))


def convert_original_to_model_coords(
    batch_dict: MultiviewHeatmapLabeledBatchDict,
    original_keypoints: Float[torch.Tensor, "batch num_views num_keypoints 2"],
) -> Float[torch.Tensor, "batch num_views num_keypoints 2"]:
    """Transform keypoints from original frame coordinates to model input coordinates."""

    batch_size, num_views, num_keypoints, _ = original_keypoints.shape

    # Get model input dimensions
    model_height = batch_dict["images"].shape[-2]  # height
    model_width = batch_dict["images"].shape[-1]  # width

    # Clone to avoid modifying original
    model_keypoints = original_keypoints.clone()

    # Process each view
    for v in range(num_views):
        bbox_slice = batch_dict["bbox"][:, 4 * v:4 * (v + 1)]  # (batch, 4)

        model_keypoints[:, v, :, :] = original_to_model(
            original_keypoints[:, v, :, :],  # (batch, num_keypoints, 2)
            bbox_slice,  # (batch, 4)
            model_width,
            model_height,
        )

    return model_keypoints


def original_to_model(
    keypoints: Float[torch.Tensor, "batch num_keypoints 2"],
    bbox: Float[torch.Tensor, "batch 4"],
    model_width: float,
    model_height: float,
) -> Float[torch.Tensor, "batch num_keypoints 2"]:
    """Convert keypoints from original image coordinates to model input coordinates.

    This combines the transformations:
    1. original → bbox: subtract offset, divide by bbox dimensions
    2. bbox → model: multiply by model dimensions

    bbox format: [x, y, h, w] where x,y is top-left corner
    """
    model_keypoints = keypoints.clone()

    if keypoints.shape[0] == bbox.shape[0]:
        # normal batch
        # Step 1: original → bbox (normalize to [0,1] relative to bbox)
        model_keypoints[:, :, 0] -= bbox[:, 0].unsqueeze(1)  # subtract bbox x offset
        model_keypoints[:, :, 0] /= bbox[:, 3].unsqueeze(1)  # divide by box width
        model_keypoints[:, :, 1] -= bbox[:, 1].unsqueeze(1)  # subtract bbox y offset
        model_keypoints[:, :, 1] /= bbox[:, 2].unsqueeze(1)  # divide by box height

        # Step 2: bbox → model (scale to model dimensions)
        model_keypoints[:, :, 0] *= model_width  # scale to model width
        model_keypoints[:, :, 1] *= model_height  # scale to model height

    else:
        # context batch; we don't have predictions for first/last two frames
        # Step 1: original → bbox
        model_keypoints[:, :, 0] -= bbox[2:-2, 0].unsqueeze(1)  # subtract bbox x offset
        model_keypoints[:, :, 0] /= bbox[2:-2, 3].unsqueeze(1)  # divide by box width
        model_keypoints[:, :, 1] -= bbox[2:-2, 1].unsqueeze(1)  # subtract bbox y offset
        model_keypoints[:, :, 1] /= bbox[2:-2, 2].unsqueeze(1)  # divide by box height

        # Step 2: bbox → model
        model_keypoints[:, :, 0] *= model_width  # scale to model width
        model_keypoints[:, :, 1] *= model_height  # scale to model height

    return model_keypoints
