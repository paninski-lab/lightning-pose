from typing import Union

import numpy as np
import torch
from typeguard import typechecked

from lightning_pose.utils.pca import KeypointPCA

# to ignore imports for sphix-autoapidoc
__all__ = [
    "pixel_error",
    "temporal_norm",
    "pca_singleview_reprojection_error",
    "pca_multiview_reprojection_error",
]


@typechecked
def pixel_error(keypoints_true: np.ndarray, keypoints_pred: np.ndarray) -> np.ndarray:
    """Root mean square error between true and predicted keypoints.

    Args:
        keypoints_true: shape (samples, n_keypoints, 2)
        keypoints_pred: shape (samples, n_keypoints, 2)

    Returns:
        shape (samples, n_keypoints)

    """
    error = np.linalg.norm(keypoints_true - keypoints_pred, axis=2)
    return error


@typechecked
def temporal_norm(keypoints_pred: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Norm of difference between keypoints on successive time bins.

    Args:
        keypoints_pred: shape (samples, n_keypoints * 2) or (samples, n_keypoints, 2)

    Returns:
        shape (samples, n_keypoints)

    """

    from lightning_pose.losses.losses import TemporalLoss

    t_loss = TemporalLoss()

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(keypoints_pred, dtype=torch.float32)

    # (samples, n_keypoints, 2) -> (samples, n_keypoints * 2)
    if len(keypoints_pred.shape) != 2:
        keypoints_pred = keypoints_pred.reshape(keypoints_pred.shape[0], -1)

    # compute loss with already-implemented class
    t_norm = t_loss.compute_loss(keypoints_pred)
    # prepend nan vector; no temporal norm for the very first frame
    t_norm = np.vstack([np.nan * np.zeros((1, t_norm.shape[1])), t_norm.numpy()])

    return t_norm


@typechecked
def pca_singleview_reprojection_error(
    keypoints_pred: Union[np.ndarray, torch.Tensor],
    pca: KeypointPCA,
) -> np.ndarray:
    """PCA reprojection error.

    Args:
        keypoints_pred: shape (samples, n_keypoints, 2)
        pca: pca object that contains info about pca subspace

    Returns:
        shape (samples, n_keypoints)

    """

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(keypoints_pred, device=pca.device, dtype=torch.float32)
    original_dims = keypoints_pred.shape

    pca_cols = pca.columns_for_singleview_pca

    # reshape: loss class expects a single last dim with num_keypoints * 2
    data_arr = pca._format_data(data_arr=keypoints_pred.reshape(keypoints_pred.shape[0], -1))

    # compute reprojection
    reproj = pca.reproject(data_arr=data_arr)

    # reshape again
    keypoints_reproj = reproj.reshape(reproj.shape[0], reproj.shape[1] // 2, 2)

    # compute pixel error
    error_pca = pixel_error(
        keypoints_pred[:, pca_cols, :].cpu().numpy(), keypoints_reproj.cpu().numpy())

    # next, put this back into a full keypoints pred arr; keypoints not included in pose for pca
    # are set to nan
    error_all = np.nan * np.zeros((original_dims[0], original_dims[1]))
    error_all[:, pca_cols] = error_pca

    return error_all


@typechecked
def pca_multiview_reprojection_error(
    keypoints_pred: Union[np.ndarray, torch.Tensor],
    pca: KeypointPCA,
) -> np.ndarray:
    """PCA reprojection error.

    Args:
        keypoints_pred: shape (samples, n_keypoints, 2)
        pca: pca object that contains info about pca subspace

    Returns:
        shape (samples, n_keypoints)

    """

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(keypoints_pred, device=pca.device, dtype=torch.float32)
    original_dims = keypoints_pred.shape

    mirrored_column_matches = list(pca.mirrored_column_matches)

    # reshape: loss class expects a single last dim with num_keypoints * 2
    data_arr = pca._format_data(data_arr=keypoints_pred.reshape(keypoints_pred.shape[0], -1))

    # compute reprojection
    reproj = pca.reproject(data_arr=data_arr)

    # reshape again
    keypoints_reproj = reproj.reshape(reproj.shape[0], reproj.shape[1] // 2, 2)

    # put original keypoints in same format
    keypoints_pred_reformat = pca._format_data(
        data_arr=keypoints_pred.reshape(keypoints_pred.shape[0], -1))
    keypoints_pred_reformat = keypoints_pred_reformat.reshape(
        keypoints_pred_reformat.shape[0], keypoints_pred_reformat.shape[1] // 2, 2)

    # compute pixel error
    error_pca = pixel_error(keypoints_pred_reformat.cpu().numpy(), keypoints_reproj.cpu().numpy())

    # next, put this back into a full keypoints pred arr
    error_pca = error_pca.reshape(
        -1,
        len(mirrored_column_matches[0]),
        len(mirrored_column_matches),
    )  # batch X num_used_keypoints X num_views
    error_all = np.nan * np.zeros((original_dims[0], original_dims[1]))
    for c, cols in enumerate(mirrored_column_matches):
        error_all[:, cols] = error_pca[:, :, c]  # just the columns belonging to view c

    return error_all
