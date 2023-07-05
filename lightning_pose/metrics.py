from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig
from typeguard import typechecked

from lightning_pose.utils.pca import KeypointPCA


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
        keypoints_pred = torch.tensor(keypoints_pred, device=t_loss.device, dtype=torch.float32)

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
    cfg: Union[DictConfig, dict],
) -> np.ndarray:
    """PCA reprojection error.

    Args:
        keypoints_pred: shape (samples, n_keypoints, 2)
        pca: pca object that contains info about pca subspace
        cfg: standard config file that carries around dataset info

    Returns:
        shape (samples, n_keypoints)

    """

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(keypoints_pred, device=pca.device, dtype=torch.float32)
    original_dims = keypoints_pred.shape

    pca_cols = pca.columns_for_singleview_pca

    # resize->reformat->reprojection->resize
    keypoints_reproj = _resize_reproject_resize(cfg, pca, keypoints_pred)

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
    cfg: Union[DictConfig, dict],
) -> np.ndarray:
    """PCA reprojection error.

    Args:
        keypoints_pred: shape (samples, n_keypoints, 2)
        pca: pca object that contains info about pca subspace
        cfg: standard config file that carries around dataset info

    Returns:
        shape (samples, n_keypoints)

    """

    if not isinstance(keypoints_pred, torch.Tensor):
        keypoints_pred = torch.tensor(keypoints_pred, device=pca.device, dtype=torch.float32)
    original_dims = keypoints_pred.shape

    mirrored_column_matches = list(pca.mirrored_column_matches)

    # resize->reformat->reprojection->resize
    keypoints_reproj = _resize_reproject_resize(cfg, pca, keypoints_pred)

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


def _resize_reproject_resize(cfg, pca, keypoints_pred):
    """Helper function for both pca singleview and pca multiview metrics."""

    # resize predictions to training dims (where pca is computed)
    keypoints_pred_resize = _resize_keypoints(
        cfg, keypoints_pred=keypoints_pred, orig_to_resize=True)

    # compute reprojection
    # adding a reshaping since the loss class expects a single last dim with num_keypoints * 2
    data_arr = pca._format_data(
        data_arr=keypoints_pred_resize.reshape(keypoints_pred_resize.shape[0], -1))
    reproj = pca.reproject(data_arr=data_arr)
    keypoints_reproj_resize = reproj.reshape(reproj.shape[0], reproj.shape[1] // 2, 2)

    # resize reprojections back to original dims
    keypoints_reproj = _resize_keypoints(cfg, keypoints_reproj_resize, orig_to_resize=False)

    return keypoints_reproj


def _resize_keypoints(cfg, keypoints_pred, orig_to_resize):
    """reshape to training dims for pca losses, which are optimized for these dims"""
    x_resize = cfg.data.image_resize_dims.width
    x_og = cfg.data.image_orig_dims.width
    y_resize = cfg.data.image_resize_dims.height
    y_og = cfg.data.image_orig_dims.height
    if orig_to_resize:
        sx = x_resize / x_og
        sy = y_resize / y_og
    else:
        sx = x_og / x_resize
        sy = y_og / y_resize
    if isinstance(keypoints_pred, np.ndarray):
        keypoints_resize = np.copy(keypoints_pred)
    else:
        keypoints_resize = torch.clone(keypoints_pred)
    keypoints_resize[:, :, 0] = keypoints_pred[:, :, 0] * sx
    keypoints_resize[:, :, 1] = keypoints_pred[:, :, 1] * sy
    return keypoints_resize
