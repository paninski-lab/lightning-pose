"""Evaluation metrics for assessing pose estimation model quality."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import MultiviewHeatmapDataset
from lightning_pose.data.datatypes import ComputeMetricsSingleResult
from lightning_pose.utils.io import fix_empty_first_row, get_keypoint_names
from lightning_pose.utils.pca import KeypointPCA

# to ignore imports for sphix-autoapidoc
__all__ = [
    "pixel_error",
    "temporal_norm",
    "pca_singleview_reprojection_error",
    "pca_multiview_reprojection_error",
    "compute_metrics_single",
]


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


def temporal_norm(keypoints_pred: np.ndarray | torch.Tensor) -> np.ndarray:
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


def pca_singleview_reprojection_error(
    keypoints_pred: np.ndarray | torch.Tensor,
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


def pca_multiview_reprojection_error(
    keypoints_pred: np.ndarray | torch.Tensor,
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

    assert pca.mirrored_column_matches is not None
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


def compute_metrics_single(
    cfg: DictConfig | ListConfig,
    labels_file: str | Path | None,
    preds_file: str | Path,
    data_module: BaseDataModule | UnlabeledDataModule | None = None,
) -> ComputeMetricsSingleResult:
    """Compute various metrics on a predictions csv file from a single view.

    Args:
        cfg: hydra config
        labels_file: path to the labels csv file; required for pixel error computation
        preds_file: path to the predictions csv file
        data_module: required for PCA metric computation

    Returns:
        ComputeMetricsSingleResult containing dataframes for each computed metric

    """
    pred_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)
    keypoint_names = get_keypoint_names(cfg, csv_file=str(preds_file), header_rows=[0, 1, 2])
    xyl_mask = pred_df.columns.get_level_values('coords').isin(['x', 'y', 'likelihood'])
    tmp = pred_df.loc[:, xyl_mask].to_numpy().reshape(pred_df.shape[0], -1, 3)

    index = pred_df.index
    if pred_df.keys()[-1][0] == 'set':
        is_video = False
        set = pred_df.iloc[:, -1].to_numpy()
    else:
        is_video = True
        set = None

    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)

    if is_video:
        metrics_to_compute = ['temporal']
    else:
        assert labels_file is not None
        metrics_to_compute = ['pixel_error']

    if (
        data_module is not None
        and cfg.data.get('columns_for_singleview_pca', None) is not None
        and len(cfg.data.columns_for_singleview_pca) != 0
        and not isinstance(data_module.dataset, MultiviewHeatmapDataset)
    ):
        metrics_to_compute += ['pca_singleview']
    if (
        data_module is not None
        and cfg.data.get('mirrored_column_matches', None) is not None
        and len(cfg.data.mirrored_column_matches) != 0
        and not isinstance(data_module.dataset, MultiviewHeatmapDataset)
    ):
        metrics_to_compute += ['pca_multiview']

    result = ComputeMetricsSingleResult()
    preds_file_path = Path(preds_file)

    if 'pixel_error' in metrics_to_compute:
        assert labels_file is not None, '"pixel_error" metric requires labels_file'
        labels_df = pd.read_csv(labels_file, header=[0, 1, 2], index_col=0)
        labels_df = fix_empty_first_row(labels_df)
        assert labels_df.index.equals(index)
        xy_mask = labels_df.columns.get_level_values('coords').isin(['x', 'y'])
        labels_df = labels_df.loc[:, xy_mask]

        keypoints_true = labels_df.to_numpy().reshape(labels_df.shape[0], -1, 2)
        error_per_keypoint = pixel_error(keypoints_true, keypoints_pred)
        error_df = pd.DataFrame(
            error_per_keypoint, index=pd.Index(index), columns=pd.Index(keypoint_names),
        )
        if set is not None:
            error_df['set'] = set
        save_file = preds_file_path.with_name(preds_file_path.stem + '_pixel_error.csv')
        error_df.to_csv(save_file)
        result.pixel_error_df = error_df

    if 'temporal' in metrics_to_compute:
        temporal_norm_per_keypoint = temporal_norm(keypoints_pred)
        temporal_norm_df = pd.DataFrame(
            temporal_norm_per_keypoint, index=pd.Index(index), columns=pd.Index(keypoint_names),
        )
        if set is not None:
            temporal_norm_df['set'] = set
        save_file = preds_file_path.with_name(preds_file_path.stem + '_temporal_norm.csv')
        temporal_norm_df.to_csv(save_file)
        result.temporal_norm_df = temporal_norm_df

    if 'pca_singleview' in metrics_to_compute:
        try:
            assert data_module is not None
            pca = KeypointPCA(
                loss_type='pca_singleview',
                data_module=data_module,
                components_to_keep=cfg.losses.pca_singleview.components_to_keep,
                empirical_epsilon_percentile=cfg.losses.pca_singleview.get(
                    'empirical_epsilon_percentile', 1.0,
                ),
                columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
                centering_method=cfg.losses.pca_singleview.get('centering_method', None),
            )
            pca()
            pcasv_error_per_keypoint = pca_singleview_reprojection_error(keypoints_pred, pca)
            pcasv_df = pd.DataFrame(
                pcasv_error_per_keypoint,
                index=pd.Index(index),
                columns=pd.Index(keypoint_names),
            )
            if set is not None:
                pcasv_df['set'] = set
            save_file = preds_file_path.with_name(
                preds_file_path.stem + '_pca_singleview_error.csv',
            )
            pcasv_df.to_csv(save_file)
            result.pca_sv_df = pcasv_df
        except ValueError as e:
            if 'cannot fit PCA' not in str(e):
                raise e

    if 'pca_multiview' in metrics_to_compute:
        assert data_module is not None
        pca = KeypointPCA(
            loss_type='pca_multiview',
            data_module=data_module,
            components_to_keep=cfg.losses.pca_singleview.components_to_keep,
            empirical_epsilon_percentile=cfg.losses.pca_singleview.get(
                'empirical_epsilon_percentile', 1.0,
            ),
            mirrored_column_matches=cfg.data.mirrored_column_matches,
        )
        pca()
        pcamv_error_per_keypoint = pca_multiview_reprojection_error(keypoints_pred, pca)
        pcamv_df = pd.DataFrame(
            pcamv_error_per_keypoint, index=pd.Index(index), columns=pd.Index(keypoint_names),
        )
        if set is not None:
            pcamv_df['set'] = set
        save_file = preds_file_path.with_name(preds_file_path.stem + '_pca_multiview_error.csv')
        pcamv_df.to_csv(save_file)
        result.pca_mv_df = pcamv_df

    return result
