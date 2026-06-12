"""Test metrics module."""

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from lightning_pose.metrics import (
    compute_metrics_single,
    pca_multiview_reprojection_error,
    pca_singleview_reprojection_error,
    pixel_error,
    temporal_norm,
)
from lightning_pose.utils.pca import KeypointPCA


def test_pixel_error():

    n_samples = 10
    n_keypoints = 5
    keypoints_true = np.random.randn(n_samples, n_keypoints, 2)

    # correctly returns zero when there is no error
    rmse_per_keypoint = pixel_error(keypoints_true, keypoints_true)
    assert np.all(rmse_per_keypoint == 0)
    assert rmse_per_keypoint.shape == (n_samples, n_keypoints)

    # all entries positive
    keypoints_pred = np.random.randn(n_samples, n_keypoints, 2)
    rmse_per_keypoint = pixel_error(keypoints_true, keypoints_pred)
    assert np.all(rmse_per_keypoint >= 0)


def test_temporal_norm():

    # make sure zero is returned for constant predictions (along dim 0)
    n_samples = 10
    n_keypoints = 5
    keypoints_pred = np.zeros((n_samples, n_keypoints, 2))
    keypoints_pred[:, 1, :] = 1
    keypoints_pred[:, 2, :] = 2
    keypoints_pred[:, 3, :] = 3
    keypoints_pred[:, 4, :] = 4
    norm_per_keypoint = temporal_norm(keypoints_pred)
    assert norm_per_keypoint.shape == (n_samples, n_keypoints)
    assert np.all(norm_per_keypoint[1:] == 0.0)

    # make sure non-negative scalar is returned
    keypoints_pred = np.random.randn(n_samples, n_keypoints, 2)
    norm_per_keypoint = temporal_norm(keypoints_pred)
    assert np.all(norm_per_keypoint[1:] >= 0.0)

    # check against actual norm
    keypoints_pred = np.zeros((n_samples, n_keypoints, 2))
    keypoints_pred[1, 0, 0] = np.sqrt(2)
    keypoints_pred[1, 0, 1] = np.sqrt(2)
    norm_per_keypoint = temporal_norm(keypoints_pred)
    assert np.isnan(norm_per_keypoint[0, 0])
    assert norm_per_keypoint[1, 0] - 2 < 1e-6


def test_compute_metrics_single_with_visible_column(tmp_path):
    """Labels CSV with a 'visible' coord column must not cause a reshape error.

    Regression test for the fix in compute_metrics_single that strips 'visible'
    coords before reshaping labels to (n_frames, n_keypoints, 2).
    Without the fix, n_keypoints x 3 cols reshaped as (-1, 2) yields the wrong
    keypoint count and raises: "operands could not be broadcast together".
    """
    n_frames = 4
    n_keypoints = 3
    keypoints = [f"kp{i}" for i in range(n_keypoints)]
    frames = [f"labeled-data/session/frame{i:04d}.png" for i in range(n_frames)]
    scorer = "scorer_test"

    # Labels CSV: x, y, visible per keypoint
    label_tuples = [
        (scorer, kp, coord)
        for kp in keypoints
        for coord in ("x", "y", "visible")
    ]
    rng = np.random.default_rng(0)
    label_data = {
        t: (rng.random(n_frames) * 100 if t[2] in ("x", "y") else np.full(n_frames, 2.0))
        for t in label_tuples
    }
    labels_df = pd.DataFrame(label_data, index=pd.Index(frames))
    labels_df.columns = pd.MultiIndex.from_tuples(
        label_tuples, names=["scorer", "bodyparts", "coords"]
    )
    labels_csv = tmp_path / "labels.csv"
    labels_df.to_csv(labels_csv)

    # Predictions CSV: x, y, likelihood per keypoint + 'set' column (triggers is_video=False)
    pred_tuples = [
        (scorer, kp, coord)
        for kp in keypoints
        for coord in ("x", "y", "likelihood")
    ] + [("set", "set", "set")]
    pred_data = {
        t: (rng.random(n_frames) * 100 if t[2] in ("x", "y")
            else (np.full(n_frames, 0.9) if t[2] == "likelihood"
                  else np.full(n_frames, "train")))
        for t in pred_tuples
    }
    preds_df = pd.DataFrame(pred_data, index=pd.Index(frames))
    preds_df.columns = pd.MultiIndex.from_tuples(
        pred_tuples, names=["scorer", "bodyparts", "coords"]
    )
    preds_csv = tmp_path / "predictions.csv"
    preds_df.to_csv(preds_csv)

    cfg = OmegaConf.create({
        "data": {"columns_for_singleview_pca": [], "mirrored_column_matches": []}
    })

    result = compute_metrics_single(cfg=cfg, labels_file=labels_csv, preds_file=preds_csv)

    assert result.pixel_error_df is not None
    # shape is (n_frames, n_keypoints) plus the appended 'set' column
    assert result.pixel_error_df.shape == (n_frames, n_keypoints + 1)
    assert list(result.pixel_error_df.columns[:n_keypoints]) == keypoints


def test_pca_singleview_reprojection_error(cfg, base_data_module):

    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_singleview",
        data_module=base_data_module,
        components_to_keep=3,
        empirical_epsilon_percentile=0.99,
        columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
    )
    kp_pca()
    pca_cols = np.zeros(base_data_module.dataset.num_keypoints, dtype=bool)
    pca_cols[kp_pca.columns_for_singleview_pca] = True

    # compute pca error on a random matrix, make sure corrects cols are floats or nans
    n_samples = 20
    n_keypoints = base_data_module.dataset.num_keypoints
    keypoint_data = np.random.randn(n_samples, n_keypoints, 2)
    error_per_keypoint = pca_singleview_reprojection_error(keypoint_data, kp_pca)
    assert error_per_keypoint.shape == (n_samples, n_keypoints)
    assert np.all(error_per_keypoint[:, pca_cols] >= 0.0)
    assert np.all(np.isnan(error_per_keypoint[:, ~pca_cols]))


def test_pca_multiview_reprojection_error(cfg, base_data_module):

    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_multiview",
        data_module=base_data_module,
        components_to_keep=3,
        empirical_epsilon_percentile=0.99,
        mirrored_column_matches=cfg.data.mirrored_column_matches,
    )
    kp_pca()
    pca_cols = np.zeros(base_data_module.dataset.num_keypoints, dtype=bool)
    assert kp_pca.mirrored_column_matches is not None
    pca_cols[np.array(list(kp_pca.mirrored_column_matches)).flatten()] = True

    # compute pca error on a random matrix, make sure corrects cols are floats or nans
    n_samples = 20
    n_keypoints = base_data_module.dataset.num_keypoints
    keypoint_data = np.random.randn(n_samples, n_keypoints, 2)
    error_per_keypoint = pca_multiview_reprojection_error(keypoint_data, kp_pca)
    assert error_per_keypoint.shape == (n_samples, n_keypoints)
    assert np.all(error_per_keypoint[:, pca_cols] >= 0.0)
    assert np.all(np.isnan(error_per_keypoint[:, ~pca_cols]))
