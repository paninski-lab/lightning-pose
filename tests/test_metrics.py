"""Test metrics module."""

import numpy as np


def test_pixel_error():

    from lightning_pose.metrics import pixel_error

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

    from lightning_pose.metrics import temporal_norm

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


def test_pca_singleview_reprojection_error(cfg, base_data_module):

    from lightning_pose.metrics import pca_singleview_reprojection_error
    from lightning_pose.utils.pca import KeypointPCA

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

    from lightning_pose.metrics import pca_multiview_reprojection_error
    from lightning_pose.utils.pca import KeypointPCA

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
    pca_cols[np.array(list(kp_pca.mirrored_column_matches)).flatten()] = True

    # compute pca error on a random matrix, make sure corrects cols are floats or nans
    n_samples = 20
    n_keypoints = base_data_module.dataset.num_keypoints
    keypoint_data = np.random.randn(n_samples, n_keypoints, 2)
    error_per_keypoint = pca_multiview_reprojection_error(keypoint_data, kp_pca)
    assert error_per_keypoint.shape == (n_samples, n_keypoints)
    assert np.all(error_per_keypoint[:, pca_cols] >= 0.0)
    assert np.all(np.isnan(error_per_keypoint[:, ~pca_cols]))
