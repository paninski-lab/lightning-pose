import torch
import numpy as np
import pytest
import yaml

from pose_est_nets.losses.losses import HeatmapWassersteinLoss


def test_regression_mse_loss():

    from pose_est_nets.losses.losses import RegressionMSELoss

    true_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )

    predicted_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )
    mse_loss = RegressionMSELoss()
    loss = mse_loss(true_keypoints, predicted_keypoints, logging=False)
    assert loss.shape == torch.Size([])
    assert loss > 0.0


def test_regression_rmse_loss():

    from pose_est_nets.losses.losses import RegressionMSELoss, RegressionRMSELoss

    n = 4
    batch_size = 3
    labels = 2 * torch.ones((batch_size, n), device="cpu")
    preds = torch.zeros((batch_size, n), device="cpu")
    true_rmse = 2.0

    mse_loss = RegressionMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1
    mse = mse_loss(labels, preds, logging=False)

    rmse_loss = RegressionRMSELoss(log_weight=np.log(0.5)) # set log_weight so weight=1
    rmse = rmse_loss(labels, preds, logging=False)

    assert rmse == true_rmse
    assert mse == true_rmse ** 2.0


def test_SingleView_PCA_loss():
    from pose_est_nets.losses.losses import SingleviewPCALoss, MultiviewPCALoss

    kept_evecs = torch.eye(n=4)[:, :2].T  # two eigenvecs, each 4D
    projection_to_obs = torch.randn(
        size=(10, 2)
    )  # random projection matrix from kept_evecs to obs
    obs = projection_to_obs @ kept_evecs  # make 10 observations
    mean = obs.mean(dim=0)
    good_arr_for_pca = obs - mean.unsqueeze(0)  # subtract mean
    epsilon = torch.tensor(0.0, device="cpu")
    reproj = (
        good_arr_for_pca @ kept_evecs.T @ kept_evecs
    )  # first matmul projects to 2D, second matmul projects back to 4D
    assert torch.allclose(
        (reproj - good_arr_for_pca), torch.zeros_like(input=reproj)
    )  # assert that reproj=good_arr_for_pca
    # now verify that the pca loss acknowledges this
    single_view_pca_loss = SingleviewPCALoss(
        keypoint_preds=obs,
        kept_eigenvectors=kept_evecs,
        mean=mean,
        epsilon=epsilon,
    )

    assert single_view_pca_loss == 0.0


def test_zero_removal():
    zeroes = torch.zeros((1, 2, 48, 48))
    ones = torch.ones((1, 3, 48, 48))
    one_example = torch.cat([zeroes, ones, zeroes], dim=1)
    second_example = torch.cat([ones, zeroes, zeroes], dim=1)
    batch = torch.cat([one_example, second_example], dim=0)

    # now zeroes check
    squeezed_batch = batch.reshape(batch.shape[0], batch.shape[1], -1)
    all_zeroes = torch.all(squeezed_batch == 0.0, dim=-1)
    assert (
        all_zeroes
        == torch.tensor(
            [
                [True, True, False, False, False, True, True],
                [False, False, False, True, True, True, True],
            ]
        )
    ).all()
    print(all_zeroes.shape)
    # mask = all_zeroes.reshape(all_zeroes.shape[0], all_zeroes.shape[1], 1, 1)
    assert batch[~all_zeroes].shape == (6, 48, 48)
    assert (batch[~all_zeroes].flatten() == 1.0).all()
    # print(torch.masked_select(batch, ~all_zeroes).shape)

    # print(batch[~mask].shape)
    # cat = torch.cat([ones, zeroes, ones, ones, zeroes])


def test_heatmap_mse_loss():
    # define the class
    from pose_est_nets.losses.losses import HeatmapMSELoss, HeatmapWassersteinLoss

    heatmap_mse_loss = HeatmapMSELoss()
    targets = torch.ones((3, 7, 48, 48)) / (48 * 48)
    predictions = (torch.ones_like(targets) / (48 * 48)) + 0.01 * torch.randn_like(
        targets
    )

    loss = heatmap_mse_loss(
        heatmaps_targ=targets, heatmaps_pred=predictions, logging=False
    )
    print("mse_loss:", loss)
    # assert loss == 0.0

    heatmap_wasser_loss = HeatmapWassersteinLoss()
    loss = heatmap_wasser_loss(
        heatmaps_targ=targets, heatmaps_pred=predictions, logging=False
    )
    print("wass_loss:", loss)


def test_temporal_loss():

    from pose_est_nets.losses.losses import TemporalLoss

    temporal_loss = TemporalLoss(epsilon=0.0)

    # make sure non-negative scalar is returned
    predicted_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )
    loss = temporal_loss(predicted_keypoints, logging=False)
    assert loss.shape == torch.Size([])
    assert loss > 0.0

    # make sure 0 is returned when no differences
    num_batch = 10
    num_keypoints = 4
    predicted_keypoints = torch.ones(
        size=(num_batch, num_keypoints),
        device="cpu",
    )
    loss = temporal_loss(predicted_keypoints, logging=False)
    assert loss == 0

    # compute actual norm
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [np.sqrt(2.0), np.sqrt(2.0)]], device="cpu"
    )
    loss = temporal_loss(predicted_keypoints, logging=False)
    assert loss.item() - 2 < 1e-6

    # test epsilon
    s2 = np.sqrt(2.0)
    s3 = np.sqrt(3.0)
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [s2, s2], [s3 + s2, s3 + s2]], device="cpu"
    )
    # [s2, s2] -> 2
    # [s3, s3] -> sqrt(6)
    loss = temporal_loss(predicted_keypoints, logging=False)
    assert (loss.item() - (2 + np.sqrt(6))) < 1e-6

    temporal_loss = TemporalLoss(epsilon=2.1)
    loss = temporal_loss(predicted_keypoints, logging=False)
    # due to epsilon the "2" entry will be zeroed out
    assert (loss.item() - np.sqrt(6)) < 1e-6


def test_unimodal_mse_loss():

    from pose_est_nets.losses.losses import UnimodalLoss

    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device="cpu",
    )
    heatmaps_pred = torch.ones(
        size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
        device="cpu",
    )
    uni_loss = UnimodalLoss(
        loss_name="unimodal_mse",
        original_image_height=img_size,
        original_image_width=img_size,
        downsampled_image_height=img_size_ds,
        downsampled_image_width=img_size_ds,
    )
    loss = uni_loss(
        keypoints_pred=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        logging=False,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0


def test_unimodal_wasserstein_loss():

    from pose_est_nets.losses.losses import UnimodalLoss

    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device="cpu",
    )
    heatmaps_pred = torch.ones(
        size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
        device="cpu",
    )
    uni_loss = UnimodalLoss(
        loss_name="unimodal_wasserstein",
        original_image_height=img_size,
        original_image_width=img_size,
        downsampled_image_height=img_size_ds,
        downsampled_image_width=img_size_ds,
    )
    loss = uni_loss(
        keypoints_pred=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        logging=False,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0
