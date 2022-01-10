"""Test loss classes."""

import torch
import numpy as np
import pytest
import yaml


stage = "train"
device = "cpu"


def test_heatmap_mse_loss():

    from lightning_pose.losses.losses import HeatmapMSELoss

    heatmap_mse_loss = HeatmapMSELoss()

    # when predictions equal targets, should return zero
    targets = torch.ones((3, 7, 48, 48)) / (48 * 48)
    predictions = torch.ones_like(targets) / (48 * 48)
    loss, logs = heatmap_mse_loss(
        heatmaps_targ=targets, heatmaps_pred=predictions, stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert loss == 0.0
    assert logs[0]["name"] == "%s_heatmap_mse_loss" % stage
    assert logs[0]["value"] == loss / heatmap_mse_loss.weight
    assert logs[1]["name"] == "heatmap_mse_weight"
    assert logs[1]["value"] == heatmap_mse_loss.weight

    # when predictions do not equal targets, should return positive value
    predictions = (torch.ones_like(targets) / (48 * 48)) + 0.01 * torch.randn_like(
        targets
    )
    loss, logs = heatmap_mse_loss(
        heatmaps_targ=targets, heatmaps_pred=predictions, stage=stage,
    )
    assert loss > 0.0


def test_heatmap_wasserstein_loss():

    from lightning_pose.losses.losses import HeatmapWassersteinLoss

    heatmap_wasser_loss = HeatmapWassersteinLoss()

    targets = torch.ones((3, 7, 48, 48)) / (48 * 48)
    # note: wasserstein loss fails when predictions exactly equal targets; need to look
    # into kornia to see why this might be the case
    predictions = (torch.ones_like(targets) / (48 * 48)) + 0.000001 * torch.randn_like(
        targets
    )

    loss, logs = heatmap_wasser_loss(
        heatmaps_targ=targets, heatmaps_pred=predictions, stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert np.isclose(loss.detach().cpu().numpy(), 0.0, rtol=1e-5)
    assert logs[0]["name"] == "%s_heatmap_wasserstein_loss" % stage
    assert logs[0]["value"] == loss / heatmap_wasser_loss.weight
    assert logs[1]["name"] == "heatmap_wasserstein_weight"
    assert logs[1]["value"] == heatmap_wasser_loss.weight

    # when predictions have higher error, should return more positive value
    predictions = (torch.ones_like(targets) / (48 * 48)) + 0.1 * torch.randn_like(
        targets
    )
    loss2, logs = heatmap_wasser_loss(
        heatmaps_targ=targets, heatmaps_pred=predictions, stage=stage,
    )
    assert loss2 > loss


def test_pca_singleview_loss(base_data_module):
    # TODO
    # from lightning_pose.losses.losses import PCALoss
    #
    # pca_loss = PCALoss(
    #     loss_name="pca_singleview", components_to_keep=2, data_module=base_data_module,
    # )
    #
    # # test pca loss on toy dataset
    # keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets)
    # loss, logs = pca_loss(keypoints_pred, stage=stage)

    # test pca loss on fake dataset
    # # make two eigenvecs, each 4D
    # kept_evecs = torch.eye(n=4, device=device)[:, :2].T
    # # random projection matrix from kept_evecs to obs
    # projection_to_obs = torch.randn(size=(10, 2), device=device)
    # # make observations
    # obs = projection_to_obs @ kept_evecs
    # mean = obs.mean(dim=0)
    # good_arr_for_pca = obs - mean.unsqueeze(0)  # subtract mean
    # # first matmul projects to 2D, second matmul projects back to 4D
    # reproj = good_arr_for_pca @ kept_evecs.T @ kept_evecs
    # # assert that reproj=good_arr_for_pca
    # assert torch.allclose(reproj - good_arr_for_pca, torch.zeros_like(input=reproj))
    #
    # # replace pca loss param
    #
    # # now verify the pca loss
    # pca_loss = SingleviewPCALoss(
    #     keypoint_preds=obs,
    #     kept_eigenvectors=kept_evecs,
    #     mean=mean,
    # )

    # assert single_view_pca_loss == 0.0
    pass


def test_pca_multiview_loss(cfg, base_data_module):

    from lightning_pose.losses.losses import PCALoss

    # raise exception when mirrored_column_matches arg is not provided
    with pytest.raises(ValueError):
        PCALoss(loss_name="pca_multiview", data_module=base_data_module)

    pca_loss = PCALoss(
        loss_name="pca_multiview", components_to_keep=3, data_module=base_data_module,
        mirrored_column_matches=cfg.data.mirrored_column_matches, device=device,
    )

    # ----------------------------
    # test pca loss on toy dataset
    # ----------------------------
    keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets)
    loss, logs = pca_loss(keypoints_pred, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_pca_multiview_loss" % stage
    assert logs[0]["value"] == loss / pca_loss.weight
    assert logs[1]["name"] == "pca_multiview_weight"
    assert logs[1]["value"] == pca_loss.weight

    # -----------------------------
    # test pca loss on fake dataset
    # -----------------------------
    # make three eigenvecs, each 4D
    kept_evecs = torch.eye(n=4, device=device)[:, :3].T
    # random projection matrix from kept_evecs to obs
    projection_to_obs = torch.randn(size=(10, 3), device=device)
    # make observations
    obs = projection_to_obs @ kept_evecs
    mean = obs.mean(dim=0)

    good_arr_for_pca = obs - mean.unsqueeze(0)  # subtract mean
    # first matmul projects to 2D, second matmul projects back to 4D
    reproj = good_arr_for_pca @ kept_evecs.T @ kept_evecs

    # assert that reproj=good_arr_for_pca
    assert torch.allclose(reproj - good_arr_for_pca, torch.zeros_like(input=reproj))

    # replace pca loss param
    pca_loss.pca.parameters["kept_eigenvectors"] = kept_evecs
    pca_loss.pca.parameters["mean"] = mean
    pca_loss.pca.mirrored_column_matches = [[0], [1]]

    # verify
    loss, logs = pca_loss(obs, stage=stage)
    assert loss == 0.0


def test_temporal_loss():

    from lightning_pose.losses.losses import TemporalLoss

    temporal_loss = TemporalLoss(epsilon=0.0)

    # make sure zero is returned for constant predictions (along dim 0)
    predicted_keypoints = torch.ones(size=(12, 32), device=device)
    predicted_keypoints[:, 1] = 2
    predicted_keypoints[:, 2] = 4
    predicted_keypoints[:, 3] = 8
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss == 0.0
    assert logs[0]["name"] == "%s_temporal_loss" % stage
    assert logs[0]["value"] == loss / temporal_loss.weight
    assert logs[1]["name"] == "temporal_weight"
    assert logs[1]["value"] == temporal_loss.weight

    # make sure non-negative scalar is returned
    predicted_keypoints = torch.rand(size=(12, 32), device=device)
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss > 0.0

    # check against actual norm
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [np.sqrt(2.0), np.sqrt(2.0)]], device=device
    )
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert loss.item() - 2 < 1e-6

    # test epsilon
    s2 = np.sqrt(2.0)
    s3 = np.sqrt(3.0)
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [s2, s2], [s3 + s2, s3 + s2]], device=device
    )
    # [s2, s2] -> 2
    # [s3, s3] -> sqrt(6)
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert (loss.item() - (2 + np.sqrt(6))) < 1e-6

    temporal_loss = TemporalLoss(epsilon=2.1)
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    # due to epsilon the "2" entry will be zeroed out
    assert (loss.item() - np.sqrt(6)) < 1e-6


def test_unimodal_mse_loss():

    from lightning_pose.losses.losses import UnimodalLoss

    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device=device,
    )
    heatmaps_pred = torch.ones(
        size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
        device=device,
    )
    uni_loss = UnimodalLoss(
        loss_name="unimodal_mse",
        original_image_height=img_size,
        original_image_width=img_size,
        downsampled_image_height=img_size_ds,
        downsampled_image_width=img_size_ds,
    )
    loss, logs = uni_loss(
        keypoints_pred=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_unimodal_mse_loss" % stage
    assert logs[0]["value"] == loss / uni_loss.weight
    assert logs[1]["name"] == "unimodal_mse_weight"
    assert logs[1]["value"] == uni_loss.weight


def test_unimodal_wasserstein_loss():

    from lightning_pose.losses.losses import UnimodalLoss

    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device=device,
    )
    heatmaps_pred = torch.ones(
        size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
        device=device,
    )
    uni_loss = UnimodalLoss(
        loss_name="unimodal_wasserstein",
        original_image_height=img_size,
        original_image_width=img_size,
        downsampled_image_height=img_size_ds,
        downsampled_image_width=img_size_ds,
    )
    loss, logs = uni_loss(
        keypoints_pred=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_unimodal_wasserstein_loss" % stage
    assert logs[0]["value"] == loss / uni_loss.weight
    assert logs[1]["name"] == "unimodal_wasserstein_weight"
    assert logs[1]["value"] == uni_loss.weight


def test_regression_mse_loss():

    from lightning_pose.losses.losses import RegressionMSELoss

    mse_loss = RegressionMSELoss()

    # when predictions equal targets, should return zero
    true_keypoints = torch.ones(size=(12, 32), device=device)
    predicted_keypoints = torch.ones(size=(12, 32), device=device)
    loss, logs = mse_loss(true_keypoints, predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss == 0.0
    assert logs[0]["name"] == "%s_regression_mse_loss" % stage
    assert logs[0]["value"] == loss / mse_loss.weight
    assert logs[1]["name"] == "regression_mse_weight"
    assert logs[1]["value"] == mse_loss.weight

    # when predictions do not equal targets, should return positive value
    true_keypoints = torch.rand(size=(12, 32), device=device)
    predicted_keypoints = torch.rand(size=(12, 32), device=device)
    loss, logs = mse_loss(true_keypoints, predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss > 0.0


def test_regression_rmse_loss():

    from lightning_pose.losses.losses import RegressionMSELoss, RegressionRMSELoss

    mse_loss = RegressionMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1
    rmse_loss = RegressionRMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1

    # when predictions equal targets, should return zero
    true_keypoints = torch.ones(size=(12, 32), device=device)
    predicted_keypoints = torch.ones(size=(12, 32), device=device)
    loss, logs = rmse_loss(true_keypoints, predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss == 0.0
    assert logs[0]["name"] == "%s_rmse_loss" % stage
    assert logs[0]["value"] == loss / mse_loss.weight
    assert logs[1]["name"] == "rmse_weight"
    assert logs[1]["value"] == mse_loss.weight

    # compute exact rmse from mse
    n = 4
    batch_size = 3
    labels = 2 * torch.ones((batch_size, n), device=device)
    preds = torch.zeros((batch_size, n), device=device)
    true_rmse = 2.0
    mse, _ = mse_loss(labels, preds, stage=stage)
    rmse, _ = rmse_loss(labels, preds, stage=stage)
    assert rmse == true_rmse
    assert mse == true_rmse ** 2.0


def test_get_loss_classes():

    from lightning_pose.losses.losses import Loss, get_loss_classes

    loss_classes = get_loss_classes()
    for loss_name, loss_class in loss_classes.items():
        assert issubclass(loss_class, Loss)


# def test_zero_removal():
#     zeroes = torch.zeros((1, 2, 48, 48))
#     ones = torch.ones((1, 3, 48, 48))
#     one_example = torch.cat([zeroes, ones, zeroes], dim=1)
#     second_example = torch.cat([ones, zeroes, zeroes], dim=1)
#     batch = torch.cat([one_example, second_example], dim=0)
#
#     # now zeroes check
#     squeezed_batch = batch.reshape(batch.shape[0], batch.shape[1], -1)
#     all_zeroes = torch.all(squeezed_batch == 0.0, dim=-1)
#     assert (
#         all_zeroes
#         == torch.tensor(
#             [
#                 [True, True, False, False, False, True, True],
#                 [False, False, False, True, True, True, True],
#             ]
#         )
#     ).all()
#     print(all_zeroes.shape)
#     # mask = all_zeroes.reshape(all_zeroes.shape[0], all_zeroes.shape[1], 1, 1)
#     assert batch[~all_zeroes].shape == (6, 48, 48)
#     assert (batch[~all_zeroes].flatten() == 1.0).all()
