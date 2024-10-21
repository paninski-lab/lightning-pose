"""Test loss classes."""

import numpy as np
import pytest
import torch
from kornia.geometry.subpix import spatial_softmax2d

from lightning_pose.data.utils import generate_heatmaps
from lightning_pose.losses.losses import (
    HeatmapJSLoss,
    HeatmapKLLoss,
    HeatmapMSELoss,
    Loss,
    PCALoss,
    RegressionMSELoss,
    RegressionRMSELoss,
    TemporalLoss,
    UnimodalLoss,
    get_loss_classes,
)
from lightning_pose.utils.pca import format_multiview_data_for_pca

stage = "train"
device = "cpu"


def test_heatmap_mse_loss():
    heatmap_mse_loss = HeatmapMSELoss()

    # when predictions equal targets, should return zero
    targets = torch.ones((3, 7, 48, 48)) / (48 * 48)
    predictions = torch.ones_like(targets) / (48 * 48)
    loss, logs = heatmap_mse_loss(
        heatmaps_targ=targets,
        heatmaps_pred=predictions,
        stage=stage,
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
        heatmaps_targ=targets,
        heatmaps_pred=predictions,
        stage=stage,
    )
    assert loss > 0.0


def test_heatmap_kl_loss():
    heatmap_loss = HeatmapKLLoss()

    m = 100  # max pixel
    keypoints = m * torch.rand((3, 7, 2))
    targets = generate_heatmaps(keypoints, height=m, width=m, output_shape=(32, 32))
    predictions = targets.clone()

    loss, logs = heatmap_loss(
        heatmaps_targ=targets,
        heatmaps_pred=predictions,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert np.isclose(loss.detach().cpu().numpy(), 0.0, rtol=1e-5)
    assert logs[0]["name"] == "%s_heatmap_kl_loss" % stage
    assert logs[0]["value"] == loss / heatmap_loss.weight
    assert logs[1]["name"] == "heatmap_kl_weight"
    assert logs[1]["value"] == heatmap_loss.weight

    # when predictions have higher error, should return more positive value
    predictions = torch.roll(predictions, shifts=1, dims=0)
    loss2, logs = heatmap_loss(
        heatmaps_targ=targets,
        heatmaps_pred=predictions,
        stage=stage,
    )
    assert loss2 > loss


def test_heatmap_js_loss():
    heatmap_loss = HeatmapJSLoss()

    m = 100  # max pixel
    keypoints = m * torch.rand((3, 7, 2))
    targets = generate_heatmaps(keypoints, height=m, width=m, output_shape=(32, 32))
    predictions = targets.clone()

    loss, logs = heatmap_loss(
        heatmaps_targ=targets,
        heatmaps_pred=predictions,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert np.isclose(loss.detach().cpu().numpy(), 0.0, rtol=1e-5)
    assert logs[0]["name"] == "%s_heatmap_js_loss" % stage
    assert logs[0]["value"] == loss / heatmap_loss.weight
    assert logs[1]["name"] == "heatmap_js_weight"
    assert logs[1]["value"] == heatmap_loss.weight

    # when predictions have higher error, should return more positive value
    predictions = torch.roll(predictions, shifts=1, dims=0)
    loss2, logs = heatmap_loss(
        heatmaps_targ=targets,
        heatmaps_pred=predictions,
        stage=stage,
    )
    assert loss2 > loss


# Tests should pass whether device is cpu or cuda.
@pytest.mark.parametrize("device", ['cpu', f'cuda:{torch.cuda.current_device()}'])
def test_pca_singleview_loss(cfg, base_data_module, device):
    pca_loss = PCALoss(
        loss_name="pca_singleview",
        components_to_keep=2,
        data_module=base_data_module,
        columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
        device=device,
    )

    # ----------------------------
    # test pca loss on toy dataset
    # ----------------------------
    keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets, device=device)
    loss, logs = pca_loss(keypoints_pred, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_pca_singleview_loss" % stage
    assert logs[0]["value"] == loss / pca_loss.weight
    assert logs[1]["name"] == "pca_singleview_weight"
    assert logs[1]["value"] == pca_loss.weight


# Tests should pass whether device is cpu or cuda.
@pytest.mark.parametrize("device", ['cpu', f'cuda:{torch.cuda.current_device()}'])
def test_pca_multiview_loss(cfg, base_data_module, device):
    # raise exception when mirrored_column_matches arg is not provided
    with pytest.raises(ValueError):
        PCALoss(loss_name="pca_multiview", data_module=base_data_module)

    pca_loss = PCALoss(
        loss_name="pca_multiview",
        components_to_keep=3,
        data_module=base_data_module,
        mirrored_column_matches=cfg.data.mirrored_column_matches,
        device=device,
    )

    # ----------------------------
    # test pca loss on toy dataset
    # ----------------------------
    keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets, device=device)
    # shape = (batch_size, num_keypoints * 2)
    # this all happens in PCALoss.__call__() but keeping it since we want to look at pre reduction
    # loss
    keypoints_pred = keypoints_pred.reshape(
        keypoints_pred.shape[0], -1, 2
    )  # shape = (batch_size, num_keypoints, 2)
    keypoints_pred = format_multiview_data_for_pca(
        data_arr=keypoints_pred,
        mirrored_column_matches=cfg.data.mirrored_column_matches,
    )
    pre_reduction_loss = pca_loss.compute_loss(keypoints_pred)
    # shape = (num_samples, num_keypoints, num_views)
    pre_reduction_loss.shape == (
        keypoints_pred.shape[0],
        len(cfg.data.mirrored_column_matches[0]),
        2,  # for 2 views in this toy dataset
    )

    # draw some numbers again, and reshape within the class
    keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets, device=device)

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
    # TODO: this is not how we currently compute reprojection. we now add mean in the final stage
    pca_loss = PCALoss(
        loss_name="pca_multiview",
        components_to_keep=3,
        data_module=base_data_module,
        mirrored_column_matches=cfg.data.mirrored_column_matches,
        device=device,
    )

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
    temporal_loss = TemporalLoss(epsilon=0.0)

    # make sure zero is returned for constant predictions (along dim 0)
    predicted_keypoints = torch.ones(size=(12, 32))
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
    predicted_keypoints = torch.rand(size=(12, 32))
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss > 0.0

    # check against actual norm
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [np.sqrt(2.0), np.sqrt(2.0)]]
    )
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert loss.item() - 2 < 1e-6

    # test epsilon
    s2 = np.sqrt(2.0)
    s3 = np.sqrt(3.0)
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [s2, s2], [s3 + s2, s3 + s2]]
    )
    # [s2, s2] -> 2
    # [s3, s3] -> sqrt(6)
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    assert (loss.item() - (2 + np.sqrt(6))) < 1e-6

    temporal_loss = TemporalLoss(epsilon=2.1)
    loss, logs = temporal_loss(predicted_keypoints, stage=stage)
    # due to epsilon the "2" entry will be zeroed out
    assert (loss.item() - np.sqrt(6)) < 1e-6


def test_temporal_loss_multi_epsilon_rectification():
    batch_size = 6
    temporal_loss = TemporalLoss(epsilon=[0.1, 0.0, 0.5])
    # define a fake batch of loss values
    # all keypoints have the same value in the batch dimension
    loss_tensor = (
        torch.tensor([0.0, 1.0, 0.4], dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch_size - 1, 1)
    )

    rectified = temporal_loss.rectify_epsilon(loss_tensor)
    assert rectified.shape == torch.Size([batch_size - 1, 3])
    assert torch.all(rectified[:, 0] == 0.0)
    assert torch.all(rectified[:, 1] == 1.0)
    assert torch.all(rectified[:, 2] == 0.0)

    temporal_loss = TemporalLoss(epsilon=[0.1, 0.0, 0.3])
    rectified = temporal_loss.rectify_epsilon(loss_tensor)
    assert rectified.shape == torch.Size([batch_size - 1, 3])
    assert torch.all(rectified[:, 0] == 0.0)
    assert torch.all(rectified[:, 1] == 1.0)
    assert torch.allclose(rectified[:, 2], torch.tensor([0.1]))

    # each keypoint has different values in the batch dimension
    loss_tensor_fancier = torch.tensor(
        data=[[1.0, 2.0, 1.5], [0.05, 0.12, 0.2]], dtype=torch.float32
    )
    temporal_loss = TemporalLoss(epsilon=[0.1, 0.15, 0.3])
    rectified = temporal_loss.rectify_epsilon(loss_tensor_fancier)
    assert rectified.shape == (2, 3)
    assert torch.allclose(rectified[0, :], torch.tensor([0.9, 1.85, 1.2]))
    assert torch.allclose(rectified[1, :], torch.tensor([0.0, 0.0, 0.0]))


def test_unimodal_mse_loss():
    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device=device,
    )
    confidences = torch.rand(size=(batch_size, num_keypoints))

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
        keypoints_pred_augmented=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        confidences=confidences,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_unimodal_mse_loss" % stage
    assert logs[0]["value"] == loss / uni_loss.weight
    assert logs[1]["name"] == "unimodal_mse_weight"
    assert logs[1]["value"] == uni_loss.weight


def test_unimodal_kl_loss():
    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device=device,
    )
    confidences = torch.rand(size=(batch_size, num_keypoints))

    heatmaps_pred = spatial_softmax2d(
        torch.randn(
            size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
            device=device,
        )
    )
    uni_loss = UnimodalLoss(
        loss_name="unimodal_kl",
        original_image_height=img_size,
        original_image_width=img_size,
        downsampled_image_height=img_size_ds,
        downsampled_image_width=img_size_ds,
    )
    loss, logs = uni_loss(
        keypoints_pred_augmented=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        confidences=confidences,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_unimodal_kl_loss" % stage
    assert logs[0]["value"] == loss / uni_loss.weight
    assert logs[1]["name"] == "unimodal_kl_weight"
    assert logs[1]["value"] == uni_loss.weight


def test_unimodal_js_loss():
    img_size = 48
    img_size_ds = 32
    batch_size = 12
    num_keypoints = 16

    # make sure non-negative scalar is returned
    keypoints_pred = img_size * torch.rand(
        size=(batch_size, 2 * num_keypoints),
        device=device,
    )
    confidences = torch.rand(size=(batch_size, num_keypoints))

    heatmaps_pred = spatial_softmax2d(
        torch.randn(
            size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
            device=device,
        )
    )
    uni_loss = UnimodalLoss(
        loss_name="unimodal_js",
        original_image_height=img_size,
        original_image_width=img_size,
        downsampled_image_height=img_size_ds,
        downsampled_image_width=img_size_ds,
    )
    loss, logs = uni_loss(
        keypoints_pred_augmented=keypoints_pred,
        heatmaps_pred=heatmaps_pred,
        confidences=confidences,
        stage=stage,
    )
    assert loss.shape == torch.Size([])
    assert loss > 0.0
    assert logs[0]["name"] == "%s_unimodal_js_loss" % stage
    assert logs[0]["value"] == loss / uni_loss.weight
    assert logs[1]["name"] == "unimodal_js_weight"
    assert logs[1]["value"] == uni_loss.weight


def test_regression_mse_loss():
    mse_loss = RegressionMSELoss()

    # when predictions equal targets, should return zero
    true_keypoints = torch.ones(size=(12, 32))
    predicted_keypoints = torch.ones(size=(12, 32))
    loss, logs = mse_loss(true_keypoints, predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss == 0.0
    assert logs[0]["name"] == "%s_regression_mse_loss" % stage
    assert logs[0]["value"] == loss / mse_loss.weight
    assert logs[1]["name"] == "regression_mse_weight"
    assert logs[1]["value"] == mse_loss.weight

    # when predictions do not equal targets, should return positive value
    true_keypoints = torch.rand(size=(12, 32))
    predicted_keypoints = torch.rand(size=(12, 32))
    loss, logs = mse_loss(true_keypoints, predicted_keypoints, stage=stage)
    assert loss.shape == torch.Size([])
    assert loss > 0.0


def test_regression_rmse_loss():
    mse_loss = RegressionMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1
    rmse_loss = RegressionRMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1

    # when predictions equal targets, should return zero
    true_keypoints = torch.ones(size=(12, 32))
    predicted_keypoints = torch.ones(size=(12, 32))
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
    labels = 2 * torch.ones((batch_size, n))
    preds = torch.zeros((batch_size, n))
    true_rmse = 2.0
    mse, _ = mse_loss(labels, preds, stage=stage)
    rmse, _ = rmse_loss(labels, preds, stage=stage)
    assert rmse == true_rmse
    assert mse == true_rmse ** 2.0


def test_get_loss_classes():
    loss_classes = get_loss_classes()
    for loss_name, loss_class in loss_classes.items():
        assert issubclass(loss_class, Loss)
