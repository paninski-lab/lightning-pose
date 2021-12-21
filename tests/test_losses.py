import torch
import numpy as np
import pytest
import yaml


def test_masked_regression_loss():

    from pose_est_nets.losses.losses import MaskedRegressionMSELoss, MaskedRMSELoss

    true_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )

    predicted_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )
    loss = MaskedRMSELoss(true_keypoints, predicted_keypoints)
    assert loss.shape == torch.Size([])
    assert loss > 0.0

    n = 4
    batch_size = 3
    labels = 2 * torch.ones((batch_size, n), device="cpu")
    preds = torch.zeros((batch_size, n), device="cpu")
    true_rmse = 2.0  # sqrt(n * (1/n))
    assert MaskedRMSELoss(labels, preds) == true_rmse
    assert MaskedRegressionMSELoss(labels, preds) == true_rmse ** 2.0


def test_masked_RMSE_loss():

    from pose_est_nets.losses.losses import MaskedRegressionMSELoss

    true_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )

    predicted_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )
    loss = MaskedRegressionMSELoss(true_keypoints, predicted_keypoints)
    assert loss.shape == torch.Size([])
    assert loss > 0.0


def test_TemporalLoss():

    from pose_est_nets.losses.losses import TemporalLoss

    # make sure non-negative scalar is returned
    predicted_keypoints = torch.rand(
        size=(12, 32),
        device="cpu",
    )
    loss = TemporalLoss(predicted_keypoints, epsilon=torch.Tensor([0.0], device="cpu"))
    assert loss.shape == torch.Size([])
    assert loss > 0.0

    # make sure 0 is returned when no differences
    num_batch = 10
    num_keypoints = 4
    predicted_keypoints = torch.ones(
        size=(num_batch, num_keypoints),
        device="cpu",
    )
    loss = TemporalLoss(predicted_keypoints, epsilon=torch.Tensor([0.0], device="cpu"))
    assert loss == 0

    # compute actual norm
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [np.sqrt(2.0), np.sqrt(2.0)]], device="cpu"
    )
    loss = TemporalLoss(predicted_keypoints, epsilon=torch.Tensor([0.0], device="cpu"))
    assert loss.item() - 2 < 1e-6

    # test epsilon
    s2 = np.sqrt(2.0)
    s3 = np.sqrt(3.0)
    predicted_keypoints = torch.Tensor(
        [[0.0, 0.0], [s2, s2], [s3 + s2, s3 + s2]], device="cpu"
    )
    # [s2, s2] -> 2
    # [s3, s3] -> sqrt(6)
    loss = TemporalLoss(predicted_keypoints, epsilon=torch.Tensor([0.0], device="cpu"))
    assert (loss.item() - (2 + np.sqrt(6))) < 1e-6

    loss = TemporalLoss(predicted_keypoints, epsilon=torch.Tensor([2.1], device="cpu"))
    assert (
        loss.item() - np.sqrt(6)
    ) < 1e-6  # due to epsilon the "2" entry will be zeroed out


def test_get_losses_dict():
    from pose_est_nets.losses.losses import get_losses_dict

    out_dict = get_losses_dict(["pca_multiview"])
    assert "pca_multiview" in list(out_dict.keys())
    assert "temporal" not in list(out_dict.keys())
    assert type(out_dict) == dict

    # test outdated because we changed type checking on this
    # pytest.raises(TypeError, get_losses_dict, ["bla"])

    # Biggest thing is to make sure shapes match: Matt


def test_SingleView_PCA_loss():
    from pose_est_nets.losses.losses import SingleviewPCALoss

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
