from pose_est_nets.losses.regression_loss import MaskedRegressionMSELoss
import torch


def test_masked_regression_loss():

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
    assert (loss > 0.0).all()
