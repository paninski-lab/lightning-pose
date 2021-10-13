from pose_est_nets.losses import MaskedRegressionMSELoss
import torch
import numpy as np
import pytest
import yaml


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
    assert loss > 0.0


def test_get_losses_dict():
    from pose_est_nets.losses.losses import get_losses_dict

    out_dict = get_losses_dict(["pca"])
    assert "pca" in list(out_dict.keys())
    assert "temporal" not in list(out_dict.keys())
    assert type(out_dict) == dict

    pytest.raises(TypeError, get_losses_dict, ["bla"])


def test_yaml():
    stream = open("pose_est_nets/losses/default_hypers.yaml", "r")

    with open("pose_est_nets/losses/default_hypers.yaml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
