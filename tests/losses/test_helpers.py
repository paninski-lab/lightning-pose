"""Test loss helper functions."""

import numpy as np
import torch


def test_empirical_epsilon():

    from lightning_pose.losses.helpers import EmpiricalEpsilon

    ee = EmpiricalEpsilon(percentile=90)

    # numpy array as input
    loss = np.arange(101, dtype="float")
    p = ee(loss)
    assert p == 90

    # pytorch tensor as input
    p = ee(torch.tensor(loss))
    assert p == 90

    # test w/ nans
    loss_nans = np.copy(loss)
    loss_nans[1::2] = np.nan
    p = ee(loss_nans)
    assert p == 90


def test_convert_dict_values_to_tensor():

    from lightning_pose.losses.helpers import convert_dict_values_to_tensors

    test_dict = {
        "param_a": 4.0,
        "param_b": 10.1,
        "param_c": 4,
    }
    test_dict_tensor = convert_dict_values_to_tensors(test_dict, device="cpu")
    for key, val in test_dict_tensor.items():
        assert isinstance(val, torch.Tensor)
        assert val.dtype == torch.float32
