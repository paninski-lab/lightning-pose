import torch
import numpy as np
import pytest
import yaml


def test_format_multiview_data_for_pca():

    from pose_est_nets.datasets.preprocessing import format_multiview_data_for_pca

    n_batches = 12
    n_keypoints = 20
    keypoints = torch.rand(
        size=(n_batches, n_keypoints, 2),
        device="cpu",
    )

    # basic two-view functionality
    column_matches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    arr = format_multiview_data_for_pca(keypoints, column_matches)
    assert arr.shape == torch.Size([
        2 * len(column_matches),
        n_batches * len(column_matches[0])
    ])

    # basic error checking
    column_matches = [[0, 1, 2, 3], [4, 5, 6]]
    with pytest.raises(AssertionError):
        arr = format_multiview_data_for_pca(keypoints, column_matches)

    # basic three-view functionality
    column_matches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    arr = format_multiview_data_for_pca(keypoints, column_matches)
    assert arr.shape == torch.Size([
        2 * len(column_matches),
        n_batches * len(column_matches[0])
    ])
