"""Test data utils functionality."""

import pytest
import torch


def test_split_sizes_from_probabilities():

    from lightning_pose.data.utils import split_sizes_from_probabilities

    # make sure we count examples properly
    total_number = 100
    train_prob = 0.8
    val_prob = 0.1
    test_prob = 0.1

    out = split_sizes_from_probabilities(total_number, train_probability=train_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number, train_probability=train_prob, val_probability=val_prob
    )
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number,
        train_probability=train_prob,
        val_probability=val_prob,
        test_probability=test_prob,
    )
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(total_number, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 15

    # test that extra samples end up in test
    out = split_sizes_from_probabilities(101, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 16

    # make sure we have at least one example in the validation set
    total_number = 10
    train_prob = 0.95
    val_prob = 0.05
    out = split_sizes_from_probabilities(
        total_number, train_probability=train_prob, val_probability=val_prob
    )
    assert sum(out) == total_number
    assert out[0] == 9
    assert out[1] == 1
    assert out[2] == 0

    # make sure an error is raised if there are not enough labeled frames
    total_number = 1
    with pytest.raises(ValueError):
        split_sizes_from_probabilities(total_number, train_probability=train_prob)


def test_clean_any_nans():

    from lightning_pose.data.utils import clean_any_nans

    a = torch.randn(10, 7)
    a[0, 1] = float("nan")
    a[0, 3] = float("nan")
    a[3, 4] = float("nan")
    a[5, 6] = float("nan")

    # remove samples (defined as columns) that have nan values
    b = clean_any_nans(a, dim=0)
    assert b.shape == (10, 3)

    # remove samples (defined as rows) that have nan values
    c = clean_any_nans(a, dim=1)
    assert c.shape == (7, 7)


def test_count_frames(video_list):
    from lightning_pose.data.utils import count_frames

    num_frames = count_frames(video_list[0])

    assert num_frames == 994


def test_compute_num_train_frames():

    from lightning_pose.data.utils import compute_num_train_frames

    len_train_data = 10

    # correctly defaults to data length with no train_frame arg
    n_frames = compute_num_train_frames(len_train_data, train_frames=None)
    assert n_frames == len_train_data

    # correctly defaults to data length when train_frame arg too large
    n_frames = compute_num_train_frames(len_train_data, train_frames=len_train_data + 1)
    assert n_frames == len_train_data

    # correctly defaults to data length when train_frame=1
    n_frames = compute_num_train_frames(len_train_data, train_frames=1)
    assert n_frames == len_train_data

    # correctly uses integer
    n_frames = compute_num_train_frames(len_train_data, train_frames=5)
    assert n_frames == 5

    # correctly uses fraction
    n_frames = compute_num_train_frames(len_train_data, train_frames=0.5)
    assert n_frames == 5

    n_frames = compute_num_train_frames(len_train_data, train_frames=0.2)
    assert n_frames == 2

    # train_frames must be positive
    with pytest.raises(ValueError):
        compute_num_train_frames(len_train_data, train_frames=-1)



def test_undo_affine_transform():

    from lightning_pose.data.utils import undo_affine_transform

    seq_len = 5
    n_keypoints = 6
    keypoints = torch.normal(mean=torch.zeros((seq_len, n_keypoints, 2)))

    # test single transform
    torch.manual_seed(0)
    transform_mat = torch.normal(mean=torch.zeros((2, 3)))
    keypoints_aug = torch.matmul(keypoints, transform_mat[:, :2].T) + transform_mat[:, -1]
    keypoints_noaug = undo_affine_transform(keypoints_aug, transform_mat)
    assert torch.allclose(keypoints, keypoints_noaug, atol=1e-4)

    # test individual transforms
    torch.manual_seed(0)
    transform_mat = torch.normal(mean=torch.zeros((seq_len, 2, 3)))
    keypoints_aug = torch.bmm(
        keypoints, transform_mat[:, :, :2].transpose(2, 1)
    ) + transform_mat[:, :, -1].unsqueeze(1)
    keypoints_noaug = undo_affine_transform(keypoints_aug, transform_mat)
    assert torch.allclose(keypoints, keypoints_noaug, atol=1e-4)


def test_undo_affine_transform_batch():

    from lightning_pose.data.utils import undo_affine_transform_batch

    seq_len = 5
    n_keypoints = 6

    # test single transform, single view
    torch.manual_seed(0)
    keypoints = torch.normal(mean=torch.zeros((seq_len, n_keypoints, 2)))
    transform_mat = torch.normal(mean=torch.zeros((2, 3)))
    keypoints_aug = torch.matmul(keypoints, transform_mat[:, :2].T) + transform_mat[:, -1]
    keypoints_aug = keypoints_aug.reshape((keypoints.shape[0], -1))
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints_aug,
        transforms=transform_mat,
        is_multiview=False,
    )
    assert torch.allclose(keypoints.reshape(keypoints_noaug.shape), keypoints_noaug, atol=1e-4)

    # test individual transforms, single view
    torch.manual_seed(1)
    keypoints = torch.normal(mean=torch.zeros((seq_len, n_keypoints, 2)))
    transform_mat = torch.normal(mean=torch.zeros((seq_len, 2, 3)))
    keypoints_aug = torch.bmm(
        keypoints, transform_mat[:, :, :2].transpose(2, 1)
    ) + transform_mat[:, :, -1].unsqueeze(1)
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints_aug.reshape((keypoints.shape[0], -1)),
        transforms=transform_mat,
        is_multiview=False,
    )
    assert torch.allclose(keypoints.reshape(keypoints_noaug.shape), keypoints_noaug, atol=1e-4)

    # test single transform, multi-view
    n_views = 3
    torch.manual_seed(2)
    keypoints = torch.normal(mean=torch.zeros((seq_len, n_keypoints * n_views, 2)))
    transform_mat = torch.normal(mean=torch.zeros((2, 3)))
    transform_mat_views = transform_mat.repeat(n_views, 1, 1)
    keypoints_aug = torch.matmul(keypoints, transform_mat[:, :2].T) + transform_mat[:, -1]
    keypoints_aug = keypoints_aug.reshape((keypoints.shape[0], -1))
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints_aug,
        transforms=transform_mat_views,
        is_multiview=True,
    )
    assert torch.allclose(keypoints.reshape(keypoints_noaug.shape), keypoints_noaug, atol=1e-4)

    # test different transforms, multi-view
    n_views = 3
    keypoints = []
    transforms = []
    keypoints_aug = []
    for v, _view in enumerate(range(n_views)):
        torch.manual_seed(v)
        # create keypoints/transforms for this view
        keypoints_v = torch.normal(mean=torch.zeros((seq_len, n_keypoints, 2)))
        transform_mat_v = torch.normal(mean=torch.zeros((2, 3)))
        keypoints_aug_v = torch.matmul(
            keypoints_v, transform_mat_v[:, :2].T
        ) + transform_mat_v[:, -1]
        # append to other views
        keypoints.append(keypoints_v.reshape((keypoints_v.shape[0], -1)))
        transforms.append(transform_mat_v)
        keypoints_aug.append(keypoints_aug_v.reshape((keypoints_v.shape[0], -1)))
    # concat across views
    keypoints = torch.concat(keypoints, dim=-1)
    keypoints_aug = torch.concat(keypoints_aug, dim=-1)
    transforms = torch.stack(transforms, dim=0)
    # test
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints_aug,
        transforms=transforms,
        is_multiview=True,
    )
    assert torch.allclose(keypoints, keypoints_noaug, atol=1e-4)

    # repeat, but with different ordering of reshaping
    keypoints = []
    transforms = []
    keypoints_aug = []
    for v, _view in enumerate(range(n_views)):
        torch.manual_seed(v)
        # create keypoints/transforms for this view
        keypoints_v = torch.normal(mean=torch.zeros((seq_len, n_keypoints, 2)))
        transform_mat_v = torch.normal(mean=torch.zeros((2, 3)))
        keypoints_aug_v = torch.matmul(
            keypoints_v, transform_mat_v[:, :2].T
        ) + transform_mat_v[:, -1]
        # append to other views
        keypoints.append(keypoints_v)
        transforms.append(transform_mat_v)
        keypoints_aug.append(keypoints_aug_v)
    # concat across views
    keypoints = torch.concat(keypoints, dim=1)
    keypoints_aug = torch.concat(keypoints_aug, dim=1)
    transforms = torch.stack(transforms, dim=0)
    # test
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints_aug.reshape((keypoints_aug.shape[0], -1)),
        transforms=transforms,
        is_multiview=True,
    )
    assert torch.allclose(keypoints.reshape(keypoints_noaug.shape), keypoints_noaug, atol=1e-4)

    # test no transform, single view
    torch.manual_seed(0)
    keypoints = torch.normal(mean=torch.zeros((seq_len, n_keypoints * 2)))
    transform_mat = torch.normal(mean=torch.zeros((seq_len, 1)))
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints,
        transforms=transform_mat,
        is_multiview=False,
    )
    assert torch.allclose(keypoints, keypoints_noaug, atol=1e-4)

    # test no transform, multiview
    torch.manual_seed(0)
    keypoints = torch.normal(mean=torch.zeros((seq_len, n_keypoints * 2)))
    transform_mat = torch.normal(mean=torch.zeros((seq_len, 1)))
    keypoints_noaug = undo_affine_transform_batch(
        keypoints_augmented=keypoints,
        transforms=transform_mat,
        is_multiview=True,
    )
    assert torch.allclose(keypoints, keypoints_noaug, atol=1e-4)
