"""Test data utils functionality."""

import copy
from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d
from kornia.geometry.transform import pyrup
import pytest
import torch

from lightning_pose.data.utils import generate_heatmaps

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_data_extractor():
    # TODO
    pass


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
        total_number,
        train_probability=train_prob,
        val_probability=val_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(
        total_number,
        train_probability=train_prob,
        val_probability=val_prob,
        test_probability=test_prob)
    assert out[0] == 80 and out[1] == 10 and out[2] == 10

    out = split_sizes_from_probabilities(total_number, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 15

    # test that extra samples end up in test
    out = split_sizes_from_probabilities(101, train_probability=0.7)
    assert out[0] == 70 and out[1] == 15 and out[2] == 16


def test_clean_any_nans():

    from lightning_pose.data.utils import clean_any_nans

    a = torch.randn(10, 7)
    a[0, 1] = float('nan')
    a[0, 3] = float('nan')
    a[3, 4] = float('nan')
    a[5, 6] = float('nan')

    # remove samples (defined as columns) that have nan values
    b = clean_any_nans(a, dim=0)
    assert b.shape == (10, 3)

    # remove samples (defined as rows) that have nan values
    c = clean_any_nans(a, dim=1)
    assert c.shape == (7, 7)


def test_count_frames(video_list):
    from lightning_pose.data.utils import count_frames
    import cv2
    num_frames = 0
    for video_file in video_list:
        cap = cv2.VideoCapture(video_file)
        num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    num_frames_ = count_frames(video_list)
    assert num_frames == num_frames_


def test_generate_heatmaps(cfg, heatmap_dataset):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width

    batch = heatmap_dataset.__getitem__(idx=0)
    heatmap_gt = batch["heatmaps"].unsqueeze(0)
    keypts_gt = batch["keypoints"].unsqueeze(0).reshape(1, -1, 2)
    heatmap_torch = generate_heatmaps(
        keypts_gt, height=im_height, width=im_width,
        output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(
        heatmap_gt.to(_TORCH_DEVICE), temperature=torch.tensor(100).to(_TORCH_DEVICE)
    )
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(
        heatmap_torch.to(_TORCH_DEVICE), temperature=torch.tensor(100).to(_TORCH_DEVICE)
    )
    preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
    confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

    assert (preds_gt == preds_torch).all()
    assert (confidences_gt == confidences_torch).all()

    # cleanup
    del batch
    del heatmap_gt, keypts_gt
    del softmaxes_gt, preds_gt, confidences_gt
    del softmaxes_torch, preds_torch, confidences_torch
    torch.cuda.empty_cache()  # remove tensors from gpu


def test_generate_heatmaps_weird_shape(cfg, toy_data_dir):

    from lightning_pose.utils.scripts import get_imgaug_transform, get_dataset

    img_shape = (384, 256)

    # update config
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.data.image_resize_dims.height = img_shape[0]
    cfg_tmp.data.image_resize_dims.width = img_shape[1]

    # build dataset with these new image dimensions
    imgaug_transform = get_imgaug_transform(cfg_tmp)
    dataset = get_dataset(
        cfg_tmp, data_dir=toy_data_dir, imgaug_transform=imgaug_transform,
    )

    # now same test as `test_generate_heatmaps`
    batch = dataset.__getitem__(idx=0)
    heatmap_gt = batch["heatmaps"].unsqueeze(0)
    keypts_gt = batch["keypoints"].unsqueeze(0).reshape(1, -1, 2)
    heatmap_torch = generate_heatmaps(
        keypts_gt, height=img_shape[0], width=img_shape[1],
        output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(
        heatmap_gt.to(_TORCH_DEVICE), temperature=torch.tensor(100).to(_TORCH_DEVICE)
    )
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(
        heatmap_torch.to(_TORCH_DEVICE), temperature=torch.tensor(100).to(_TORCH_DEVICE)
    )
    preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
    confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

    assert (preds_gt == preds_torch).all()
    assert (confidences_gt == confidences_torch).all()

    # cleanup
    del batch
    del heatmap_gt, keypts_gt
    del softmaxes_gt, preds_gt, confidences_gt
    del softmaxes_torch, preds_torch, confidences_torch
    torch.cuda.empty_cache()  # remove tensors from gpu


# def test_heatmap_generation():
#
#     # want to compare the output of our manual function to kornia's
#     # if it works, move to kornia
#
#     # a batch size of 2, with 3 keypoints per batch.
#     from time import time
#     from kornia.geometry.subpix import render_gaussian2d
#
#     batch_dim_1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) * 10.0
#     batch_dim_2 = batch_dim_1 * 2.0
#     data = torch.stack((batch_dim_1, batch_dim_2), dim=0)
#     t_s = time()
#     fake_heatmaps = generate_heatmaps(
#         keypoints=data,
#         height=256,
#         width=256,
#         output_shape=(64, 64),
#         normalize=True,
#         nan_heatmap_mode="zero",
#     )
#     t_e = time()
#     t_ours = t_e - t_s
#     t_s = time()
#     data[:, :, 0] *= 64.0 / 256.0  # make it 4 times smaller
#     data[:, :, 1] *= 64.0 / 256.0  # make it 4 times smaller
#     kornia_heatmaps = render_gaussian2d(
#         mean=data.reshape(-1, 2), std=torch.tensor((1.0, 1.0)), size=(64, 64)
#     )
#     t_e = time()
#     t_kornia = t_e - t_s
#     print(kornia_heatmaps[0, :, :].flatten())
#     print(fake_heatmaps[0, :, :].flatten())
#     print((kornia_heatmaps[0, :, :].flatten()).sum())
#     print((fake_heatmaps[0, :, :].flatten().sum()))
#     kornia_heatmaps = kornia_heatmaps.reshape(2, 3, 64, 64)
#     kornia_min_max = (kornia_heatmaps.min(), kornia_heatmaps.max())
#     print(kornia_min_max)
#     our_min_max = (fake_heatmaps.min(), fake_heatmaps.max())
#     print(our_min_max)
#     data_1 = data.reshape(-1, 2)
#     data_2 = data_1.reshape(2, 3, 2)
#     (data == data_2).all()
#     kornia_heatmaps.shape
#     fake_keypoints = torch.tensor(3)
