"""Test data utils functionality."""

import copy

import numpy as np
import pytest
import torch
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d

from lightning_pose.data.utils import (
    convert_original_to_model_coords,
    generate_heatmaps,
    original_to_model,
)


def test_data_extractor(base_data_module_combined, multiview_heatmap_data_module_combined):

    from lightning_pose.data.utils import DataExtractor

    # ---------------------------
    # supervised single view
    # ---------------------------
    num_frames = (
        len(base_data_module_combined.dataset)
        * base_data_module_combined.train_probability
    )
    keypoint_tensor, _ = DataExtractor(
        data_module=base_data_module_combined, cond="train"
    )()
    assert keypoint_tensor.shape == (num_frames, 34)  # 72 = 0.8 * 90 images, 17 * 2 coordinates

    keypoint_tensor, images_tensor = DataExtractor(
        data_module=base_data_module_combined, cond="train", extract_images=True
    )()
    assert images_tensor.shape == (num_frames, 3, 128, 128)

    # ---------------------------
    # supervised multiview
    # ---------------------------
    num_frames = (
        len(multiview_heatmap_data_module_combined.dataset)
        * multiview_heatmap_data_module_combined.train_probability
    )
    keypoint_tensor, _ = DataExtractor(
        data_module=multiview_heatmap_data_module_combined, cond="train"
    )()
    assert keypoint_tensor.shape == (num_frames, 28)  # 72 = 0.8 * 90 images, 7 * 2 * 2 coords


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


def test_generate_heatmaps(cfg, heatmap_dataset):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width

    batch = heatmap_dataset.__getitem__(idx=0)
    heatmap_gt = batch["heatmaps"].unsqueeze(0)
    keypts_gt = batch["keypoints"].unsqueeze(0).reshape(1, -1, 2)
    heatmap_torch = generate_heatmaps(
        keypts_gt,
        height=im_height,
        width=im_width,
        output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(heatmap_gt, temperature=torch.tensor(100))
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(heatmap_torch, temperature=torch.tensor(100))
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


def test_generate_uniform_heatmaps(cfg, toy_data_dir):

    from lightning_pose.utils.scripts import get_dataset, get_imgaug_transform

    # update config
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.training.uniform_heatmaps_for_nan_keypoints = True

    # build dataset with these new image dimensions
    imgaug_transform = get_imgaug_transform(cfg_tmp)
    heatmap_dataset = get_dataset(
        cfg_tmp,
        data_dir=toy_data_dir,
        imgaug_transform=imgaug_transform,
    )

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width

    batch = heatmap_dataset.__getitem__(idx=0)
    heatmap_gt = batch["heatmaps"].unsqueeze(0)
    keypts_gt = batch["keypoints"].unsqueeze(0).reshape(1, -1, 2)

    heatmap_uniform_torch = generate_heatmaps(
        keypts_gt,
        height=im_height,
        width=im_width,
        output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
        uniform_heatmaps=True,
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(heatmap_gt, temperature=torch.tensor(100))
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(
        heatmap_uniform_torch, temperature=torch.tensor(100)
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

    from lightning_pose.utils.scripts import get_dataset, get_imgaug_transform

    img_shape = (384, 256)

    # update config
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.data.image_resize_dims.height = img_shape[0]
    cfg_tmp.data.image_resize_dims.width = img_shape[1]

    # build dataset with these new image dimensions
    imgaug_transform = get_imgaug_transform(cfg_tmp)
    dataset = get_dataset(
        cfg_tmp,
        data_dir=toy_data_dir,
        imgaug_transform=imgaug_transform,
    )

    # now same test as `test_generate_heatmaps`
    batch = dataset.__getitem__(idx=0)
    heatmap_gt = batch["heatmaps"].unsqueeze(0)
    keypts_gt = batch["keypoints"].unsqueeze(0).reshape(1, -1, 2)
    heatmap_torch = generate_heatmaps(
        keypts_gt,
        height=img_shape[0],
        width=img_shape[1],
        output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
    )

    # find soft argmax and confidence of ground truth heatmap
    softmaxes_gt = spatial_softmax2d(heatmap_gt, temperature=torch.tensor(100))
    preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
    confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

    # find soft argmax and confidence of generated heatmap
    softmaxes_torch = spatial_softmax2d(heatmap_torch, temperature=torch.tensor(100))
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


def test_evaluate_heatmaps_at_location():

    from lightning_pose.data.utils import evaluate_heatmaps_at_location

    height = 24
    width = 12

    # make sure this works when we have a single frame and/or keypoint
    for n_batch in [1, 5]:
        for n_keypoints in [1, 6]:

            heatmaps = torch.zeros((n_batch, n_keypoints, height, width))

            h_locs = torch.randint(0, height, (n_batch, n_keypoints))
            w_locs = torch.randint(0, width, (n_batch, n_keypoints))
            locs = torch.stack([w_locs, h_locs], dim=2)  # x then y
            # set heatmaps values to .2 at 5 locations near the central pixel.
            for i, l1 in enumerate(locs):
                for j, l2 in enumerate(l1):
                    l2_1_offset, l2_0_offset = l2[1] + 1, l2[0] + 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1] - 1, l2[0] - 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1], l2[0]
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1] + 1, l2[0] - 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1] - 1, l2[0] + 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2
            # heatmap values should sum to 1 even when values are spread across the heatmap
            vals = evaluate_heatmaps_at_location(heatmaps=heatmaps, locs=locs)
            assert torch.all(vals == 1.0)

    # more tests

    batch = 1
    num_keypoints = 1
    heat_height = 32
    heat_width = 32

    # ----------------------------------
    # make delta heatmap
    # ----------------------------------
    idx0 = 5
    heatmaps = torch.zeros((batch, num_keypoints, heat_height, heat_width))
    heatmaps[0, 0, idx0, idx0] = 1

    # if we choose the correct location, do we get 1?
    locs0 = torch.zeros((batch, num_keypoints, 2))
    locs0[0, 0, 0] = idx0
    locs0[0, 0, 1] = idx0
    confs0 = evaluate_heatmaps_at_location(heatmaps, locs0)
    assert confs0.shape == (batch, num_keypoints)
    assert torch.allclose(confs0[0], torch.tensor(1.0))

    # if we choose almost the correct location, do we get 1?
    idx1 = idx0 + 1
    locs1 = torch.zeros((batch, num_keypoints, 2))
    locs1[0, 0, 0] = idx1
    locs1[0, 0, 1] = idx1
    confs1 = evaluate_heatmaps_at_location(heatmaps, locs1)
    assert torch.allclose(confs1[0], torch.tensor(1.0))

    # if we choose a completely wrong location, do we get 0?
    idx2 = 25
    locs2 = torch.zeros((batch, num_keypoints, 2))
    locs2[0, 0, 0] = idx2
    locs2[0, 0, 1] = idx2
    confs2 = evaluate_heatmaps_at_location(heatmaps, locs2)
    assert torch.allclose(confs2[0], torch.tensor(0.0))

    # ----------------------------------
    # make a gaussain heatmap
    # ----------------------------------
    heatmaps_g = generate_heatmaps(
        locs0,
        height=heat_height,
        width=heat_width,
        output_shape=(heat_height, heat_width),
    )

    # if we choose the correct location, do we get close to 1?
    confs0_g = evaluate_heatmaps_at_location(heatmaps_g, locs0)
    assert confs0_g[0] > 0
    assert confs0_g[0] <= 1.0

    # if we choose almost the correct location, do we get less than the correct location?
    confs1_g = evaluate_heatmaps_at_location(heatmaps_g, locs1)
    assert confs0_g[0] > confs1_g[0]

    # if we choose a completely wrong location, do we get 0?
    confs2_g = evaluate_heatmaps_at_location(heatmaps_g, locs2)
    assert torch.allclose(confs2_g[0], torch.tensor(0.0))


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
    for v, view in enumerate(range(n_views)):
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
    for v, view in enumerate(range(n_views)):
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


def test_normalized_to_bbox():

    from lightning_pose.data.utils import normalized_to_bbox

    # test when keypoints and bboxes are same size
    keypoints = torch.tensor([
        [[0.0, 0.0]],  # xy for 1 keypoint
        [[1.0, 1.0]],
        [[0.5, 0.5]],
    ])

    bboxes = [
        torch.tensor([0, 0, 100, 200]),  # xyhw
        torch.tensor([20, 30, 100, 200]),
    ]
    for bbox in bboxes:
        kps = normalized_to_bbox(keypoints.clone(), bbox.unsqueeze(0).repeat([3, 1]))
        # (0.0, 0.0) should map to top left corner
        assert kps[0, 0, 0] == bbox[0]
        assert kps[0, 0, 1] == bbox[1]
        # (1.0, 1.0) should map to bottom right corner
        assert kps[1, 0, 0] == bbox[3] + bbox[0]
        assert kps[1, 0, 1] == bbox[2] + bbox[1]
        # (0.5, 0.5) should map to top left corner plus half the new height/width
        assert kps[2, 0, 0] == bbox[3] / 2 + bbox[0]
        assert kps[2, 0, 1] == bbox[2] / 2 + bbox[1]

    # test when keypoints come from context model and bboxes have extra entries for edges
    for bbox in bboxes:
        kps = normalized_to_bbox(keypoints.clone(), bbox.unsqueeze(0).repeat([7, 1]))
        # (0.0, 0.0) should map to top left corner
        assert kps[0, 0, 0] == bbox[0]
        assert kps[0, 0, 1] == bbox[1]
        # (1.0, 1.0) should map to bottom right corner
        assert kps[1, 0, 0] == bbox[3] + bbox[0]
        assert kps[1, 0, 1] == bbox[2] + bbox[1]
        # (0.5, 0.5) should map to top left corner plus half the new height/width
        assert kps[2, 0, 0] == bbox[3] / 2 + bbox[0]
        assert kps[2, 0, 1] == bbox[2] / 2 + bbox[1]


def test_convert_bbox_coords(heatmap_data_module, multiview_heatmap_data_module):

    from lightning_pose.data.utils import convert_bbox_coords

    # -------------------------------------
    # test on single view dataset
    # -------------------------------------
    # params
    x_crop = 25
    y_crop = 40

    # get training batch
    batch_dict = next(iter(heatmap_data_module.train_dataloader()))
    orig_converted = convert_bbox_coords(batch_dict, batch_dict['keypoints'])
    old_image_dims = [batch_dict['images'].size(-2), batch_dict['images'].size(-1)]
    old_bbox = batch_dict["bbox"]
    x_pix = x_crop * old_bbox[:, 3] / old_image_dims[1]
    y_pix = y_crop * old_bbox[:, 2] / old_image_dims[0]

    # create a new batch with smaller & cropped images
    new_dict = batch_dict
    new_dict['images'] = new_dict['images'][:, :, y_crop:-y_crop, x_crop:-x_crop]
    new_dict['bbox'][:, 0] = new_dict['bbox'][:, 0] + x_pix
    new_dict['bbox'][:, 1] = new_dict['bbox'][:, 1] + y_pix
    new_dict['bbox'][:, 2] = new_dict['bbox'][:, 2] - 2 * y_pix
    new_dict['bbox'][:, 3] = new_dict['bbox'][:, 3] - 2 * x_pix
    new_dict['keypoints'][:, 0::2] += x_crop  # keypoints x,y shifted in image
    new_dict['keypoints'][:, 1::2] += y_crop
    new_converted = convert_bbox_coords(new_dict, new_dict['keypoints'])

    # orig and new converted coordinates should be the same
    assert torch.allclose(orig_converted, new_converted, equal_nan=True)

    # -------------------------------------
    # test on dummy multi view dataset
    # -------------------------------------
    batch_dict = {
        "images": torch.tensor(np.random.randn(2, 2, 3, 10, 10)),  # batch, views, RGB, h, w
        "predicted_keypoints": torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # xy, xy (2 keypoints
            [10.0, 10.0, 10.0, 10.0],
        ]),
        "bbox": torch.tensor([
            [5.0, 6.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # xyhw x 2
            [0.0, 0.0, 123.0, 124.0, 0.0, 0.0, 3.0, 4.0],
        ]),
        "num_views": torch.tensor([2, 2]),
    }
    converted = convert_bbox_coords(batch_dict, batch_dict["predicted_keypoints"])
    assert converted[0, 0] == batch_dict["bbox"][0, 0]
    assert converted[0, 1] == batch_dict["bbox"][0, 1]
    assert converted[0, 2] == batch_dict["bbox"][0, 4]
    assert converted[0, 3] == batch_dict["bbox"][0, 5]
    assert converted[1, 0] == batch_dict["bbox"][1, 3]
    assert converted[1, 1] == batch_dict["bbox"][1, 2]
    assert converted[1, 2] == batch_dict["bbox"][1, 7]
    assert converted[1, 3] == batch_dict["bbox"][1, 6]

    # -------------------------------------
    # test on dummy multi view context dataset
    # -------------------------------------
    batch_dict = {
        "images": torch.tensor(np.random.randn(2, 2, 3, 10, 10)),  # batch, views, RGB, h, w
        "predicted_keypoints": torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # xy, xy (2 keypoints)
            [10.0, 10.0, 10.0, 10.0],
        ]),
        "bbox": torch.tensor([
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
            [5.0, 6.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # xyhw x 2
            [0.0, 0.0, 123.0, 124.0, 0.0, 0.0, 3.0, 4.0],
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
            [1.0, 2.0, 100.0, 101.0, 10.0, 11.0, 102.0, 103.0],  # context, will be removed
        ]),
        "num_views": torch.tensor([2, 2, 2, 2, 2, 2]),
    }
    converted = convert_bbox_coords(batch_dict, batch_dict["predicted_keypoints"])
    assert converted[0, 0] == batch_dict["bbox"][2, 0]
    assert converted[0, 1] == batch_dict["bbox"][2, 1]
    assert converted[0, 2] == batch_dict["bbox"][2, 4]
    assert converted[0, 3] == batch_dict["bbox"][2, 5]
    assert converted[1, 0] == batch_dict["bbox"][3, 3]
    assert converted[1, 1] == batch_dict["bbox"][3, 2]
    assert converted[1, 2] == batch_dict["bbox"][3, 7]
    assert converted[1, 3] == batch_dict["bbox"][3, 6]

    # -------------------------------------
    # test error on multi view dataset
    # -------------------------------------
    # get training batch
    batch_dict = next(iter(multiview_heatmap_data_module.train_dataloader()))
    # change number of views for one batch element
    batch_dict["num_views"][0] = 16
    # make sure code complains when batch elements have different numbers of views
    with pytest.raises(ValueError):
        convert_bbox_coords(batch_dict, batch_dict['keypoints'])


class TestConvertOriginalToModelCoords:

    def test_convert_original_to_model_coords_basic(self):
        """Test convert_original_to_model_coords with multiview setup."""

        # Create mock batch_dict
        batch_dict = {
            "num_views": torch.tensor([2, 2]),  # 2 views per batch element
            "images": torch.zeros(2, 2, 3, 256, 256),  # (batch, views, channels, height, width)
            "bbox": torch.tensor([
                # Batch element 0: view 0, view 1
                [0., 0., 100., 200., 50., 25., 100., 200.],
                # Batch element 1: view 0, view 1
                [10., 10., 80., 160., 60., 30., 80., 160.],
            ])
        }

        # Original keypoints: (batch=2, views=2, keypoints=3, xy=2)
        original_keypoints = torch.tensor([
            [  # Batch element 0
                [  # View 0: bbox [0, 0, 100, 200]
                    [0., 0.],  # top-left
                    [200., 100.],  # bottom-right
                    [100., 50.],  # center
                ],
                [  # View 1: bbox [50, 25, 100, 200]
                    [50., 25.],  # top-left
                    [250., 125.],  # bottom-right
                    [150., 75.],  # center
                ]
            ],
            [  # Batch element 1
                [  # View 0: bbox [10, 10, 80, 160]
                    [10., 10.],  # top-left
                    [170., 90.],  # bottom-right
                    [90., 50.],  # center
                ],
                [  # View 1: bbox [60, 30, 80, 160]
                    [60., 30.],  # top-left
                    [220., 110.],  # bottom-right
                    [140., 70.],  # center
                ]
            ]
        ])

        # Convert to model coordinates
        model_keypoints = convert_original_to_model_coords(batch_dict, original_keypoints)

        # Check output shape
        assert model_keypoints.shape == (2, 2, 3, 2)

        # Check that all corner points map correctly
        # Top-left corners should be (0, 0)
        assert torch.allclose(model_keypoints[:, :, 0, :], torch.zeros(2, 2, 2), atol=1e-6)

        # Bottom-right corners should be (256, 256)
        assert torch.allclose(model_keypoints[:, :, 1, :], torch.full((2, 2, 2), 256.0), atol=1e-6)

        # Centers should be (128, 128)
        assert torch.allclose(model_keypoints[:, :, 2, :], torch.full((2, 2, 2), 128.0), atol=1e-6)

    def test_convert_original_to_model_coords_different_views(self):
        """Test with different number of views and keypoints."""

        # Create batch_dict with 3 views
        batch_dict = {
            "num_views": torch.tensor([3, 3]),
            "images": torch.zeros(2, 3, 3, 128, 128),  # 128x128 model input
            "bbox": torch.tensor([
                # Batch 0: 3 views with different bboxes
                [0., 0., 50., 100., 25., 25., 50., 100., 50., 50., 50., 100.],
                # Batch 1: 3 views
                [10., 10., 60., 120., 30., 30., 60., 120., 60., 60., 60., 120.],
            ])
        }

        # Test with 2 keypoints per view
        original_keypoints = torch.tensor([
            [  # Batch 0
                [[0., 0.], [100., 50.]],  # View 0: corners of bbox [0,0,50,100]
                [[25., 25.], [125., 75.]],  # View 1: corners of bbox [25,25,50,100]
                [[50., 50.], [150., 100.]],  # View 2: corners of bbox [50,50,50,100]
            ],
            [  # Batch 1
                [[10., 10.], [130., 70.]],  # View 0: corners of bbox [10,10,60,120]
                [[30., 30.], [150., 90.]],  # View 1: corners of bbox [30,30,60,120]
                [[60., 60.], [180., 120.]],  # View 2: corners of bbox [60,60,60,120]
            ]
        ])

        model_keypoints = convert_original_to_model_coords(batch_dict, original_keypoints)

        # Check output shape: (batch=2, views=3, keypoints=2, xy=2)
        assert model_keypoints.shape == (2, 3, 2, 2)

        # All top-left corners should map to (0, 0)
        assert torch.allclose(model_keypoints[:, :, 0, :], torch.zeros(2, 3, 2), atol=1e-6)

        # All bottom-right corners should map to (128, 128) since model is 128x128
        assert torch.allclose(model_keypoints[:, :, 1, :], torch.full((2, 3, 2), 128.0), atol=1e-6)


class TestOriginalToModel:

    def test_original_to_model_basic(self):
        """Test original_to_model with basic coordinate transformations."""

        model_width = 256.
        model_height = 256.

        bboxes = [
            torch.tensor([0., 0., 100., 200.]),  # bbox at origin, height=100, width=200
            torch.tensor([50., 25., 100., 200.]),  # bbox offset, same dimensions
        ]

        for bbox in bboxes:

            # Define test keypoints based on the bbox
            x, y, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
            keypoints = torch.tensor([
                [[x.item(), y.item()]],           # top-left corner of bbox
                [[x.item() + w.item(), y.item() + h.item()]], # bottom-right corner of bbox
                [[x.item() + w.item()/2, y.item() + h.item()/2]], # center of bbox
            ])

            kps = original_to_model(
                keypoints.clone(),
                bbox.unsqueeze(0).repeat([3, 1]),
                model_width,
                model_height,
            )

            # Top-left corner of bbox (0,0 in normalized space) should map to (0, 0) in model space
            expected_x = 0.0
            expected_y = 0.0
            assert torch.isclose(kps[0, 0, 0], torch.tensor(expected_x), atol=1e-6)
            assert torch.isclose(kps[0, 0, 1], torch.tensor(expected_y), atol=1e-6)

            # Bottom-right corner of bbox (1,1 in normalized space) should map to
            # (model_width, model_height)
            expected_x = model_width
            expected_y = model_height
            assert torch.isclose(kps[1, 0, 0], torch.tensor(expected_x), atol=1e-6)
            assert torch.isclose(kps[1, 0, 1], torch.tensor(expected_y), atol=1e-6)

            # Center of bbox (0.5, 0.5 in normalized space) should map to
            # (model_width/2, model_height/2)
            expected_x = model_width / 2
            expected_y = model_height / 2
            assert torch.isclose(kps[2, 0, 0], torch.tensor(expected_x), atol=1e-6)
            assert torch.isclose(kps[2, 0, 1], torch.tensor(expected_y), atol=1e-6)

    def test_original_to_model_context_batch(self):
        """Test original_to_model with context batch (extra bbox entries for edges)."""

        model_width = 256.
        model_height = 256.

        # Test different bboxes with context (7 entries, uses middle 3: [2:-2])
        bboxes = [
            torch.tensor([0., 0., 100., 200.]),  # bbox at origin
            torch.tensor([50., 25., 100., 200.]),  # bbox offset
        ]

        for bbox in bboxes:

            # Define test keypoints based on the bbox
            x, y, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
            keypoints = torch.tensor([
                [[x.item(), y.item()]],  # top-left corner of bbox
                [[x.item() + w.item(), y.item() + h.item()]],  # bottom-right corner
                [[x.item() + w.item() / 2, y.item() + h.item() / 2]],  # center of bbox
            ])

            # Create 7-entry bbox tensor for context batch
            bbox_context = bbox.unsqueeze(0).repeat([7, 1])

            kps = original_to_model(
                keypoints.clone(),
                bbox_context,
                model_width,
                model_height,
            )

            # Same assertions as basic test since the function should use bbox[2:-2]
            # which gives us the middle entries (same as the original bbox)

            # Top-left corner
            assert torch.isclose(kps[0, 0, 0], torch.tensor(0.0), atol=1e-6)
            assert torch.isclose(kps[0, 0, 1], torch.tensor(0.0), atol=1e-6)

            # Bottom-right corner
            assert torch.isclose(kps[1, 0, 0], torch.tensor(model_width), atol=1e-6)
            assert torch.isclose(kps[1, 0, 1], torch.tensor(model_height), atol=1e-6)

            # Center
            assert torch.isclose(kps[2, 0, 0], torch.tensor(model_width / 2), atol=1e-6)
            assert torch.isclose(kps[2, 0, 1], torch.tensor(model_height / 2), atol=1e-6)

    def test_original_to_model_different_dimensions(self):
        """Test with non-square model dimensions."""

        keypoints = torch.tensor([
            [[50., 25.]],  # top-left of bbox
            [[150., 75.]],  # bottom-right of bbox
            [[100., 50.]],  # center of bbox
        ])

        bbox = torch.tensor([50., 25., 50., 100.])  # x=50, y=25, h=50, w=100
        model_width = 128.
        model_height = 64.

        kps = original_to_model(
            keypoints,
            bbox.unsqueeze(0).repeat([3, 1]),
            model_width,
            model_height
        )

        # Top-left: (50-50)/100 * 128 = 0, (25-25)/50 * 64 = 0
        assert torch.isclose(kps[0, 0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(kps[0, 0, 1], torch.tensor(0.0), atol=1e-6)

        # Bottom-right: (150-50)/100 * 128 = 128, (75-25)/50 * 64 = 64
        assert torch.isclose(kps[1, 0, 0], torch.tensor(128.0), atol=1e-6)
        assert torch.isclose(kps[1, 0, 1], torch.tensor(64.0), atol=1e-6)

        # Center: (100-50)/100 * 128 = 64, (50-25)/50 * 64 = 32
        assert torch.isclose(kps[2, 0, 0], torch.tensor(64.0), atol=1e-6)
        assert torch.isclose(kps[2, 0, 1], torch.tensor(32.0), atol=1e-6)
