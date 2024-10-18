"""Test data utils functionality."""

import copy

import pytest
import torch
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d

from lightning_pose.data.utils import generate_heatmaps


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
    assert images_tensor.shape == (num_frames, 3, 256, 256)

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
    import cv2

    from lightning_pose.data.utils import count_frames

    # make sure value is correct in the single view case
    num_frames = 0
    for video_file in video_list:
        cap = cv2.VideoCapture(video_file)
        num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    num_frames_1 = count_frames(video_list)
    assert num_frames == num_frames_1

    # with multiview we have a list of lists, make sure we only count frames from one view
    video_list_2 = [
        [video_list[0]],  # view 0
        [video_list[0]],  # view 1
        [video_list[0]],  # view 2
    ]
    num_frames_2 = count_frames(video_list_2)
    assert num_frames == num_frames_2


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
