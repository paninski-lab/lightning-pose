"""Test basic dataset functionality."""

import numpy as np
import torch


def test_base_dataset(cfg, base_dataset):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    num_targets = base_dataset.num_targets

    # check stored object properties
    assert base_dataset.height == im_height
    assert base_dataset.width == im_width

    # check batch properties
    batch = base_dataset[0]
    assert batch["images"].shape == (3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert isinstance(batch["images"], torch.Tensor)
    assert isinstance(batch["keypoints"], torch.Tensor)


def test_heatmap_dataset(cfg, heatmap_dataset):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    num_targets = heatmap_dataset.num_targets

    # check stored object properties
    assert heatmap_dataset.height == im_height
    assert heatmap_dataset.width == im_width

    # check batch properties
    batch = heatmap_dataset[0]
    assert batch["images"].shape == (3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == heatmap_dataset.output_shape
    assert isinstance(batch["images"], torch.Tensor)
    assert isinstance(batch["keypoints"], torch.Tensor)


def test_multiview_heatmap_dataset(cfg_multiview, multiview_heatmap_dataset):

    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    num_targets = multiview_heatmap_dataset.num_targets

    # check stored object properties
    assert multiview_heatmap_dataset.height == im_height
    assert multiview_heatmap_dataset.width == im_width

    # check batch properties
    batch = multiview_heatmap_dataset[0]
    assert batch["images"].shape == (
        len(cfg_multiview.data.csv_file),
        3,
        im_height,
        im_width,
    )
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == multiview_heatmap_dataset.output_shape
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_heatmap_dataset_context(cfg, heatmap_dataset_context):

    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width
    num_targets = heatmap_dataset_context.num_targets

    # check stored object properties
    assert heatmap_dataset_context.height == im_height
    assert heatmap_dataset_context.width == im_width

    # check batch properties
    batch = heatmap_dataset_context[0]
    assert batch["images"].shape == (5, 3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == heatmap_dataset_context.output_shape
    assert isinstance(batch["images"], torch.Tensor)
    assert isinstance(batch["keypoints"], torch.Tensor)


def test_multiview_heatmap_dataset_context(cfg_multiview, multiview_heatmap_dataset_context):
    im_height = cfg_multiview.data.image_resize_dims.height
    im_width = cfg_multiview.data.image_resize_dims.width
    num_targets = multiview_heatmap_dataset_context.num_targets

    # check stored object properties
    assert multiview_heatmap_dataset_context.height == im_height
    assert multiview_heatmap_dataset_context.width == im_width

    # check batch properties
    batch = multiview_heatmap_dataset_context[0]
    assert batch["images"].shape == (2, 5, 3, im_height, im_width)
    assert batch["keypoints"].shape == (num_targets,)
    assert batch["heatmaps"].shape[1:] == multiview_heatmap_dataset_context.output_shape
    assert type(batch["images"]) is torch.Tensor
    assert type(batch["keypoints"]) is torch.Tensor


def test_equal_return_sizes(base_dataset, heatmap_dataset):
    # can only assert the batches are the same if not using imgaug pipeline
    assert base_dataset[0]["images"].shape == heatmap_dataset[0]["images"].shape


def test_multiview_scale_translate_keypoints():

    from lightning_pose.data.datasets import MultiviewHeatmapDataset

    # test array (median value must be zero for below tests to work)
    keypoints_3d = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
        [np.nan, np.nan, np.nan],
    ])

    # test scale
    factor = 0.5
    kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
        keypoints_3d=keypoints_3d,
        scale_params=(factor, factor),  # force to scale by 0.5
        shift_param=0.0,  # no shift
    )
    assert np.allclose(kp_aug, keypoints_3d * factor, equal_nan=True)

    factor = 2.0
    kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
        keypoints_3d=keypoints_3d,
        scale_params=(factor, factor),  # force to scale by 0.5
        shift_param=0.0,  # no shift
    )
    assert np.allclose(kp_aug, keypoints_3d * factor, equal_nan=True)

    # test translation
    kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
        keypoints_3d=keypoints_3d,
        scale_params=(1.0, 1.0),  # no scale
        shift_param=1.0,  # random shift
    )
    # make sure all keypoints are translated the same in each dim
    for d in range(3):
        assert kp_aug[:-1, d].all()

    # make sure median centering works
    median = 5
    factor = 1.5
    kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
        keypoints_3d=keypoints_3d + median,
        scale_params=(factor, factor),
        shift_param=0.0,
    )
    assert np.allclose(kp_aug, keypoints_3d * factor + median, equal_nan=True)


def test_multiview_rotate_cameras():

    from aniposelib.cameras import Camera
    from scipy.spatial.transform import Rotation

    from lightning_pose.data.cameras import CameraGroup
    from lightning_pose.data.datasets import MultiviewHeatmapDataset

    # hard-coded example from fly-anipose
    intrinsics = np.array(
        [[[1.4633e+04, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.4633e+04, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.6343e+04, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.6343e+04, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]]],
    )
    extrinsics = np.array(
        [[[7.9065e-01, -1.3940e-01, 5.9619e-01, -1.4132e+00],
          [-2.8695e-02, 9.6423e-01, 2.6351e-01, -1.0720e+00],
          [-6.1160e-01, -2.2545e-01, 7.5837e-01, 4.7490e+01]],

         [[9.6419e-01, 1.3962e-01, 2.2546e-01, 1.2773e-01],
          [-1.2986e-01, 9.8986e-01, -5.7627e-02, -5.1388e-01],
          [-2.3122e-01, 2.6284e-02, 9.7255e-01, 7.0362e+01]]],
    )
    distortions = np.array(
        [[-21.4957, 0.0000, 0.0000, 0.0000, 0.0000],
         [-14.0726, 0.0000, 0.0000, 0.0000, 0.0000]],
    )

    pi_over_two_rot_vecs = np.array(
        [[-7.295e-01, 3.090e-01, 1.558e+00],
         [np.nan, np.nan, np.nan]],
    )
    cameras = [
        Camera(
            matrix=intrinsics[c],
            dist=distortions[c],
            rvec=Rotation.from_matrix(extrinsics[c][:3, :3]).as_rotvec(),
            tvec=extrinsics[c][:3, 3],
        )
        for c in range(2)
    ]

    # --------------------------
    # test _rotate_camera
    # --------------------------

    # no rotation, no params should change
    camera_rotated = MultiviewHeatmapDataset._rotate_camera(cameras[0], angle_rad=0)
    assert np.allclose(cameras[0].get_params(), camera_rotated.get_params())

    # rotate by pi/2
    camera_rotated = MultiviewHeatmapDataset._rotate_camera(cameras[0], angle_rad=np.pi / 2)
    assert not np.allclose(cameras[0].get_rotation(), camera_rotated.get_rotation())
    assert np.allclose(camera_rotated.get_rotation(), pi_over_two_rot_vecs[0], rtol=1e-3)
    assert np.allclose(cameras[0].get_translation(), camera_rotated.get_translation())
    assert np.allclose(cameras[0].get_focal_length(), camera_rotated.get_focal_length())
    assert np.allclose(cameras[0].get_distortions(), camera_rotated.get_distortions())

    # --------------------------
    # test _rotate_cameras
    # --------------------------
    camgroup = CameraGroup(cameras)

    # no rotation, params should not change
    camgroup_rotated = MultiviewHeatmapDataset._rotate_cameras(camgroup, rotation_max_angle=0)
    for cam, cam_rot in zip(cameras, camgroup_rotated.cameras):
        assert np.allclose(cam.get_params(), cam_rot.get_params())

    # rotate by pi max
    camgroup_rotated = MultiviewHeatmapDataset._rotate_cameras(camgroup, rotation_max_angle=np.pi)
    for cam, cam_rot in zip(cameras, camgroup_rotated.cameras):
        assert not np.allclose(cam.get_rotation(), cam_rot.get_rotation())
        assert np.allclose(cam.get_translation(), cam_rot.get_translation())
        assert np.allclose(cam.get_focal_length(), cam_rot.get_focal_length())
        assert np.allclose(cam.get_distortions(), cam_rot.get_distortions())
