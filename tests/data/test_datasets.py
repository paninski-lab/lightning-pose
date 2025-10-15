"""Test basic dataset functionality."""

import copy

import numpy as np
import pytest
import torch

from lightning_pose.data.datasets import MultiviewHeatmapDataset


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


class TestHeatmapDataset:

    def test_heatmap_dataset(self, cfg, heatmap_dataset):

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

    def test_heatmap_dataset_context(self, cfg, heatmap_dataset_context):
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


class TestMultiviewHeatmapDataset:

    @pytest.fixture
    def sample_array_3d(self):
        # test array (median value must be zero for below tests to work)
        keypoints_3d = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
            [np.nan, np.nan, np.nan],
        ])
        return keypoints_3d

    def test_dataset_properties(self, cfg_multiview, multiview_heatmap_dataset):

        im_height = cfg_multiview.data.image_resize_dims.height
        im_width = cfg_multiview.data.image_resize_dims.width
        num_views = len(cfg_multiview.data.view_names)

        assert multiview_heatmap_dataset.height == im_height
        assert multiview_heatmap_dataset.width == im_width
        assert multiview_heatmap_dataset.num_views == num_views

    def test_batch_properties(self, cfg_multiview, multiview_heatmap_dataset):

        im_height = cfg_multiview.data.image_resize_dims.height
        im_width = cfg_multiview.data.image_resize_dims.width
        num_targets = multiview_heatmap_dataset.num_targets
        num_views = multiview_heatmap_dataset.num_views

        batch = multiview_heatmap_dataset[0]

        assert type(batch["images"]) is torch.Tensor
        assert type(batch["keypoints"]) is torch.Tensor

        assert batch["images"].shape == (num_views, 3, im_height, im_width)
        assert batch["keypoints"].shape == (num_targets,)
        assert batch["heatmaps"].shape[1:] == multiview_heatmap_dataset.output_shape
        assert batch["bbox"].shape == (num_views * 4,)  # xyhw for each view

        # no camera params
        assert batch["keypoints_3d"].shape == (1,)
        assert batch["intrinsic_matrix"].shape == (1, 3, 3)
        assert batch["extrinsic_matrix"].shape == (1, 3, 4)
        assert batch["distortions"].shape == (1, 5)

    def test_dataset_properties_context(self, cfg_multiview, multiview_heatmap_dataset_context):

        im_height = cfg_multiview.data.image_resize_dims.height
        im_width = cfg_multiview.data.image_resize_dims.width
        num_views = len(cfg_multiview.data.view_names)

        assert multiview_heatmap_dataset_context.height == im_height
        assert multiview_heatmap_dataset_context.width == im_width
        assert multiview_heatmap_dataset_context.num_views == num_views

    def test_batch_properties_context(self, cfg_multiview, multiview_heatmap_dataset_context):

        im_height = cfg_multiview.data.image_resize_dims.height
        im_width = cfg_multiview.data.image_resize_dims.width
        num_views = len(cfg_multiview.data.view_names)
        num_targets = multiview_heatmap_dataset_context.num_targets

        batch = multiview_heatmap_dataset_context[0]

        assert type(batch["images"]) is torch.Tensor
        assert type(batch["keypoints"]) is torch.Tensor

        assert batch["images"].shape == (num_views, 5, 3, im_height, im_width)
        assert batch["keypoints"].shape == (num_targets,)
        assert batch["heatmaps"].shape[1:] == multiview_heatmap_dataset_context.output_shape
        assert batch["bbox"].shape == (num_views * 4,)  # xyhw for each view

        # no camera params
        assert batch["keypoints_3d"].shape == (1,)
        assert batch["intrinsic_matrix"].shape == (1, 3, 3)
        assert batch["extrinsic_matrix"].shape == (1, 3, 4)
        assert batch["distortions"].shape == (1, 5)

    def test_scale_keypoints(self, sample_array_3d):

        factor = 0.5
        kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
            keypoints_3d=sample_array_3d,
            scale_params=(factor, factor),  # force to scale by 0.5
            shift_param=0.0,  # no shift
        )
        assert np.allclose(kp_aug, sample_array_3d * factor, equal_nan=True)

        factor = 2.0
        kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
            keypoints_3d=sample_array_3d,
            scale_params=(factor, factor),  # force to scale by 0.5
            shift_param=0.0,  # no shift
        )
        assert np.allclose(kp_aug, sample_array_3d * factor, equal_nan=True)

    def test_translate_keypoints(self, sample_array_3d):

        kp_aug = MultiviewHeatmapDataset._scale_translate_keypoints(
            keypoints_3d=sample_array_3d,
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
            keypoints_3d=sample_array_3d + median,
            scale_params=(factor, factor),
            shift_param=0.0,
        )
        assert np.allclose(kp_aug, sample_array_3d * factor + median, equal_nan=True)

    def test_resize_keypoints(self, multiview_heatmap_dataset):

        # extract required info from dataset
        im_height = multiview_heatmap_dataset.height
        im_width = multiview_heatmap_dataset.width
        num_views = multiview_heatmap_dataset.num_views

        # test keypoints
        keypoints = np.array([
            [[100, 150], [200, 250]],  # view 0: 2 keypoints
            [[50, 75], [150, 175]],  # view 1: 2 keypoints
        ])

        # test bboxes format: [x_offset, y_offset, height, width]
        bboxes = [
            torch.tensor([12, 34, 100, 200]),  # view 0
            torch.tensor([56, 78, 300, 400])  # view 1
        ]

        result = multiview_heatmap_dataset._resize_keypoints(keypoints, bboxes)

        # check output structure
        assert len(result) == num_views
        for idx_view in range(num_views):
            assert result[idx_view].shape == (4,)

        # check exact values
        for idx_view in range(num_views):
            keypoints_ = keypoints[idx_view].copy()  # shape (num_keypoints, 2)
            bbox_ = bboxes[idx_view].cpu().numpy()
            keypoints_[:, 0] = ((keypoints_[:, 0] - bbox_[0]) / bbox_[3]) * im_width
            keypoints_[:, 1] = ((keypoints_[:, 1] - bbox_[1]) / bbox_[2]) * im_height
            assert np.array_equal(result[idx_view], keypoints_.reshape(-1))

    def test_resize_images(self, multiview_heatmap_dataset):

        # extract required info from dataset
        im_height = multiview_heatmap_dataset.height
        im_width = multiview_heatmap_dataset.width
        num_views = multiview_heatmap_dataset.num_views

        # create test images with different sizes
        images = [
            torch.randn(3, 384, 296),  # view 0
            torch.randn(3, 512, 488)  # view 1
        ]

        result = multiview_heatmap_dataset._resize_images(images)

        # check output structure
        assert len(result) == num_views

        for idx_view in range(num_views):
            # check that images are resized to target dimensions
            assert result[idx_view].shape == (3, im_height, im_width)
            # check that output is a different tensor (cloned)
            assert result[idx_view] is not images[idx_view]

    def test_get_2d_keypoints_from_example_dict_absolute_coords(self, multiview_heatmap_dataset):

        from lightning_pose.data.utils import normalized_to_bbox

        # test data
        data_dict = {
            "view_0": {
                "keypoints": torch.tensor([64, 64, 96, 96, 16, 16, 256, 256], dtype=torch.float32),
                "images": torch.randn(3, 256, 256),  # shape: (C, H, W)
                "bbox": torch.tensor([0, 0, 512, 512], dtype=torch.float32),  # [x, y, h, w]
            },
            "view_1": {
                "keypoints": torch.tensor([32, 32, 64, 64, 96, 96, 128, 128], dtype=torch.float32),
                "images": torch.randn(3, 128, 128),  # shape: (C, H, W)
                "bbox": torch.tensor([10, 10, 300, 300], dtype=torch.float32),  # [x, y, h, w]
            },
        }

        # mock dataset
        dataset = copy.deepcopy(multiview_heatmap_dataset)
        num_keypoints = 8  # 4 for each view
        dataset.num_keypoints = num_keypoints
        dataset.view_names = list(data_dict.keys())
        num_views = dataset.num_views

        # call function
        result = dataset._get_2d_keypoints_from_example_dict_absolute_coords(data_dict)

        # check output shape
        assert result.shape == (len(data_dict), num_keypoints // num_views, 2)

        # check result type
        assert isinstance(result, np.ndarray)

        # check exact values
        for idx_view, (view, example_dict) in enumerate(data_dict.items()):
            # create a copy to avoid modifying the original data
            keypoints_curr = example_dict["keypoints"].reshape(
                num_keypoints // num_views, 2
            ).clone()
            # transform keypoints from bbox coordinates to absolute frame coordinates
            # 1. divide by image dims to get 0-1 normalized coords
            keypoints_curr[:, 0] /= example_dict["images"].shape[-1]  # -1 dim is "x"
            keypoints_curr[:, 1] /= example_dict["images"].shape[-2]  # -2 dim is "y"
            # 2. multiply and add by bbox dims
            keypoints_curr = normalized_to_bbox(
                keypoints=keypoints_curr.unsqueeze(0),
                bbox=example_dict["bbox"].unsqueeze(0),
            )[0].cpu().numpy()
            assert np.array_equal(keypoints_curr, result[idx_view])

    def test_apply_3d_transforms_all_nans(self, multiview_heatmap_dataset):

        num_keypoints = multiview_heatmap_dataset.num_keypoints
        num_views = multiview_heatmap_dataset.num_views

        datadict = {}
        for view in multiview_heatmap_dataset.view_names:
            datadict[view] = multiview_heatmap_dataset.dataset[view].__getitem__(
                0, ignore_nans=True,
            )
            datadict[view]["keypoints"].fill_(float('nan'))
        datadict_aug, keypoints_3d = multiview_heatmap_dataset.apply_3d_transforms(datadict, None)
        assert torch.all(torch.isnan(keypoints_3d))
        assert keypoints_3d.shape == (num_keypoints // num_views, 3)
        for view in multiview_heatmap_dataset.view_names:
            assert datadict_aug[view]["images"].shape == (3, 256, 256)
            assert datadict_aug[view]["keypoints"].shape == (num_keypoints,)
            assert datadict_aug[view]["bbox"].shape == (4,)


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
