"""Test basic dataset functionality."""

import copy
import logging
import os
from unittest.mock import patch

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import pytest
import torch
from aniposelib.cameras import Camera
from PIL import Image

from lightning_pose.data.bboxes import norm_to_frame
from lightning_pose.data.cameras import CameraGroup
from lightning_pose.data.datamodules import BaseDataModule
from lightning_pose.data.datasets import (
    BaseTrackingDataset,
    HeatmapDataset,
    MultiviewHeatmapDataset,
)


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

    @pytest.fixture
    def sample_images(self):
        """Create sample torch images for testing."""
        device = torch.device("cpu")
        # Create 3 sample images with different patterns for easier verification
        img1 = torch.ones((3, 100, 100), device=device) * 0.5  # gray image
        img2 = torch.zeros((3, 120, 80), device=device)  # black image
        img3 = torch.ones((3, 90, 110), device=device)  # white image
        return [img1, img2, img3]

    @pytest.fixture
    def sample_keypoints_simple(self):
        """Create simple keypoint pairs for testing basic transformation."""
        # Original keypoints in normalized coordinates (3 views, 4 keypoints each)
        keypoints_orig = np.array([
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],  # square
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],  # larger square
            [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],  # smaller square
        ])

        # Augmented keypoints (slightly scaled and translated)
        keypoints_aug = np.array([
            [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],  # smaller, centered
            [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]],  # slightly smaller
            [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]],  # smaller
        ])

        return keypoints_orig, keypoints_aug

    @pytest.fixture
    def sample_keypoints_with_nan(self):
        """Create keypoints with some NaN values to test masking."""
        keypoints_orig = np.array([
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [np.nan, np.nan]],
            [[0.1, 0.1], [0.9, 0.1], [np.nan, np.nan], [0.1, 0.9]],
            [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
        ])

        keypoints_aug = np.array([
            [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [np.nan, np.nan]],
            [[0.15, 0.15], [0.85, 0.15], [np.nan, np.nan], [0.15, 0.85]],
            [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]],
        ])

        return keypoints_orig, keypoints_aug

    @pytest.fixture
    def sample_bboxes(self):
        """Create sample bounding boxes in format [x, y, h, w]."""
        return [
            np.array([0.0, 0.0, 1.0, 1.0]),  # full image
            np.array([0.0, 0.0, 1.0, 1.0]),  # full image
            np.array([0.0, 0.0, 1.0, 1.0]),  # full image
        ]

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

    def test_transform_images_basic(self, sample_images, sample_keypoints_simple, sample_bboxes):
        """Test _transform_images basic transformation functionality."""
        keypoints_orig, keypoints_aug = sample_keypoints_simple

        # Mock cv2.estimateAffinePartial2D to return a predictable transformation matrix
        with patch("cv2.estimateAffinePartial2D") as mock_estimate:
            # Return identity-like transformation
            mock_estimate.return_value = (
                np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]], dtype=np.float32), None
            )

            # Mock the kornia transform function
            with patch("kornia.geometry.transform.warp_affine") as mock_warp:
                mock_warp.side_effect = lambda img, M, **kwargs: img  # return unchanged image

                result = MultiviewHeatmapDataset._transform_images(
                    images=sample_images,
                    keypoints_orig=keypoints_orig,
                    keypoints_aug=keypoints_aug,
                    bboxes=sample_bboxes,
                )

                # Check that we get the right number of transformed images
                assert len(result) == len(sample_images)

                # Check that each result is a tensor
                for transformed_img in result:
                    assert isinstance(transformed_img, torch.Tensor)

                # Verify cv2.estimateAffinePartial2D was called for each view
                assert mock_estimate.call_count == len(sample_images)

                # Verify warp_affine was called for each view
                assert mock_warp.call_count == len(sample_images)

    def test_transform_images_nan_keypoint_handling(
        self, sample_images, sample_keypoints_with_nan, sample_bboxes,
    ):
        """Test that NaN keypoints are properly filtered out."""
        keypoints_orig, keypoints_aug = sample_keypoints_with_nan

        with patch("cv2.estimateAffinePartial2D") as mock_estimate:
            mock_estimate.return_value = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), None)

            with patch("kornia.geometry.transform.warp_affine") as mock_warp:
                mock_warp.side_effect = lambda img, M, **kwargs: img

                # This should work despite NaN values
                result = MultiviewHeatmapDataset._transform_images(
                    images=sample_images,
                    keypoints_orig=keypoints_orig,
                    keypoints_aug=keypoints_aug,
                    bboxes=sample_bboxes
                )

                assert len(result) == len(sample_images)

                # Check that the function filtered out NaN keypoints by examining call args
                call_args_list = mock_estimate.call_args_list

                # First view should have 3 valid keypoints (4th is NaN)
                first_call_orig_pts = call_args_list[0][0][0]
                assert len(first_call_orig_pts) == 3

                # Second view should have 3 valid keypoints (3rd is NaN)
                second_call_orig_pts = call_args_list[1][0][0]
                assert len(second_call_orig_pts) == 3

                # Third view should have 4 valid keypoints (no NaN)
                third_call_orig_pts = call_args_list[2][0][0]
                assert len(third_call_orig_pts) == 4

    def test_transform_images_insufficient_keypoints_error(self, sample_images, sample_bboxes):
        """Test that an error is raised when there are fewer than 3 valid keypoints."""

        # create keypoints with insufficient valid points (< 3)
        keypoints_orig = np.array([[[0.2, 0.2], [0.8, 0.2], [np.nan, np.nan], [np.nan, np.nan]]])
        keypoints_aug = np.array([[[0.25, 0.25], [0.7, 0.3], [np.nan, np.nan], [np.nan, np.nan]]])

        # only use first image and corresponding bbox
        images = sample_images[:1]
        bboxes = sample_bboxes[:1]

        with pytest.raises(RuntimeError, match="Fewer than 3 valid keypoints"):
            MultiviewHeatmapDataset._transform_images(
                images=images,
                keypoints_orig=keypoints_orig,
                keypoints_aug=keypoints_aug,
                bboxes=bboxes,
            )

    def test_transform_images_coordinate_transformation(
        self, sample_images, sample_keypoints_simple, sample_bboxes,
    ):
        """Test that keypoint coordinates are correctly transformed from bbox to image space."""
        keypoints_orig, keypoints_aug = sample_keypoints_simple

        # Use a bbox that's not the full image to test coordinate transformation
        custom_bboxes = [
            np.array([0.1, 0.2, 0.6, 0.5]),  # x=0.1, y=0.2, h=0.6, w=0.5
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
        ]

        with patch("cv2.estimateAffinePartial2D") as mock_estimate:
            mock_estimate.return_value = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), None)

            with patch("kornia.geometry.transform.warp_affine") as mock_warp:
                mock_warp.side_effect = lambda img, M, **kwargs: img

                MultiviewHeatmapDataset._transform_images(
                    images=sample_images,
                    keypoints_orig=keypoints_orig,
                    keypoints_aug=keypoints_aug,
                    bboxes=custom_bboxes,
                )

                # Check the transformed coordinates passed to cv2.estimateAffinePartial2D
                first_call_args = mock_estimate.call_args_list[0]
                orig_pts_transformed = first_call_args[0][0]

                # Original keypoint [0.2, 0.2] in bbox [0.1, 0.2, 0.6, 0.5]
                # should transform to image coordinates
                # x: (0.2 - 0.1) / 0.5 * 100 = 20.0
                # y: (0.2 - 0.2) / 0.6 * 100 = 0.0
                expected_x = (0.2 - 0.1) / 0.5 * 100  # image width = 100
                expected_y = (0.2 - 0.2) / 0.6 * 100  # image height = 100

                # Check first transformed point
                assert np.isclose(orig_pts_transformed[0, 0], expected_x)
                assert np.isclose(orig_pts_transformed[0, 1], expected_y)

    @pytest.mark.parametrize("img_height,img_width", [(64, 64), (128, 256), (480, 640)])
    def test_transform_images_different_image_sizes(self, img_height, img_width):
        """Test the function with different image dimensions."""
        device = torch.device("cpu")
        images = [torch.randn((3, img_height, img_width), device=device)]

        keypoints_orig = np.array([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]])
        keypoints_aug = np.array([[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]])
        bboxes = [np.array([0.0, 0.0, 1.0, 1.0])]

        with patch("cv2.estimateAffinePartial2D") as mock_estimate:
            mock_estimate.return_value = (
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), None)

            with patch("kornia.geometry.transform.warp_affine") as mock_warp:
                mock_warp.side_effect = lambda img, M, dsize, **kwargs: img

                result = MultiviewHeatmapDataset._transform_images(
                    images=images,
                    keypoints_orig=keypoints_orig,
                    keypoints_aug=keypoints_aug,
                    bboxes=bboxes
                )

                assert len(result) == 1

                # Check that dsize parameter matches the original image dimensions
                call_args = mock_warp.call_args_list[0]
                kwargs = call_args[1]
                assert kwargs["dsize"] == (img_height, img_width)

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
        for idx_view, (_view, example_dict) in enumerate(data_dict.items()):
            # create a copy to avoid modifying the original data
            keypoints_curr = example_dict["keypoints"].reshape(
                num_keypoints // num_views, 2
            ).clone()
            # transform keypoints from bbox coordinates to absolute frame coordinates
            # 1. divide by image dims to get 0-1 normalized coords
            keypoints_curr[:, 0] /= example_dict["images"].shape[-1]  # -1 dim is "x"
            keypoints_curr[:, 1] /= example_dict["images"].shape[-2]  # -2 dim is "y"
            # 2. multiply and add by bbox dims
            keypoints_curr = norm_to_frame(
                keypoints=keypoints_curr.unsqueeze(0),
                bbox=example_dict["bbox"].unsqueeze(0),
            )[0].cpu().numpy()
            assert np.array_equal(keypoints_curr, result[idx_view])


class TestApply3DTransforms:
    """Tests for MultiviewHeatmapDataset.apply_3d_transforms method."""

    @pytest.fixture
    def camera_group(self):
        """Create a mock CameraGroup object."""

        # ------------------------------------------
        # hard-coded values from anipose-fly example
        # (using aniposelib CameraGroup object)
        # ------------------------------------------

        intrinsics = torch.tensor(
            [[[1.4633e+04, 0.0000e+00, 4.1600e+02],
              [0.0000e+00, 1.4633e+04, 3.1600e+02],
              [0.0000e+00, 0.0000e+00, 1.0000e+00]],

             [[1.6343e+04, 0.0000e+00, 4.1600e+02],
              [0.0000e+00, 1.6343e+04, 3.1600e+02],
              [0.0000e+00, 0.0000e+00, 1.0000e+00]]]
        )
        extrinsics = torch.tensor(
            [[[7.9065e-01, -1.3940e-01, 5.9619e-01, -1.4132e+00],
              [-2.8695e-02, 9.6423e-01, 2.6351e-01, -1.0720e+00],
              [-6.1160e-01, -2.2545e-01, 7.5837e-01, 4.7490e+01]],

             [[9.6419e-01, 1.3962e-01, 2.2546e-01, 1.2773e-01],
              [-1.2986e-01, 9.8986e-01, -5.7627e-02, -5.1388e-01],
              [-2.3122e-01, 2.6284e-02, 9.7255e-01, 7.0362e+01]]]
        )
        distortions = torch.tensor(
            [[-21.4957, 0.0000, 0.0000, 0.0000, 0.0000],
             [-14.0726, 0.0000, 0.0000, 0.0000, 0.0000]]
        )

        cameras = []
        for i in range(2):

            # Convert rotation matrix to rotation vector using Rodrigues
            rotation_matrix = extrinsics[i][:3, :3].numpy()
            rvec, _ = cv2.Rodrigues(rotation_matrix)
            rvec = rvec.flatten()  # Make it a 1D vector

            camera = Camera(
                matrix=intrinsics[i].numpy(),
                rvec=rvec.astype(float),
                tvec=extrinsics[i][:3, 3].numpy(),  # Translation vector (3,)
                dist=distortions[i].numpy(),
            )
            cameras.append(camera)

        # Create the CameraGroup
        return CameraGroup(cameras)

    @pytest.fixture
    def valid_data_dict(self, multiview_heatmap_dataset):
        """Create a data dictionary with valid keypoints for testing."""
        datadict = {}
        for view in multiview_heatmap_dataset.view_names:
            datadict[view] = multiview_heatmap_dataset.dataset[view].__getitem__(
                0,
                ignore_nans=True,
            )
            # Ensure we have some valid keypoints by replacing NaNs with reasonable values
            keypoints = datadict[view]["keypoints"]
            if torch.any(torch.isnan(keypoints)):
                # Create a simple pattern of valid keypoints
                num_keypoints = len(keypoints) // 2
                for i in range(num_keypoints):
                    x = 100 + (i % 3) * 50  # x coordinates: 100, 150, 200, ...
                    y = 100 + (i // 3) * 50  # y coordinates: 100, 100, 100, 150, ...
                    keypoints[i * 2] = x
                    keypoints[i * 2 + 1] = y
        return datadict

    def test_augmentation_occurs(
        self,
        multiview_heatmap_dataset,
        valid_data_dict,
        camera_group,
    ):
        """Test that augmentation changes the data when scale/shift parameters are applied."""

        # Set fixed seed for reproducible results
        np.random.seed(42)
        torch.manual_seed(42)

        # Apply transforms with known parameters that should cause changes
        scale_params = (0.5, 0.5)  # Fixed scaling factor
        shift_param = 0.3  # Fixed shift

        datadict_aug, keypoints_3d_aug = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=valid_data_dict,
            camgroup=camera_group,
            scale_params=scale_params,
            shift_param=shift_param,
        )

        # Extract original keypoints for comparison
        original_keypoints_2d = \
            multiview_heatmap_dataset._get_2d_keypoints_from_example_dict_absolute_coords(
                valid_data_dict,
            )
        original_3d = camera_group.triangulate_fast(original_keypoints_2d.copy())

        # Verify that 3D keypoints were actually augmented
        assert not np.allclose(keypoints_3d_aug.numpy(), original_3d, atol=1e-1)

        # Verify that 2D keypoints in each view changed
        for view in multiview_heatmap_dataset.view_names:
            original_kp = valid_data_dict[view]["keypoints"]
            augmented_kp = datadict_aug[view]["keypoints"]

            # Should not be identical (unless all keypoints were NaN, which we've avoided)
            assert not torch.allclose(original_kp, augmented_kp, atol=1e-1)

        # Verify that images were transformed (should have different pixel values)
        for view in multiview_heatmap_dataset.view_names:
            original_img = valid_data_dict[view]["images"]
            augmented_img = datadict_aug[view]["images"]

            # Images should be different after transformation
            assert not torch.allclose(original_img, augmented_img, atol=1e-1)

    def test_no_augmentation_with_identity_params(
        self,
        multiview_heatmap_dataset,
        valid_data_dict,
        camera_group,
    ):
        """Test that no augmentation occurs with identity parameters."""

        # Set fixed seed
        np.random.seed(42)
        torch.manual_seed(42)

        # Use identity parameters (no scaling, no shifting)
        scale_params = (1.0, 1.0)  # No scaling
        shift_param = 0.0  # No shifting

        datadict_aug, keypoints_3d_aug = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=valid_data_dict,
            camgroup=camera_group,
            scale_params=scale_params,
            shift_param=shift_param,
        )

        # Extract original 3D keypoints
        original_keypoints_2d = \
            multiview_heatmap_dataset._get_2d_keypoints_from_example_dict_absolute_coords(
                valid_data_dict,
            )
        original_3d = camera_group.triangulate_fast(original_keypoints_2d.copy())

        # With identity parameters, 3D keypoints should be very similar
        # (small differences may occur due to numerical precision in projection/triangulation)
        assert np.allclose(keypoints_3d_aug.numpy(), original_3d, atol=1e-3)

    def test_scaling_effect(
        self,
        multiview_heatmap_dataset,
        valid_data_dict,
        camera_group,
    ):
        """Test that larger scaling parameters produce more dramatic changes."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Small scaling
        datadict_small, keypoints_3d_small = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=valid_data_dict,
            camgroup=camera_group,
            scale_params=(0.9, 1.1),
            shift_param=0.0,
        )

        # Reset seed to same starting point
        np.random.seed(42)
        torch.manual_seed(42)

        # Large scaling
        datadict_large, keypoints_3d_large = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=valid_data_dict,
            camgroup=camera_group,
            scale_params=(0.5, 2.0),
            shift_param=0.0,
        )

        # Get original keypoints for reference
        original_keypoints_2d = \
            multiview_heatmap_dataset._get_2d_keypoints_from_example_dict_absolute_coords(
                valid_data_dict,
            )
        original_3d = camera_group.triangulate_fast(original_keypoints_2d.copy())

        # Calculate distances from original
        small_distance = np.linalg.norm(keypoints_3d_small.numpy() - original_3d)
        large_distance = np.linalg.norm(keypoints_3d_large.numpy() - original_3d)

        # Larger scaling parameters should produce larger changes
        assert large_distance > small_distance

    def test_translation_effect(
        self,
        multiview_heatmap_dataset,
        valid_data_dict,
        camera_group,
    ):
        """Test that larger translation parameter produces more dramatic changes."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Small shift
        datadict_small, keypoints_3d_small = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=valid_data_dict,
            camgroup=camera_group,
            scale_params=(1.0, 1.0),
            shift_param=0.1,
        )

        # Reset seed to same starting point
        np.random.seed(42)
        torch.manual_seed(42)

        # Large shift
        datadict_large, keypoints_3d_large = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=valid_data_dict,
            camgroup=camera_group,
            scale_params=(1.0, 1.0),
            shift_param=1.0,
        )

        # Get original keypoints for reference
        original_keypoints_2d = \
            multiview_heatmap_dataset._get_2d_keypoints_from_example_dict_absolute_coords(
                valid_data_dict,
            )
        original_3d = camera_group.triangulate_fast(original_keypoints_2d.copy())

        # Calculate distances from original
        small_distance = np.linalg.norm(keypoints_3d_small.numpy() - original_3d)
        large_distance = np.linalg.norm(keypoints_3d_large.numpy() - original_3d)

        # Larger scaling parameters should produce larger changes
        assert large_distance > small_distance

    def test_insufficient_keypoints(
        self,
        multiview_heatmap_dataset,
        valid_data_dict,
        camera_group,
    ):
        """Test fallback behavior when insufficient keypoints are available."""

        def no_change(datadict_to_test):
            np.random.seed(42)
            torch.manual_seed(42)

            datadict_aug, keypoints_3d_aug = multiview_heatmap_dataset.apply_3d_transforms(
                data_dict=datadict_to_test,
                camgroup=camera_group,
                scale_params=(0.5, 2.0),  # Even with extreme params
                shift_param=0.5,
            )

            # Get original keypoints for comparison
            original_keypoints_2d = \
                multiview_heatmap_dataset._get_2d_keypoints_from_example_dict_absolute_coords(
                    datadict_to_test,
                )
            original_3d = camera_group.triangulate_fast(original_keypoints_2d.copy())

            # When insufficient keypoints, should fall back to no augmentation
            # 3D keypoints should be very similar to original (no scaling/translation)
            assert np.allclose(keypoints_3d_aug.numpy(), original_3d, equal_nan=True, atol=1e-3)

            # 2D keypoints should also be very similar
            for view in multiview_heatmap_dataset.view_names:
                original_kp = datadict_to_test[view]["keypoints"]
                augmented_kp = datadict_aug[view]["keypoints"]

                # Should be very close (accounting for numerical precision)
                valid_mask = ~torch.isnan(original_kp)
                assert torch.allclose(
                    original_kp[valid_mask],
                    augmented_kp[valid_mask],
                    equal_nan=True,
                    atol=1e-3,
                )

        # Create data dict with insufficient valid keypoints on all views
        datadict = {}
        for view in multiview_heatmap_dataset.view_names:
            datadict[view] = multiview_heatmap_dataset.dataset[view].__getitem__(
                0,
                ignore_nans=True,
            )
            # Set most keypoints to NaN, leaving only 2 valid ones (insufficient)
            keypoints = datadict[view]["keypoints"]
            keypoints.fill_(float("nan"))
            keypoints[:4] = torch.tensor([100.0, 100.0, 200.0, 200.0])  # Only 2 valid keypoints
        no_change(datadict)

        # Create data dict with insufficient keypoints only on a single view
        # view_name = multiview_heatmap_dataset.view_names[0]
        datadict_1 = copy.deepcopy(valid_data_dict)
        datadict_1[view]["keypoints"].fill_(float("nan"))
        no_change(datadict_1)

        # Create data dict with insufficient keypoints only on a single view, not all nans
        datadict_2 = copy.deepcopy(valid_data_dict)
        datadict_2[view]["keypoints"].fill_(float("nan"))
        datadict_2[view]["keypoints"][:4] = torch.tensor([100.0, 100.0, 200.0, 200.0])
        no_change(datadict_2)

    def test_mismatched_valid_keypoints_across_views(
        self,
        multiview_heatmap_dataset,
        valid_data_dict,
        camera_group,
    ):
        """Test fallback when each view has >=3 valid keypoints but the valid sets don't overlap.

        Without the valid_kps < 3 check, this case would not be caught:
        each view independently has >=3 valid keypoints, but no keypoint is valid in all views
        simultaneously, so triangulation produces 0 valid 3D points. The code would then raise a
        RuntimeError inside _transform_images.
        """

        # num_kp_per_view = (
        #     multiview_heatmap_dataset.num_keypoints // multiview_heatmap_dataset.num_views
        # )
        view_names = multiview_heatmap_dataset.view_names

        datadict = copy.deepcopy(valid_data_dict)

        # View 0: keypoints 0,1,2 valid; rest NaN
        kp0 = datadict[view_names[0]]["keypoints"]
        kp0.fill_(float("nan"))
        kp0[0:6] = torch.tensor([100.0, 100.0, 150.0, 100.0, 200.0, 100.0])

        # View 1: keypoints 3,4,5 valid; rest NaN  (no overlap with view 0)
        kp1 = datadict[view_names[1]]["keypoints"]
        kp1.fill_(float("nan"))
        kp1[6:12] = torch.tensor([110.0, 110.0, 160.0, 110.0, 210.0, 110.0])

        # Each view has 3 valid keypoints, but valid_kps after triangulation == 0 < 3.
        # Without the valid_kps < 3 guard this call raises RuntimeError.
        np.random.seed(42)
        torch.manual_seed(42)
        datadict_aug, keypoints_3d_aug = multiview_heatmap_dataset.apply_3d_transforms(
            data_dict=datadict,
            camgroup=camera_group,
            scale_params=(0.5, 2.0),
            shift_param=0.5,
        )

        # Should fall back: 3D keypoints unchanged (all NaN, nothing triangulated)
        original_keypoints_2d = (
            multiview_heatmap_dataset._get_2d_keypoints_from_example_dict_absolute_coords(datadict)
        )
        original_3d = camera_group.triangulate_fast(original_keypoints_2d.copy())
        assert np.allclose(keypoints_3d_aug.numpy(), original_3d, equal_nan=True, atol=1e-3)

        # 2D keypoints should also be unchanged
        for view in view_names:
            original_kp = datadict[view]["keypoints"]
            augmented_kp = datadict_aug[view]["keypoints"]
            valid_mask = ~torch.isnan(original_kp)
            assert torch.allclose(
                original_kp[valid_mask],
                augmented_kp[valid_mask],
                equal_nan=True,
                atol=1e-3,
            )

    def test_all_nans(self, multiview_heatmap_dataset):

        num_keypoints = multiview_heatmap_dataset.num_keypoints
        num_views = multiview_heatmap_dataset.num_views

        datadict = {}
        for view in multiview_heatmap_dataset.view_names:
            datadict[view] = multiview_heatmap_dataset.dataset[view].__getitem__(
                0, ignore_nans=True,
            )
            datadict[view]["keypoints"].fill_(float("nan"))
        datadict_aug, keypoints_3d = multiview_heatmap_dataset.apply_3d_transforms(datadict, None)
        assert torch.all(torch.isnan(keypoints_3d))
        assert keypoints_3d.shape == (num_keypoints // num_views, 3)
        for view in multiview_heatmap_dataset.view_names:
            assert datadict_aug[view]["images"].shape == (3, 256, 256)
            assert datadict_aug[view]["keypoints"].shape == (num_keypoints,)
            assert datadict_aug[view]["bbox"].shape == (4,)


class TestLoadCamgroup:
    """Test MultiviewHeatmapDataset._load_camgroup."""

    def test_load_camgroup_returns_camgroup(self, mocker, multiview_heatmap_dataset):
        """Loaded camgroup is returned when camera names match view_names."""
        # Arrange
        mock_camgroup = mocker.MagicMock()
        mock_camgroup.get_names.return_value = np.array(multiview_heatmap_dataset.view_names)
        mocker.patch('lightning_pose.data.datasets.CameraGroup.load', return_value=mock_camgroup)

        # Act
        result = multiview_heatmap_dataset._load_camgroup('calibration.toml')

        # Assert
        assert result is mock_camgroup

    def test_load_camgroup_name_mismatch_raises(self, mocker, multiview_heatmap_dataset):
        """AssertionError raised when calibration camera names don't match view_names."""
        # Arrange
        mock_camgroup = mocker.MagicMock()
        mock_camgroup.get_names.return_value = np.array(['wrong', 'names'])
        mocker.patch('lightning_pose.data.datasets.CameraGroup.load', return_value=mock_camgroup)

        # Act / Assert
        with pytest.raises(AssertionError, match='same camera order'):
            multiview_heatmap_dataset._load_camgroup('calibration.toml')


class TestLoadCamParamsFromCsv:
    """Test MultiviewHeatmapDataset._load_cam_params_from_csv."""

    @pytest.fixture
    def cam_params_csv(self, multiview_heatmap_dataset, tmp_path):
        """CSV mapping every frame to calibration.toml."""
        view0 = multiview_heatmap_dataset.view_names[0]
        image_names = multiview_heatmap_dataset.dataset[view0].image_names
        calib_file = 'calibration.toml'
        df = pd.DataFrame(
            {'file': [calib_file] * len(image_names)},
            index=pd.Index([n.split('/')[-1] for n in image_names]),
        )
        csv_path = tmp_path / 'cam_params.csv'
        df.to_csv(csv_path)
        return str(csv_path), calib_file, len(image_names)

    def test_load_cam_params_from_csv_success(
        self, mocker, multiview_heatmap_dataset, cam_params_csv,
    ):
        """Returns aligned DataFrame and camgroup dict for a valid CSV."""
        # Arrange
        csv_path, calib_file, n_frames = cam_params_csv
        mock_camgroup = mocker.MagicMock()
        mock_camgroup.get_names.return_value = np.array(multiview_heatmap_dataset.view_names)
        mocker.patch('lightning_pose.data.datasets.CameraGroup.load', return_value=mock_camgroup)

        # Act
        cam_params_df, cam_params_file_to_camgroup = (
            multiview_heatmap_dataset._load_cam_params_from_csv(csv_path)
        )

        # Assert
        assert len(cam_params_df) == n_frames
        assert list(cam_params_df['file']) == [calib_file] * n_frames
        assert cam_params_file_to_camgroup[calib_file] is mock_camgroup

    def test_load_cam_params_from_csv_do_context_raises(
        self, multiview_heatmap_dataset, cam_params_csv,
    ):
        """AssertionError raised when do_context=True."""
        # Arrange
        csv_path, _, _ = cam_params_csv
        ds = copy.deepcopy(multiview_heatmap_dataset)
        ds.do_context = True

        # Act / Assert
        with pytest.raises(AssertionError):
            ds._load_cam_params_from_csv(csv_path)

    def test_load_cam_params_from_csv_mismatched_names_raises(
        self, multiview_heatmap_dataset, tmp_path,
    ):
        """AssertionError raised when CSV index doesn't align with image names."""
        # Arrange
        df = pd.DataFrame({'file': ['calibration.toml']}, index=pd.Index(['wrong_name.png']))
        csv_path = tmp_path / 'cam_params.csv'
        df.to_csv(csv_path)

        # Act / Assert
        with pytest.raises(AssertionError):
            multiview_heatmap_dataset._load_cam_params_from_csv(str(csv_path))


class TestDiscoverCamParamsFromImagePaths:
    """Test MultiviewHeatmapDataset._discover_cam_params_from_image_paths."""

    @pytest.fixture
    def fake_ds(self, mocker, tmp_path):
        """Minimal stand-in for MultiviewHeatmapDataset with two frames from session0."""
        class _Stub:
            pass

        ds = _Stub()
        ds.root_directory = str(tmp_path)  # type: ignore[attr-defined]
        ds.view_names = ['top', 'bot']  # type: ignore[attr-defined]
        ds.do_context = False  # type: ignore[attr-defined]
        mock_view = mocker.MagicMock()
        mock_view.image_names = [
            'labeled-data/session0_top/img0000.png',
            'labeled-data/session0_top/img0001.png',
        ]
        ds.dataset = {'top': mock_view, 'bot': mocker.MagicMock()}  # type: ignore[attr-defined]
        ds._load_camgroup = mocker.MagicMock(return_value=mocker.MagicMock())  # type: ignore[attr-defined]
        return ds

    def _discover(self, ds):
        return MultiviewHeatmapDataset._discover_cam_params_from_image_paths(ds)

    def test_discover_session_specific_toml(self, fake_ds, tmp_path):
        """Returns per-frame df when calibrations/<session>.toml exists."""
        # Arrange
        calib_dir = tmp_path / 'calibrations'
        calib_dir.mkdir()
        (calib_dir / 'session0.toml').write_text('')

        # Act
        cam_params_df, cam_params_file_to_camgroup = self._discover(fake_ds)

        # Assert
        assert cam_params_df is not None
        assert cam_params_file_to_camgroup is not None
        assert list(cam_params_df['file']) == ['calibrations/session0.toml'] * 2
        assert 'calibrations/session0.toml' in cam_params_file_to_camgroup

    def test_discover_fallback_toml(self, fake_ds, tmp_path):
        """Falls back to calibration.toml when no session-specific file exists."""
        # Arrange
        (tmp_path / 'calibration.toml').write_text('')

        # Act
        cam_params_df, cam_params_file_to_camgroup = self._discover(fake_ds)

        # Assert
        assert cam_params_df is not None
        assert cam_params_file_to_camgroup is not None
        assert list(cam_params_df['file']) == ['calibration.toml'] * 2
        assert 'calibration.toml' in cam_params_file_to_camgroup

    def test_discover_no_calibration_returns_none(self, fake_ds):
        """Returns (None, None) when no calibration file is found."""
        # Act
        cam_params_df, cam_params_file_to_camgroup = self._discover(fake_ds)

        # Assert
        assert cam_params_df is None
        assert cam_params_file_to_camgroup is None

    def test_discover_mixed_calibration_disables_3d(self, fake_ds, tmp_path, caplog):
        """Returns (None, None) and logs a warning when only some frames have calibration."""
        # Arrange: session0 has a toml, session1 does not
        calib_dir = tmp_path / 'calibrations'
        calib_dir.mkdir()
        (calib_dir / 'session0.toml').write_text('')
        fake_ds.dataset['top'].image_names = [
            'labeled-data/session0_top/img0000.png',
            'labeled-data/session1_top/img0001.png',
        ]

        # Act
        with caplog.at_level(logging.WARNING, logger='lightning_pose'):
            cam_params_df, cam_params_file_to_camgroup = self._discover(fake_ds)

        # Assert
        assert cam_params_df is None
        assert cam_params_file_to_camgroup is None
        assert any(
            'calibration file not found' in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_discover_multi_session(self, fake_ds, tmp_path):
        """Each session is mapped to its own calibration file."""
        # Arrange
        calib_dir = tmp_path / 'calibrations'
        calib_dir.mkdir()
        (calib_dir / 'sessionA.toml').write_text('')
        (calib_dir / 'sessionB.toml').write_text('')
        fake_ds.dataset['top'].image_names = [
            'labeled-data/sessionA_top/img0000.png',
            'labeled-data/sessionB_top/img0001.png',
        ]

        # Act
        cam_params_df, cam_params_file_to_camgroup = self._discover(fake_ds)

        # Assert
        assert cam_params_df is not None
        assert cam_params_file_to_camgroup is not None
        assert list(cam_params_df['file']) == [
            'calibrations/sessionA.toml',
            'calibrations/sessionB.toml',
        ]
        assert 'calibrations/sessionA.toml' in cam_params_file_to_camgroup
        assert 'calibrations/sessionB.toml' in cam_params_file_to_camgroup
        assert fake_ds._load_camgroup.call_count == 2

    def test_discover_path_without_labeled_data_raises(self, fake_ds):
        """ValueError raised when image path doesn't contain labeled-data/."""
        # Arrange
        fake_ds.dataset['top'].image_names = ['some/other/path/img0000.png']

        # Act / Assert
        with pytest.raises(ValueError, match='labeled-data'):
            self._discover(fake_ds)

    def test_discover_folder_without_underscore_raises(self, fake_ds):
        """ValueError raised when folder name has no underscore separating session and view."""
        # Arrange
        fake_ds.dataset['top'].image_names = ['labeled-data/sessiononly/img0000.png']

        # Act / Assert
        with pytest.raises(ValueError, match='expected pattern'):
            self._discover(fake_ds)

    def test_discover_do_context_raises_when_calibration_found(self, fake_ds, tmp_path):
        """AssertionError raised when do_context=True and a calibration file is found."""
        # Arrange
        fake_ds.do_context = True
        calib_dir = tmp_path / 'calibrations'
        calib_dir.mkdir()
        (calib_dir / 'session0.toml').write_text('')

        # Act / Assert
        with pytest.raises(AssertionError, match='context model'):
            self._discover(fake_ds)


class TestBaseTrackingDatasetVisibility:
    """Test visibility column parsing in BaseTrackingDataset and HeatmapDataset."""

    @pytest.fixture
    def visibility_csv(self, tmp_path) -> str:
        """DLC-format CSV with a visible column; two keypoints, two frames."""
        content = (
            'scorer,scorer,scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp1,kp2,kp2,kp2\n'
            'coords,x,y,visible,x,y,visible\n'
            # frame 0: kp1 visible (2) with valid coords; kp2 occluded (1) with NaN coords
            'img01.png,64.0,64.0,2,,,1\n'
            # frame 1: kp1 visible (2); kp2 not labeled (0) with NaN coords
            'img02.png,32.0,96.0,2,,,0\n'
        )
        p = tmp_path / 'labels.csv'
        p.write_text(content)
        return str(p)

    @pytest.fixture
    def standard_csv(self, tmp_path) -> str:
        """Standard DLC-format CSV without visibility columns."""
        content = (
            'scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp2,kp2\n'
            'coords,x,y,x,y\n'
            'img01.png,64.0,64.0,32.0,96.0\n'
            'img02.png,10.0,20.0,30.0,40.0\n'
        )
        p = tmp_path / 'labels_standard.csv'
        p.write_text(content)
        return str(p)

    @pytest.fixture
    def imgaug(self):
        return iaa.Sequential([])

    def _make_base_dataset(self, csv_path, tmp_path, imgaug):
        return BaseTrackingDataset(
            root_directory=str(tmp_path),
            csv_path=csv_path,
            image_resize_height=128,
            image_resize_width=128,
            imgaug_transform=imgaug,
        )

    def _make_heatmap_dataset(self, csv_path, tmp_path, imgaug):
        return HeatmapDataset(
            root_directory=str(tmp_path),
            csv_path=csv_path,
            image_resize_height=128,
            image_resize_width=128,
            imgaug_transform=imgaug,
        )

    def test_base_tracking_dataset_visibility_parsed(self, visibility_csv, tmp_path, imgaug):
        """visibility tensor is populated with correct int values from the CSV."""
        ds = self._make_base_dataset(visibility_csv, tmp_path, imgaug)

        assert ds.visibility is not None
        assert ds.visibility.shape == (2, 2)  # (N=2 frames, K=2 keypoints)
        assert ds.visibility.dtype == torch.long
        # frame 0: kp1=visible(2), kp2=occluded(1)
        assert ds.visibility[0, 0] == 2
        assert ds.visibility[0, 1] == 1
        # frame 1: kp1=visible(2), kp2=not_labeled(0)
        assert ds.visibility[1, 0] == 2
        assert ds.visibility[1, 1] == 0

    def test_base_tracking_dataset_no_visibility(self, standard_csv, tmp_path, imgaug):
        """visibility is None for a standard CSV without visible column."""
        ds = self._make_base_dataset(standard_csv, tmp_path, imgaug)
        assert ds.visibility is None

    def test_base_tracking_dataset_invalid_visibility_raises(self, tmp_path, imgaug):
        """ValueError is raised when the visibility column contains values outside {0, 1, 2}."""
        content = (
            'scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp1\n'
            'coords,x,y,visible\n'
            'img01.png,64.0,64.0,9\n'
        )
        bad_csv = tmp_path / 'bad.csv'
        bad_csv.write_text(content)
        with pytest.raises(ValueError, match='visibility column contains invalid values'):
            BaseTrackingDataset(
                root_directory=str(tmp_path),
                csv_path=str(bad_csv),
                image_resize_height=128,
                image_resize_width=128,
                imgaug_transform=imgaug,
            )

    def test_base_tracking_dataset_occluded_with_coords_warns(
        self, tmp_path, imgaug, caplog,
    ):
        """A warning is logged when vis=1 keypoints have non-NaN x,y coordinates."""
        content = (
            'scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp1\n'
            'coords,x,y,visible\n'
            # kp1 has coordinates but is marked occluded (vis=1)
            'img01.png,64.0,64.0,1\n'
        )
        p = tmp_path / 'occluded_with_coords.csv'
        p.write_text(content)
        with caplog.at_level(logging.WARNING, logger='lightning_pose.data.datasets'):
            BaseTrackingDataset(
                root_directory=str(tmp_path),
                csv_path=str(p),
                image_resize_height=128,
                image_resize_width=128,
                imgaug_transform=imgaug,
            )
        assert any('visible=1' in record.message for record in caplog.records)

    def test_heatmap_dataset_getitem_populates_visibility(self, visibility_csv, tmp_path, imgaug):
        """__getitem__ populates the 'visibility' key from self.visibility[idx]."""
        for name in ('img01.png', 'img02.png'):
            Image.fromarray(
                (128 * torch.ones(128, 128, 3)).byte().numpy()
            ).save(os.path.join(str(tmp_path), name))

        ds = self._make_heatmap_dataset(visibility_csv, tmp_path, imgaug)

        ex0 = BaseTrackingDataset.__getitem__(ds, 0)
        assert ex0['visibility'] is not None
        assert torch.equal(ex0['visibility'], torch.tensor([2, 1], dtype=torch.long))

        ex1 = BaseTrackingDataset.__getitem__(ds, 1)
        assert ex1['visibility'] is not None
        assert torch.equal(ex1['visibility'], torch.tensor([2, 0], dtype=torch.long))

    def test_heatmap_dataset_getitem_no_visibility(self, standard_csv, tmp_path, imgaug):
        """HeatmapDataset synthesizes visibility=2 for all valid keypoints in a standard CSV."""
        for name in ('img01.png', 'img02.png'):
            Image.fromarray(
                (128 * torch.ones(128, 128, 3)).byte().numpy()
            ).save(os.path.join(str(tmp_path), name))

        ds = self._make_heatmap_dataset(standard_csv, tmp_path, imgaug)
        # standard CSV → synthesis sets vis=2 for all labeled (non-NaN) keypoints
        assert ds.visibility is not None
        assert (ds.visibility == 2).all()
        ex = BaseTrackingDataset.__getitem__(ds, 0)
        assert torch.equal(ex['visibility'], torch.tensor([2, 2], dtype=torch.long))

    def test_heatmap_dataset_compute_heatmap_uses_visibility(
        self, visibility_csv, tmp_path, imgaug,
    ):
        """compute_heatmap reads visibility from example_dict, not self.visibility[idx]."""

        ds = self._make_heatmap_dataset(visibility_csv, tmp_path, imgaug)
        H, W = ds.output_shape

        uniform_heatmap = torch.ones(H, W) / (H * W)
        zero_heatmap = torch.zeros(H, W)

        # Frame 0: kp1=vis2 with valid coords, kp2=vis1 with NaN coords
        example_dict = {
            'images': torch.zeros(3, 128, 128),
            'keypoints': torch.tensor([64.0, 64.0, float('nan'), float('nan')]),
            'bbox': torch.tensor([0, 0, 128, 128]),
            'idxs': 0,
            'visibility': torch.tensor([2, 1], dtype=torch.long),
        }
        heatmaps = ds.compute_heatmap(example_dict, ignore_nans=True)  # type: ignore[arg-type]

        # kp1 (vis=2): Gaussian — non-zero, sums to ~1
        assert not torch.allclose(heatmaps[0], zero_heatmap)
        assert torch.isclose(heatmaps[0].sum(), torch.tensor(1.0))
        # kp2 (vis=1): uniform
        assert torch.allclose(heatmaps[1], uniform_heatmap)

        # Frame 1: kp2=vis0 — zero heatmap; visibility from example_dict, not self.visibility[1]
        example_dict_1 = {
            'images': torch.zeros(3, 128, 128),
            'keypoints': torch.tensor([32.0, 96.0, float('nan'), float('nan')]),
            'bbox': torch.tensor([0, 0, 128, 128]),
            'idxs': 1,
            'visibility': torch.tensor([2, 0], dtype=torch.long),
        }
        heatmaps_1 = ds.compute_heatmap(example_dict_1, ignore_nans=True)  # type: ignore[arg-type]
        assert torch.allclose(heatmaps_1[1], zero_heatmap)

        # Verify compute_heatmap uses example_dict["visibility"], not self.visibility[idx]:
        # pass a different visibility tensor than what self.visibility[0] would return
        example_dict_override = {
            'images': torch.zeros(3, 128, 128),
            'keypoints': torch.tensor([64.0, 64.0, float('nan'), float('nan')]),
            'bbox': torch.tensor([0, 0, 128, 128]),
            'idxs': 0,
            # override kp1 to vis=0 — should produce zero heatmap despite kp1 having coords
            'visibility': torch.tensor([0, 1], dtype=torch.long),
        }
        heatmaps_override = ds.compute_heatmap(
            example_dict_override, ignore_nans=True,  # type: ignore[arg-type]
        )
        assert torch.allclose(heatmaps_override[0], zero_heatmap)


def test_equal_return_sizes(base_dataset, heatmap_dataset):
    # can only assert the batches are the same if not using imgaug pipeline
    assert base_dataset[0]["images"].shape == heatmap_dataset[0]["images"].shape


class TestBuildHflipSwapIndices:
    """Test BaseTrackingDataset._build_hflip_swap_indices."""

    def test_no_lateralized_keypoints_returns_identity(self):
        """keypoints with no _left/_right suffix map to themselves."""
        names = ['nose', 'tail', 'spine']
        indices = BaseTrackingDataset._build_hflip_swap_indices(names)
        assert list(indices) == [0, 1, 2]

    def test_single_pair_swaps(self):
        """a single _left/_right pair is correctly swapped."""
        names = ['nose', 'ear_left', 'ear_right', 'tail']
        indices = BaseTrackingDataset._build_hflip_swap_indices(names)
        # ear_left (idx 1) <-> ear_right (idx 2); nose and tail stay
        assert list(indices) == [0, 2, 1, 3]

    def test_multiple_pairs_swap(self):
        """multiple _left/_right pairs are all correctly swapped."""
        names = ['ear_left', 'paw_left', 'nose', 'paw_right', 'ear_right']
        indices = BaseTrackingDataset._build_hflip_swap_indices(names)
        # ear_left(0)<->ear_right(4), paw_left(1)<->paw_right(3), nose(2) stays
        assert list(indices) == [4, 3, 2, 1, 0]

    def test_unmatched_left_raises(self):
        """ValueError raised when a _left keypoint has no _right partner."""
        with pytest.raises(ValueError, match='_left'):
            BaseTrackingDataset._build_hflip_swap_indices(['ear_left', 'nose'])

    def test_unmatched_right_raises(self):
        """ValueError raised when a _right keypoint has no _left partner."""
        with pytest.raises(ValueError, match='_right'):
            BaseTrackingDataset._build_hflip_swap_indices(['ear_right', 'nose'])

    def test_return_dtype_is_intp(self):
        """returned array has dtype np.intp for safe indexing."""
        indices = BaseTrackingDataset._build_hflip_swap_indices(['a_left', 'a_right'])
        assert indices.dtype == np.intp


class TestImgaugHflip:
    """Test the imgaug_hflip augmentation in BaseTrackingDataset and HeatmapDataset."""

    @pytest.fixture
    def lateralized_csv(self, tmp_path) -> str:
        """DLC-format CSV with _left/_right keypoints and one non-lateralized keypoint."""
        # 5 keypoints: nose, ear_left, ear_right, paw_left, paw_right (x, y each)
        bp = (
            'nose,nose,ear_left,ear_left,ear_right,ear_right,'
            'paw_left,paw_left,paw_right,paw_right'
        )
        content = (
            'scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer\n'
            f'bodyparts,{bp}\n'
            'coords,x,y,x,y,x,y,x,y,x,y\n'
            'img01.png,64.0,64.0,30.0,40.0,90.0,40.0,20.0,80.0,100.0,80.0\n'
            'img02.png,64.0,64.0,35.0,45.0,85.0,45.0,25.0,75.0,95.0,75.0\n'
        )
        p = tmp_path / 'lateralized.csv'
        p.write_text(content)
        return str(p)

    @pytest.fixture
    def dummy_images(self, tmp_path):
        """Write two dummy PNG images so __getitem__ can open them."""
        for name in ('img01.png', 'img02.png'):
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / name)

    @pytest.fixture
    def hflip_dataset(self, lateralized_csv, dummy_images, tmp_path):
        """HeatmapDataset with imgaug_hflip=True."""
        return HeatmapDataset(
            root_directory=str(tmp_path),
            csv_path=lateralized_csv,
            image_resize_height=128,
            image_resize_width=128,
            imgaug_transform=iaa.Sequential([]),
            imgaug_hflip=True,
        )

    def test_swap_indices_built_correctly(self, hflip_dataset):
        """_hflip_swap_indices correctly maps lateralized pairs."""
        # keypoints: nose(0), ear_left(1), ear_right(2), paw_left(3), paw_right(4)
        expected = np.array([0, 2, 1, 4, 3], dtype=np.intp)
        assert np.array_equal(hflip_dataset._hflip_swap_indices, expected)

    def test_unmatched_left_raises_at_init(self, tmp_path):
        """ValueError raised at dataset init when a _left has no _right partner."""
        content = (
            'scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,ear_left,ear_left,nose,nose\n'
            'coords,x,y,x,y\n'
            'img01.png,30.0,40.0,64.0,64.0\n'
        )
        (tmp_path / 'bad.csv').write_text(content)
        with pytest.raises(ValueError, match='_left'):
            HeatmapDataset(
                root_directory=str(tmp_path),
                csv_path=str(tmp_path / 'bad.csv'),
                image_resize_height=128,
                image_resize_width=128,
                imgaug_transform=iaa.Sequential([]),
                imgaug_hflip=True,
            )

    def test_hflip_false_does_not_build_swap_indices(self, tmp_path):
        """With imgaug_hflip=False, no validation is run and identity indices are stored."""
        # CSV with unmatched _left — would raise if imgaug_hflip=True, but not with False
        content = (
            'scorer,scorer,scorer\n'
            'bodyparts,ear_left,ear_left\n'
            'coords,x,y\n'
            'img01.png,30.0,40.0\n'
        )
        (tmp_path / 'unmatched.csv').write_text(content)
        ds = HeatmapDataset(
            root_directory=str(tmp_path),
            csv_path=str(tmp_path / 'unmatched.csv'),
            image_resize_height=128,
            image_resize_width=128,
            imgaug_transform=iaa.Sequential([]),
            imgaug_hflip=False,
        )
        assert not ds.imgaug_hflip
        assert np.array_equal(ds._hflip_swap_indices, np.arange(1, dtype=np.intp))

    def test_hflip_mirrors_x_coordinates(self, hflip_dataset):
        """After a forced hflip, all x-coordinates are mirrored and lateralized pairs swapped."""
        width = hflip_dataset.image_resize_width
        # original keypoints for frame 0: nose(64,64), ear_left(30,40), ear_right(90,40),
        # paw_left(20,80), paw_right(100,80)
        orig_kps = hflip_dataset.keypoints[0].numpy()  # (5, 2)

        with patch('numpy.random.random', return_value=0.0):  # force flip (0.0 < 0.5)
            batch = hflip_dataset[0]

        kps = batch['keypoints'].numpy().reshape(5, 2)

        # after flip and swap:
        # position 0 (nose, non-lateral): x = 128 - 64 = 64
        assert np.isclose(kps[0, 0], width - orig_kps[0, 0])
        # position 1 gets ear_right's flipped x: 128 - 90 = 38
        assert np.isclose(kps[1, 0], width - orig_kps[2, 0])
        # position 2 gets ear_left's flipped x: 128 - 30 = 98
        assert np.isclose(kps[2, 0], width - orig_kps[1, 0])
        # y-coordinates are unchanged for both lateralized keypoints
        assert np.isclose(kps[1, 1], orig_kps[2, 1])
        assert np.isclose(kps[2, 1], orig_kps[1, 1])

    def test_hflip_no_flip_when_random_above_half(self, hflip_dataset):
        """When random >= 0.5, no flip is applied and keypoints are unchanged."""
        orig_kps = hflip_dataset.keypoints[0].numpy()

        with patch('numpy.random.random', return_value=0.9):  # force no flip
            batch = hflip_dataset[0]

        kps = batch['keypoints'].numpy().reshape(5, 2)
        # after standard imgaug (empty pipeline + resize to 128x128), coords are unchanged
        assert np.allclose(kps, orig_kps, equal_nan=True)

    def test_hflip_swaps_visibility(self, tmp_path, dummy_images):
        """After hflip, visibility values are swapped for lateralized pairs."""
        content = (
            'scorer,scorer,scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,ear_left,ear_left,ear_left,ear_right,ear_right,ear_right\n'
            'coords,x,y,visible,x,y,visible\n'
            'img01.png,30.0,40.0,2,90.0,40.0,1\n'
            'img02.png,30.0,40.0,1,90.0,40.0,2\n'
        )
        (tmp_path / 'vis.csv').write_text(content)
        ds = HeatmapDataset(
            root_directory=str(tmp_path),
            csv_path=str(tmp_path / 'vis.csv'),
            image_resize_height=128,
            image_resize_width=128,
            imgaug_transform=iaa.Sequential([]),
            imgaug_hflip=True,
        )
        # frame 0: ear_left=vis2, ear_right=vis1; after hflip swap → pos0=vis1, pos1=vis2
        with patch('numpy.random.random', return_value=0.0):
            batch = ds[0]
        assert batch['visibility'][0].item() == 1  # was ear_right
        assert batch['visibility'][1].item() == 2  # was ear_left

    def test_hflip_context_all_frames_flipped(self, lateralized_csv, tmp_path):
        """In context mode, all frames receive the same horizontal flip."""
        # create enough context frames: img01.png needs neighbours
        for i in range(10):
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[:, :64, 0] = 200  # left half red → right half after flip
            Image.fromarray(arr).save(tmp_path / f'img{i:02d}.png')

        # rewrite csv to use img05.png as center (has neighbours on both sides)
        bp = (
            'nose,nose,ear_left,ear_left,ear_right,ear_right,'
            'paw_left,paw_left,paw_right,paw_right'
        )
        content = (
            'scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer,scorer\n'
            f'bodyparts,{bp}\n'
            'coords,x,y,x,y,x,y,x,y,x,y\n'
            'img05.png,64.0,64.0,30.0,40.0,90.0,40.0,20.0,80.0,100.0,80.0\n'
        )
        (tmp_path / 'ctx.csv').write_text(content)

        ds = HeatmapDataset(
            root_directory=str(tmp_path),
            csv_path=str(tmp_path / 'ctx.csv'),
            image_resize_height=128,
            image_resize_width=128,
            imgaug_transform=iaa.Sequential([]),
            imgaug_hflip=True,
            do_context=True,
        )

        with patch('numpy.random.random', return_value=0.0):
            batch = ds[0]

        # images tensor is (5, 3, H, W); after flip the right half should be brighter
        images = batch['images']  # (5, 3, 128, 128)
        assert images.shape == (5, 3, 128, 128)
        for frame_idx in range(5):
            left_mean = images[frame_idx, 0, :, :64].mean().item()
            right_mean = images[frame_idx, 0, :, 64:].mean().item()
            # original left half was red (high), so after flip right half should be higher
            assert right_mean > left_mean, (
                f'frame {frame_idx}: expected right > left after hflip'
            )

    def test_hflip_disabled_for_val_test_in_datamodule(
        self, cfg, hflip_dataset,
    ):
        """datamodule resets imgaug_hflip=False on val and test subsets."""
        dm = BaseDataModule(
            dataset=hflip_dataset,
            train_batch_size=2,
            val_batch_size=2,
            test_batch_size=2,
            train_probability=0.6,
            val_probability=0.2,
        )
        assert dm.train_dataset.dataset.imgaug_hflip is True  # type: ignore[union-attr]
        assert dm.val_dataset.dataset.imgaug_hflip is False  # type: ignore[union-attr]
        assert dm.test_dataset.dataset.imgaug_hflip is False  # type: ignore[union-attr]
