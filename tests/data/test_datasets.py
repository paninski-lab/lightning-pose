"""Test basic dataset functionality."""

import copy
from unittest.mock import MagicMock, patch

import cv2
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

@pytest.fixture
def camera_group():
    """Create a mock CameraGroup object."""

    # ------------------------------------------
    # hard-coded values from anipose-fly example
    # (using aniposelib CameraGroup object)
    # ------------------------------------------

    from aniposelib.cameras import Camera
    from lightning_pose.data.cameras import CameraGroup

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
            rvec=rvec,
            tvec=extrinsics[i][:3, 3].numpy(),  # Translation vector (3,)
            dist=distortions[i].numpy(),
        )
        cameras.append(camera)

    # Create the CameraGroup
    return CameraGroup(cameras)

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

    def test_sufficient_keypoints_for_augmentation(self):
        """Test _sufficient_keypoints_for_augmentation with various keypoint configurations."""

        # Test case: All views have sufficient keypoints (>=3 valid)
        kps_1 = np.array([
            # View 1: 4 valid keypoints
            [[100, 200], [150, 250], [250, 350]],
            # View 2: 3 valid keypoints (minimum required)
            [[110, 210], [160, 260], [210, 310]],
            # View 3: 5 valid keypoints
            [[120, 220], [170, 270], [320, 420]],
        ])
        assert MultiviewHeatmapDataset._sufficient_keypoints_for_augmentation(kps_1)

        # Test case: Views with NaN values but still sufficient valid keypoints
        kps_2 = np.array([
            # View 1: 4 valid keypoints + 1 NaN keypoint
            [[100, 200], [150, 250], [200, 300], [250, 350], [np.nan, np.nan]],
            # View 2: 3 valid keypoints + 2 partial NaN keypoints
            [[110, 210], [160, 260], [210, 310], [np.nan, np.nan], [np.nan, np.nan]],
            # View 3: 5 valid keypoints + 1 completely NaN keypoint
            [[120, 220], [170, 270], [220, 320], [270, 370], [320, 420]],
        ])
        assert MultiviewHeatmapDataset._sufficient_keypoints_for_augmentation(kps_2)

        # Test case: Views with NaN values that result in insufficient valid keypoints
        kps_3 = np.array([
            # View 1: 3 valid keypoints + 2 NaN keypoints
            [[100, 200], [150, 250], [200, 300], [np.nan, np.nan], [np.nan, np.nan]],
            # View 2: Only 2 valid keypoints + 3 NaN/partial NaN keypoints
            [[110, 210], [160, 260], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            # View 3: 4 valid keypoints
            [[120, 220], [170, 270], [220, 320], [270, 370], [275, 375]],
        ])
        assert not MultiviewHeatmapDataset._sufficient_keypoints_for_augmentation(kps_3)

        # Test case: All keypoints are NaN
        kps_4 = np.array([
            # View 1: All NaN keypoints
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            # View 2: All NaN keypoints
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ])
        assert not MultiviewHeatmapDataset._sufficient_keypoints_for_augmentation(kps_4)

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

    def test_init_with_discovered_calibrations(
        self, cfg_multiview, imgaug_transform, tmp_path, camera_group,
    ):
        """Test initialization when camera_params_path is None and calibrations are discovered."""
        import shutil
        from lightning_pose.utils.scripts import get_dataset

        # 1. Setup mock root directory by copying data_dir
        # Use toy data as base but move to tmp_path
        shutil.copytree(cfg_multiview.data.data_dir, tmp_path, dirs_exist_ok=True)

        # 2. Setup calibrations directory and serialized CameraGroup
        calib_dir = tmp_path / "calibrations"
        calib_dir.mkdir()
        calib_file = calib_dir / "session0.toml"
        camera_group.cameras[0].name = 'top'
        camera_group.cameras[1].name = 'bot'
        camera_group.cameras[0].size = (1080, 1920),  # Fake size, doesn't get used.
        camera_group.cameras[1].size = (1080, 1920),  # Fake size, doesn't get used.
        camera_group.dump(str(calib_file))

        # 3. Modify config to point to tmp_path
        cfg_tmp = copy.deepcopy(cfg_multiview)
        assert cfg_tmp.model.model_type == "heatmap"
        cfg_tmp.data.data_dir = str(tmp_path)

        # 4. Construct dataset with camera_params_path=None (default)
        dataset = get_dataset(
            cfg_tmp,
            data_dir=str(tmp_path),
            imgaug_transform=imgaug_transform,
        )

        # 5. Verifications
        assert isinstance(dataset, MultiviewHeatmapDataset)
        # verify calibrations were found in the directory
        assert dataset.cam_params_df is None
        assert dataset.cam_params_file_to_camgroup is not None
        assert "calibrations/session0.toml" in dataset.cam_params_file_to_camgroup
        assert isinstance(
            dataset.cam_params_file_to_camgroup["calibrations/session0.toml"],
            type(camera_group),
        )


class TestApply3DTransforms:
    """Tests for MultiviewHeatmapDataset.apply_3d_transforms method."""


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
        view_name = multiview_heatmap_dataset.view_names[0]
        datadict_1 = copy.deepcopy(valid_data_dict)
        datadict_1[view]["keypoints"].fill_(float("nan"))
        no_change(datadict_1)

        # Create data dict with insufficient keypoints only on a single view, not all nans
        datadict_2 = copy.deepcopy(valid_data_dict)
        datadict_2[view]["keypoints"].fill_(float("nan"))
        datadict_2[view]["keypoints"][:4] = torch.tensor([100.0, 100.0, 200.0, 200.0])
        no_change(datadict_2)

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


def test_equal_return_sizes(base_dataset, heatmap_dataset):
    # can only assert the batches are the same if not using imgaug pipeline
    assert base_dataset[0]["images"].shape == heatmap_dataset[0]["images"].shape
