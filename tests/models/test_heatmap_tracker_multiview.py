"""Test the initialization, training, and inference of multiview heatmap models."""

import copy

import numpy as np
import torch

from lightning_pose.data.datasets import MultiviewHeatmapDataset
from lightning_pose.data.datatypes import MultiviewHeatmapLabeledExampleDict


def test_multiview_transformer(
    cfg_multiview,
    multiview_heatmap_dataset,
    video_dataloader_multiview,
    trainer,
    run_model_test,
):
    """Test initialization and training of a multiview model with heatmap head."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_multiview_transformer"
    cfg_tmp.model.backbone = "vits_dino"
    cfg_tmp.model.head = "heatmap_cnn"
    cfg_tmp.model.losses_to_use = []

    # make mock dataset that returns fake camera parameters
    from lightning_pose.utils.scripts import get_data_module
    data_module = get_data_module(
        cfg_tmp,
        dataset=MockCameraDatasetWrapper(
            multiview_heatmap_dataset,
            num_views=len(cfg_tmp.data.view_names),
        ),
        video_dir=None,
    )

    run_model_test(
        cfg=cfg_tmp,
        data_module=data_module,
        video_dataloader=video_dataloader_multiview,
        trainer=trainer,
    )


def test_semisupervised_multiview_transformer_temporal(
    cfg_multiview,
    multiview_heatmap_dataset,
    video_dataloader_multiview,
    trainer,
    run_model_test,
):
    """Test initialization and training of a semi-supervised multiview model with heatmap head."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap_multiview_transformer"
    cfg_tmp.model.backbone = "vits_dino"
    cfg_tmp.model.head = "heatmap_cnn"
    cfg_tmp.model.losses_to_use = ["temporal"]

    # make mock dataset that returns fake camera parameters
    from lightning_pose.utils.scripts import get_data_module
    data_module = get_data_module(
        cfg_tmp,
        dataset=MockCameraDatasetWrapper(
            multiview_heatmap_dataset,
            num_views=len(cfg_tmp.data.view_names),
        ),
        video_dir=cfg_tmp.data.video_dir,
    )

    run_model_test(
        cfg=cfg_tmp,
        data_module=data_module,
        video_dataloader=video_dataloader_multiview,
        trainer=trainer,
    )


class MockCameraDatasetWrapper(MultiviewHeatmapDataset):
    """Wrapper that adds mock camera calibration data to an existing dataset."""

    def __init__(self, original_dataset, num_views=2, image_height=256, image_width=256):
        self.original_dataset = original_dataset
        self.view_names = [f"view{i}" for i in range(num_views)]
        self.image_resize_height = image_height
        self.image_resize_width = image_width

    def __len__(self):
        return len(self.original_dataset)

    def __getattr__(self, name):
        # Delegate all other attributes to the original dataset
        return getattr(self.original_dataset, name)

    def __getitem__(self, idx: int) -> MultiviewHeatmapLabeledExampleDict:
        """Override __getitem__ to add mock camera calibration data."""

        # Get the original data
        original_data = self.original_dataset.__getitem__(idx)

        # Create realistic mock camera calibration data

        # Mock intrinsic matrices (typical camera parameters)
        # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        intrinsic_matrix = torch.zeros(self.num_views, 3, 3)
        for view_idx in range(self.num_views):
            # Realistic focal lengths and principal points
            fx = fy = 800.0 + torch.randn(1) * 50  # ~800 ± 50 pixels
            cx = self.image_resize_width / 2 + torch.randn(1) * 10  # near image center
            cy = self.image_resize_height / 2 + torch.randn(1) * 10

            intrinsic_matrix[view_idx] = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

        # Mock extrinsic matrices (camera poses)
        # Format: [R|t] where R is 3x3 rotation, t is 3x1 translation
        extrinsic_matrix = torch.zeros(self.num_views, 3, 4)
        for view_idx in range(self.num_views):
            # Create rotation around Y-axis for different viewpoints
            angle = (view_idx / self.num_views) * 2 * np.pi
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Rotation matrix (rotation around Y-axis)
            R = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])

            # Translation (cameras arranged in a circle)
            radius = 2.0  # meters
            t = torch.tensor([
                radius * np.sin(angle),
                0.0,
                radius * np.cos(angle)
            ]).unsqueeze(1)

            extrinsic_matrix[view_idx] = torch.cat([R, t], dim=1)

        # Mock distortion parameters (assuming 5 parameters: k1, k2, p1, p2, k3)
        distortions = torch.randn(self.num_views, 5) * 0.1  # Small distortions

        # Mock 3D keypoints (assuming same number as 2D keypoints)
        if hasattr(original_data, 'keypoints') and original_data['keypoints'].numel() > 1:
            num_keypoints = original_data['keypoints'].shape[0] // 2  # Assuming x,y pairs
            keypoints_3d = torch.randn(num_keypoints, 3) * 0.5  # Random 3D points
        else:
            keypoints_3d = torch.randn(17, 3) * 0.5  # Default to 17 keypoints (COCO-style)

        # Update the original data with mock camera parameters
        updated_data = dict(original_data)
        updated_data.update({
            'keypoints_3d': keypoints_3d,
            'intrinsic_matrix': intrinsic_matrix,
            'extrinsic_matrix': extrinsic_matrix,
            'distortions': distortions,
        })

        return MultiviewHeatmapLabeledExampleDict(**updated_data)

    # Implement copy support to avoid deepcopy issues
    def __copy__(self):
        return MockCameraDatasetWrapper(
            copy.copy(self.original_dataset),
            self.num_views,
            self.image_resize_height,
            self.image_resize_width,
        )

    def __deepcopy__(self, memo):
        return MockCameraDatasetWrapper(
            copy.deepcopy(self.original_dataset, memo),
            self.num_views,
            self.image_resize_height,
            self.image_resize_width,
        )
