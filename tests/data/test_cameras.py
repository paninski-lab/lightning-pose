import pytest
import torch

from lightning_pose.data.cameras import project_3d_to_2d, project_camera_pairs_to_3d

# --------------------------------------------
# file-level fixtures from anipose-fly example
# 05272019_fly1_0_R1C24_rot-ccw-0.06_sec
# --------------------------------------------

@pytest.fixture
def points_2d():
    return torch.tensor(
        [[[244.00537479, 169.14706428],
          [157.12294343, 177.34581021]],

         [[260.49358356, 119.39195845],
          [168.3701722 , 130.26763126]],

         [[463.32075159, 186.66484865],
          [303.22096047, 260.25092964]]],
    )


@pytest.fixture
def points_3d():
    return torch.tensor(
        [[[[-1.59978732, -0.39616025, 3.38618251],
           [-2.05475903, -0.4051827 , 3.45997281]],

          [[-1.59978732, -0.39616025, 3.38618251],
           [-2.05475903, -0.4051827 , 3.45997281]],

          [[-1.59978732, -0.39616025, 3.38618251],
           [-2.05475903, -0.4051827 , 3.45997281]]]]
    )


@pytest.fixture
def points_3d_single():
    """Single 3D keypoints for projection to 2D (without camera pair dimension)."""
    return torch.tensor(
        [[-1.59978732, -0.39616025, 3.38618251],
         [-2.05475903, -0.4051827 , 3.45997281]]
    )


@pytest.fixture
def intrinsics():
    return torch.tensor(
        [[[1.4633e+04, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.4633e+04, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.6343e+04, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.6343e+04, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[1.0100e+05, 0.0000e+00, 4.1600e+02],
          [0.0000e+00, 1.0100e+05, 3.1600e+02],
          [0.0000e+00, 0.0000e+00, 1.0000e+00]]],
    )


@pytest.fixture
def extrinsics():
    return torch.tensor(
        [[[7.9065e-01, -1.3940e-01, 5.9619e-01, -1.4132e+00],
          [-2.8695e-02, 9.6423e-01, 2.6351e-01, -1.0720e+00],
          [-6.1160e-01, -2.2545e-01, 7.5837e-01, 4.7490e+01]],

         [[9.6419e-01, 1.3962e-01, 2.2546e-01, 1.2773e-01],
          [-1.2986e-01, 9.8986e-01, -5.7627e-02, -5.1388e-01],
          [-2.3122e-01, 2.6284e-02, 9.7255e-01, 7.0362e+01]],

         [[7.7300e-01, 4.1197e-01, -4.8244e-01, 3.1492e+00],
          [-3.8902e-01, 9.0852e-01, 1.5250e-01, -1.0950e+00],
          [5.0113e-01, 6.9803e-02, 8.6255e-01, 2.4393e+02]]],
    )


@pytest.fixture
def distortions():
    return torch.tensor(
        [[-21.4957, 0.0000, 0.0000, 0.0000, 0.0000],
         [-14.0726, 0.0000, 0.0000, 0.0000, 0.0000],
         [-1905.7119, 0.0000, 0.0000, 0.0000, 0.0000]],
    )


class TestProjectCameraPairsTo3d:

    def test_basic(self, points_2d, points_3d, intrinsics, extrinsics, distortions):

        p3d = project_camera_pairs_to_3d(
            points=points_2d.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            dist=distortions.unsqueeze(0),
        )

        assert p3d.shape == (1, 3, 2, 3)  # batch, views, keypoints, coords
        assert torch.allclose(p3d, points_3d, rtol=1e-2)

    def test_nan_handling(self, points_2d, points_3d, intrinsics, extrinsics, distortions):

        points_2d[0, 0, 0] = float('nan')
        points_2d[0, 0, 1] = float('nan')
        p3d = project_camera_pairs_to_3d(
            points=points_2d.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            dist=distortions.unsqueeze(0),
        )
        assert p3d.shape == (1, 3, 2, 3)  # batch, view combos, keypoints, coords
        assert torch.all(torch.isnan(p3d[0, 0, 0, :]))
        assert torch.allclose(p3d[0, 0, 1, :], points_3d[0, 0, 1, :], rtol=1e-2)
        assert torch.all(torch.isnan(p3d[0, 1, 0, :]))
        assert torch.allclose(p3d[0, 1, 1, :], points_3d[0, 1, 1, :], rtol=1e-2)
        assert torch.allclose(p3d[0, 2], points_3d[0, 2], rtol=1e-3)


class TestProject3dTo2d:

    def test_basic(self, points_2d, points_3d_single, intrinsics, extrinsics, distortions):

        p2d = project_3d_to_2d(
            points_3d=points_3d_single.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            dist=distortions.unsqueeze(0),
        )

        assert p2d.shape == (1, 3, 2, 2)  # batch, views, keypoints, coords
        assert torch.allclose(p2d, points_2d.unsqueeze(0), rtol=1e-4)

    def test_nan_handling(self, points_2d, points_3d_single, intrinsics, extrinsics, distortions):

        points_3d_with_nan = points_3d_single.clone()
        points_3d_with_nan[0, 0] = float('nan')  # make first keypoint invalid

        p2d_nan = project_3d_to_2d(
            points_3d=points_3d_with_nan.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            dist=distortions.unsqueeze(0),
        )

        assert p2d_nan.shape == (1, 3, 2, 2)  # batch, views, keypoints, coords
        # First keypoint should be NaN in all views
        assert torch.all(torch.isnan(p2d_nan[0, :, 0, :]))
        # Second keypoint should be valid in all views
        assert torch.allclose(p2d_nan[0, :, 1, :], points_2d.unsqueeze(0)[0, :, 1, :], rtol=1e-4)

    def test_all_nan_input(self, points_3d_single, intrinsics, extrinsics, distortions):
        points_3d_all_nan = torch.full_like(points_3d_single, float('nan'))

        p2d_all_nan = project_3d_to_2d(
            points_3d=points_3d_all_nan.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
            dist=distortions.unsqueeze(0),
        )

        assert p2d_all_nan.shape == (1, 3, 2, 2)
        assert torch.all(torch.isnan(p2d_all_nan))

    def test_batch_dim(self, points_2d, points_3d_single, intrinsics, extrinsics, distortions):

        batch_size = 2
        points_3d_batch = points_3d_single.unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics_batch = intrinsics.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        extrinsics_batch = extrinsics.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        distortions_batch = distortions.unsqueeze(0).repeat(batch_size, 1, 1)

        p2d_batch = project_3d_to_2d(
            points_3d=points_3d_batch,
            intrinsics=intrinsics_batch,
            extrinsics=extrinsics_batch,
            dist=distortions_batch,
        )

        assert p2d_batch.shape == (batch_size, 3, 2, 2)
        # Both batch elements should be identical
        assert torch.allclose(p2d_batch[0], p2d_batch[1], rtol=1e-6)


def test_camera_round_trip_accuracy(points_2d, intrinsics, extrinsics, distortions):
    """Test how much error accumulates through the full 2D→3D→2D pipeline."""

    # Start with known good 2D points (from fixtures)
    original_2d = points_2d.unsqueeze(0)  # Add batch dim

    # Go through the full pipeline
    points_3d = project_camera_pairs_to_3d(
        points=original_2d,
        intrinsics=intrinsics.unsqueeze(0),
        extrinsics=extrinsics.unsqueeze(0),
        dist=distortions.unsqueeze(0),
    )

    # Average across camera pairs (like MVT)
    points_3d_mean = torch.mean(points_3d, dim=1)

    # Project back to 2D
    recovered_2d = project_3d_to_2d(
        points_3d=points_3d_mean,
        intrinsics=intrinsics.unsqueeze(0),
        extrinsics=extrinsics.unsqueeze(0),
        dist=distortions.unsqueeze(0),
    )

    # Compare original vs recovered
    error = torch.norm(original_2d - recovered_2d, dim=-1)
    print(f"Round-trip error: {error}")
    print(f"Max error: {error.max():.4f} pixels")

    # This should be very small if pipeline is working correctly
    assert error.max() < 1.0, f"Round-trip error too large: {error.max()}"


def test_full_coordinate_pipeline_roundtrip(points_2d, intrinsics, extrinsics, distortions):
    """Test the complete coordinate transformation pipeline used in training."""

    from lightning_pose.data.utils import convert_bbox_coords, convert_original_to_model_coords

    # Mock batch_dict with realistic values
    batch_size = 1
    num_views = 3
    num_keypoints = 2
    model_height, model_width = 256, 256

    # Create mock bounding boxes (x, y, h, w format)
    # Simulate cropped regions from original image
    bbox = torch.tensor([
        [100, 50, 600, 600],  # view 1: crop from (100,50) with size 600x600
        [200, 100, 500, 500],  # view 2: crop from (200,100) with size 500x500
        [50, 75, 700, 700],  # view 3: crop from (50,75) with size 700x700
    ]).float().reshape(1, -1)  # Shape: (batch=1, 12) for 3 views * 4 coords

    mock_batch_dict = {
        "images": torch.zeros(batch_size, num_views, 3, model_height, model_width),
        "bbox": bbox,
        "intrinsic_matrix": intrinsics.unsqueeze(0),
        "extrinsic_matrix": extrinsics.unsqueeze(0),
        "distortions": distortions.unsqueeze(0),
        "is_multiview": True,
    }

    # Start with world coordinates (your fixture data)
    world_2d_original = points_2d.unsqueeze(
        0)  # Shape: (1, 3, 2, 2) -> (batch=1, num_views=3, num_keypoints=2, coords=2)
    print(f"world_2d_original shape: {world_2d_original.shape}")
    print(f"Original world coords: {world_2d_original[0, :, 0]}")  # First keypoint across views

    # STEP 1: Convert world coords to model coords (reverse of convert_bbox_coords)
    model_2d_coords = convert_original_to_model_coords(
        batch_dict=mock_batch_dict,
        original_keypoints=world_2d_original,
    )  # Output: (batch=1, num_views=3, num_keypoints=2, 2)
    print(f"model_2d_coords shape: {model_2d_coords.shape}")
    print(f"Original model coords: {model_2d_coords[0, :, 0]}")

    # Flatten for pipeline (simulating pred_keypoints format)
    # Need to flatten to (batch, num_targets) where num_targets = num_views * num_keypoints * 2
    pred_keypoints_flat = model_2d_coords.reshape(batch_size, num_views * num_keypoints * 2)
    print(f"Flattened model coords shape: {pred_keypoints_flat.shape}")  # Should be (1, 12)
    print(f"0: Flattened model coords: {pred_keypoints_flat[0, :4]}")

    # STEP 2: Convert model coords back to world coords (convert_bbox_coords)
    pred_keypoints_world = convert_bbox_coords(
        mock_batch_dict, pred_keypoints_flat, in_place=False,
    )
    # Output: (batch, num_targets) -> reshape to (batch, num_views, num_keypoints, 2)
    world_2d_recovered = pred_keypoints_world.reshape(batch_size, num_views, num_keypoints, 2)
    print(f"world_2d_recoveredshape: {world_2d_recovered.shape}")
    print(f"Recovered world coords: {world_2d_recovered[0, :, 0]}")
    print(f"1: Flattened model coords: {pred_keypoints_flat[0, :4]}")

    # Check if we recovered original world coordinates
    world_coord_error = torch.norm(world_2d_original - world_2d_recovered, dim=-1)
    print(f"World coordinate round-trip error: {world_coord_error[0, :, 0]}")
    assert world_coord_error.max() < 0.1, \
        f"World coordinate round-trip failed: {world_coord_error.max()}"

    # STEP 3: Project to 3D (using recovered world coords)
    keypoints_pred_3d = project_camera_pairs_to_3d(
        points=world_2d_recovered,  # (batch, num_views, num_keypoints, 2)
        intrinsics=mock_batch_dict["intrinsic_matrix"],  # (batch, num_views, 3, 3)
        extrinsics=mock_batch_dict["extrinsic_matrix"],  # (batch, num_views, 3, 4)
        dist=mock_batch_dict["distortions"],  # (batch, num_views, num_params)
    )  # Output: (batch, cam_pairs, num_keypoints, 3)
    print(f"3D coords shape: {keypoints_pred_3d.shape}")
    print(f"3D coords sample: {keypoints_pred_3d[0, 0, 0]}")  # First pair, first keypoint
    print(f"2: Flattened model coords: {pred_keypoints_flat[0, :4]}")

    # STEP 4: Average across camera pairs and project back to 2D world coords
    keypoints_3d_mean = torch.mean(keypoints_pred_3d, dim=1)  # (batch, num_keypoints, 3)
    keypoints_pred_2d_reprojected_world = project_3d_to_2d(
        points_3d=keypoints_3d_mean,  # (batch, num_keypoints, 3)
        intrinsics=mock_batch_dict["intrinsic_matrix"],  # (batch, num_views, 3, 3)
        extrinsics=mock_batch_dict["extrinsic_matrix"],  # (batch, num_views, 3, 4)
        dist=mock_batch_dict["distortions"],  # (batch, num_views, num_params)
    )  # Output: (batch, num_views, num_keypoints, 2)
    print(f"Reprojected world coords: {keypoints_pred_2d_reprojected_world[0, :, 0]}")
    print(f"3: Flattened model coords: {pred_keypoints_flat[0, :4]}")

    # STEP 5: Convert back to model coords for heatmap generation
    keypoints_pred_2d_reprojected_model = convert_original_to_model_coords(
        batch_dict=mock_batch_dict,
        original_keypoints=keypoints_pred_2d_reprojected_world,
        # (batch, num_views, num_keypoints, 2)
    )  # Output: (batch, num_views, num_keypoints, 2)
    print(f"4: Flattened model coords: {pred_keypoints_flat[0, :4]}")

    # Flatten to match the pipeline format
    final_model_coords = keypoints_pred_2d_reprojected_model.reshape(
        batch_size, num_views * num_keypoints * 2,
    )
    original_model_coords_flat = model_2d_coords.reshape(batch_size, num_views * num_keypoints * 2)
    print(f"Original model coords (flat): {pred_keypoints_flat[0, :4]}")  # First 2 coordinates
    print(f"Reprojected model coords (flat): {final_model_coords[0, :4]}")  # First 2 coordinates

    # CRITICAL TEST: Compare final model coords with original model coords
    model_coord_error = torch.norm(
        original_model_coords_flat - final_model_coords,
        dim=-1
    )
    print(f"Full pipeline model coordinate error: {model_coord_error[0]:.4f}")

    # This should be small if your pipeline is working correctly
    # But I suspect this will fail, revealing where the coordinate bug is
    assert model_coord_error < 5.0, f"Full pipeline round-trip failed: {model_coord_error.item()}"
