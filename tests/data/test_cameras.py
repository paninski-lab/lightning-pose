import pytest

import torch

from lightning_pose.data.cameras import project_3d_to_2d, project_camera_pairs_to_3d


# --------------------------------------------
# file-level fixtures from anipose-fly example
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
