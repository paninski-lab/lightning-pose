import copy
import itertools

import cv2
import numpy as np
import torch
from aniposelib.cameras import CameraGroup as CameraGroupAnipose
from kornia.geometry.calibration import distort_points, undistort_points
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.epipolar import triangulate_points
from torchtyping import TensorType

# to ignore imports for sphix-autoapidoc
__all__ = [
    "project_camera_pairs_to_3d",
    "project_3d_to_2d",
    "CameraGroup",
]


def project_camera_pairs_to_3d(
    points: TensorType["batch", "num_views", "num_keypoints", 2],
    intrinsics: TensorType["batch", "num_views", 3, 3],
    extrinsics: TensorType["batch", "num_views", 3, 4],
    dist: TensorType["batch", "num_views", "num_params"],
) -> TensorType["batch", "cam_pair", "num_keypoints", 3]:
    """Project 2D keypoints from each pair of cameras into 3D world space."""

    num_batch, num_views, num_keypoints, _ = points.shape

    points = undistort_points(
        points=points,
        K=intrinsics,
        dist=dist,
        new_K=torch.eye(3, device=points.device).expand(num_batch, num_views, 3, 3),
    )

    p3d = []
    for j1, j2 in itertools.combinations(range(num_views), 2):

        points1 = points[:, j1, ...]
        points2 = points[:, j2, ...]

        # create a mask for valid keypoints
        # a keypoint is valid if it's not NaN in BOTH views
        valid_mask = ~(
            torch.isnan(points1).any(dim=-1)
            | torch.isnan(points2).any(dim=-1)
        )

        # prepare points for triangulation
        tri = torch.full(
            (num_batch, num_keypoints, 3),
            float('nan'),
            device=points.device,
            dtype=points.dtype,
        )

        # triangulate only valid points
        for batch_idx in range(num_batch):
            # get valid keypoint indices for this batch
            batch_valid_indices = torch.where(valid_mask[batch_idx])[0]

            if len(batch_valid_indices) > 0:
                # extract valid points for this batch
                batch_points1 = points1[batch_idx][valid_mask[batch_idx]]
                batch_points2 = points2[batch_idx][valid_mask[batch_idx]]

                # triangulate valid points
                batch_tri = triangulate_points(
                    P1=extrinsics[batch_idx, j1],
                    P2=extrinsics[batch_idx, j2],
                    points1=batch_points1,
                    points2=batch_points2,
                )

                # place triangulated points back in the full tensor
                tri[batch_idx, valid_mask[batch_idx]] = batch_tri

        p3d.append(tri)

    return torch.stack(p3d, dim=1)


def project_3d_to_2d(
    points_3d: TensorType["batch", "num_keypoints", 3],
    intrinsics: TensorType["batch", "num_views", 3, 3],
    extrinsics: TensorType["batch", "num_views", 3, 4],
    dist: TensorType["batch", "num_views", "num_params"],
) -> TensorType["batch", "num_views", "num_keypoints", 2]:
    """Project 3D keypoints to 2D using camera parameters.

    Fully vectorized and differentiable implementation.

    Args:
        points_3d: 3D points in world coordinates
        intrinsics: Camera intrinsic matrices (3x3)
        extrinsics: Camera extrinsic matrices (3x4)
        dist: Camera distortion parameters

    Returns:
        2D projected points for each camera view
    """
    num_batch, num_keypoints, _ = points_3d.shape
    num_views = intrinsics.shape[1]
    device = points_3d.device
    dtype = points_3d.dtype

    # Convert 3x3 intrinsics to 4x4 format
    K_4x4 = torch.eye(
        4, device=device, dtype=dtype,
    ).unsqueeze(0).unsqueeze(0).repeat(num_batch, num_views, 1, 1)
    K_4x4[:, :, :3, :3] = intrinsics

    # Convert 3x4 extrinsics to 4x4 format
    E_4x4 = torch.eye(
        4, device=device, dtype=dtype,
    ).unsqueeze(0).unsqueeze(0).repeat(num_batch, num_views, 1, 1)
    E_4x4[:, :, :3, :4] = extrinsics

    # Dummy height/width (not used in projection but required by PinholeCamera)
    height = torch.ones(num_batch, device=device, dtype=dtype)
    width = torch.ones(num_batch, device=device, dtype=dtype)

    # Initialize output
    points_2d = torch.full(
        (num_batch, num_views, num_keypoints, 2),
        float('nan'),
        device=device,
        dtype=dtype,
    )

    # Process each view (we can't fully vectorize due to PinholeCamera API limitations)
    for view_idx in range(num_views):
        # Create cameras for all batches for this view
        cameras = PinholeCamera(
            intrinsics=K_4x4[:, view_idx],  # (batch, 4, 4)
            extrinsics=E_4x4[:, view_idx],  # (batch, 4, 4)
            height=height,  # (batch,)
            width=width  # (batch,)
        )

        # Project all 3D points for all batches at once
        # PinholeCamera.project handles batching automatically
        projected_points = cameras.project(points_3d)  # (batch, num_keypoints, 2)

        # Apply distortion for all batches at once
        has_distortion = torch.any(dist[:, view_idx] != 0, dim=-1)  # (batch,)

        if torch.any(has_distortion):
            # Only apply distortion where needed, but in a vectorized way
            distorted_points = distort_points(
                points=projected_points,  # (batch, num_keypoints, 2)
                K=intrinsics[:, view_idx],  # (batch, 3, 3)
                dist=dist[:, view_idx]  # (batch, num_params)
            )

            # Use where to select distorted vs undistorted points
            final_points = torch.where(
                has_distortion.unsqueeze(-1).unsqueeze(-1),  # (batch, 1, 1)
                distorted_points,
                projected_points
            )
        else:
            final_points = projected_points

        # Assign to output tensor
        points_2d[:, view_idx] = final_points

    return points_2d


class CameraGroup(CameraGroupAnipose):
    """Inherit Anipose camera group and add new non-jitted triangulation method for dataloaders."""

    def triangulate_fast(self, points, undistort=True):
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        assert points.shape[0] == len(self.cameras), \
            f"Invalid points shape, first dim should be equal to" \
            f" number of cameras ({len(self.cameras)}), but shape is {points.shape}"

        one_point = False
        if len(points.shape) == 2:
            points = points.reshape(-1, 1, 2)
            one_point = True

        if undistort:
            new_points = np.empty(points.shape)
            for cnum, cam in enumerate(self.cameras):
                # must copy in order to satisfy opencv underneath
                sub = np.copy(points[cnum])
                new_points[cnum] = cam.undistort_points(sub)
            points = new_points

        n_cams, n_points, _ = points.shape

        cam_Rt_mats = np.array([cam.get_extrinsics_mat()[:3] for cam in self.cameras])

        p3d_allview_withnan = []
        for j1, j2 in itertools.combinations(range(n_cams), 2):
            pts1, pts2 = points[j1], points[j2]
            Rt1, Rt2 = cam_Rt_mats[j1], cam_Rt_mats[j2]
            tri = cv2.triangulatePoints(Rt1, Rt2, pts1.T, pts2.T)
            tri = tri[:3] / tri[3]
            p3d_allview_withnan.append(tri.T)
        p3d_allview_withnan = np.array(p3d_allview_withnan)
        out = np.nanmedian(p3d_allview_withnan, axis=0)

        if one_point:
            out = out[0]

        return out

    def copy(self):
        cameras = [cam.copy() for cam in self.cameras]
        metadata = copy.copy(self.metadata)
        return CameraGroup(cameras, metadata)

    def copy_with_new_cameras(self, cameras):
        """Create a new CameraGroup with the same properties but different cameras."""
        new_group = copy.deepcopy(self)
        new_group.cameras = cameras
        return new_group

    @classmethod
    def load(cls, path):
        parent_instance = super().load(path)  # Load using parent class
        return cls(**vars(parent_instance))  # Return as CameraGroup
