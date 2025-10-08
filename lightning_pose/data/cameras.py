import copy
import itertools

import cv2
import numpy as np
import torch
from aniposelib.cameras import CameraGroup as CameraGroupAnipose
from kornia.geometry.calibration import undistort_points
from kornia.geometry.epipolar import triangulate_points
from torchtyping import TensorType

# to ignore imports for sphix-autoapidoc
__all__ = [
    "project_camera_pairs_to_3d",
    "get_valid_projection_masks",
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
                torch.isnan(points1).any(dim=-1) |
                torch.isnan(points2).any(dim=-1)
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


def get_valid_projection_masks(
    points: TensorType["batch", "num_views", "num_keypoints", 2]
) -> TensorType["batch", "cam_pair", "num_keypoints"]:

    num_batch, num_views, num_keypoints, _ = points.shape

    m3d = []
    for j1, j2 in itertools.combinations(range(num_views), 2):
        points1 = points[:, j1, :, 0]
        points2 = points[:, j2, :, 0]
        m3d.append(~torch.isnan(points1 + points2))
    return torch.stack(m3d, dim=1)


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
