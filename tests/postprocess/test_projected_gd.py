import pytest
import torch
from lightning_pose.utils.pca import KeypointPCA
from lightning_pose.postprocess import ProjectedGD


def test_l2_grad_norm_toy_example():
    # not inputing data for now
    proj_gd = ProjectedGD(lr=1e-3, max_iter=1000, tol=1e-3, verbose=False)
    diff = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # norm = 2
    grad = proj_gd.l2_grad(diff, scalar=0.5)
    desired_out = diff * 0.5 * 0.5
    assert torch.allclose(grad, desired_out)
    pass


# def test_l2_grad_norm_computation(cfg, base_data_module_combined):

#     num_train_ims = (
#         len(base_data_module_combined.dataset)
#         * base_data_module_combined.train_probability
#     )
#     num_keypoints = base_data_module_combined.dataset.num_keypoints
#     num_keypoints_both_views = 7

#     # initialize an instance
#     kp_pca = KeypointPCA(
#         loss_type="pca_multiview",
#         error_metric="reprojection_error",
#         data_module=base_data_module_combined,
#         components_to_keep=3,
#         empirical_epsilon_percentile=0.3,
#         mirrored_column_matches=cfg.data.mirrored_column_matches,
#     )
