import pytest
import torch
from lightning_pose.utils.pca import KeypointPCA
from lightning_pose.postprocess.projected_gd import ProjectedGD

def test_init():
    proj_gd = ProjectedGD(data=torch.zeros((2,2)), lr=1e-3, max_iter=1000, tol=1e-3, verbose=False)
    assert torch.allclose(proj_gd.optimized_preds, torch.zeros((2,2)))

def test_l2_grad_norm_toy_example():
    # not inputing data for now
    proj_gd = ProjectedGD(data=torch.zeros((2,2)), lr=1e-3, max_iter=1000, tol=1e-3, verbose=False)
    diff = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # norm = 2
    grad = proj_gd.l2_grad(diff, scalar=0.5)
    desired_out = diff * 0.5 * 0.5
    assert torch.allclose(grad, desired_out)

def test_pca_inputs(cfg, base_data_module_combined):
    # initialize an instance
    singleview_pca = KeypointPCA(
        loss_type="pca_singleview",
        error_metric="reprojection_error",
        data_module=base_data_module_combined,
        components_to_keep=4, # something lossy
        empirical_epsilon_percentile=1.0,
        columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
    )
    singleview_pca()  # fit it to have all the parameters

    # give that proj_gd instance the data and proj_params
    proj_gd = ProjectedGD(data=singleview_pca.data_arr.clone(), proj_params={"pca_singleview": singleview_pca}, lr=1e-1, max_iter=100, tol=1e-3, verbose=True)
    reproj = proj_gd.project(proj_gd.optimized_preds)
    assert reproj.shape == proj_gd.optimized_preds.shape
    fitted_preds = proj_gd.fit()
    assert ~torch.isnan(fitted_preds).any()
    assert torch.allclose(fitted_preds, proj_gd.optimized_preds)

def test_norm_calc():
    data = torch.randn((10, 14, 2))
    norm = torch.linalg.norm(data, dim=2, keepdim=True)
    assert norm.shape == (10, 14, 1)





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
