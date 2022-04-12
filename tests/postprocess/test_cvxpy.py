import torch
from lightning_pose.utils.pca import KeypointPCA
from lightning_pose.postprocess.cvxpy_optim import PostProcessorCVXPY
import numpy as np

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

    # convert pca params to numpy
    pca_param_np = {}
    for p_name, p_val in singleview_pca.parameters.items():
        if isinstance(p_val, torch.Tensor):
            pca_param_np[p_name] = p_val.detach().cpu().numpy()
        else:
            pca_param_np[p_name] = p_val
    
    # reproject the data, so no viloations
    reprojected = singleview_pca.reproject()
    # optimize and see that the output=input because we're at the optimun.
    post_processor = PostProcessorCVXPY(keypoints_preds=reprojected.detach().cpu().numpy(), \
    confidences=np.array([1.]), pca_param_np=pca_param_np)
    prob = post_processor.build_problem()
    val = prob.solve()
    assert(val < 1e-8) # loss is zero
    # assert optimized result == input
    assert np.allclose(post_processor.x.value, post_processor.keypoints_preds_2d)