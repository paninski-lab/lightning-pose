import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, List, Union, Optional
from typeguard import typechecked

# TODO: test that the variable names are good

@typechecked
class PostProcessorCVXPY:
    def __init__(self, keypoints_preds: np.ndarray, pca_param_np: Dict[str, Union[np.ndarray, np.float64]], confidences: Optional[np.ndarray] = None) -> None:
        self.keypoints_preds = keypoints_preds
        self.pca_param_np = pca_param_np
        self.keypoints_preds_2d = keypoints_preds.reshape(-1, 2) # shape (samples* n_keypoints, 2)
        self.x = cp.Variable(self.keypoints_preds_2d.shape)
        if confidences is None:
            self.weights = np.ones(self.keypoints_preds_2d.shape[0])
        else:
            self.weights = confidences.reshape(-1)
        #self.weights = confidences
        #self.weights_flat = self.weights.reshape(-1)
    
    @property
    def orig_shape(self) -> Tuple[int, int]:
        return self.keypoints_preds.shape
    
    @property
    def pca_mean(self) -> np.ndarray:
        return self.pca_param_np["mean"]
    
    @property
    def pca_kept_evecs(self) -> np.ndarray:
        return self.pca_param_np["kept_eigenvectors"]

    @property
    def pca_epsilon(self) -> float:
        return self.pca_param_np["epsilon"]
    
    def reproject_cvxpy(self, x):
        """ equivalent to our utils.pca.reproject() but suitable for cvxpy """
        mean = np.tile(self.pca_mean, (x.shape[0], 1)) # for each sample, repeat mean, to broadcast
        evecs = self.pca_kept_evecs
        # transform data into low-d space as in scikit learn's _BasePCA.transform()
        # https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/decomposition/_base.py#L97
        centered_data = x - mean
        low_d_projection = centered_data @ evecs.T

        # project back up to observation space, as in scikit learn's _BasePCA.inverse_transform()
        # https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/decomposition/_base.py#L125
        reprojection = low_d_projection @ evecs + mean
        return reprojection
    
    def compute_recon_error(self, x, reconstruction):
        # do it like in training
        diff = x - reconstruction
        # flatten it back to a skinny vector using fortran indexing
        diff_flat = cp.reshape(diff.T, shape=(2, self.keypoints_preds_2d.shape[0])).T
        # compute the norm of the difference per keypoint
        recon_err = cp.norm(diff_flat, p=2, axis=1)
        # returns a long and skinny vector with reconstruction errors per keypoint
        return recon_err

    
    def build_pca_constraint(self) -> List[cp.constraints.Inequality]:
        # reshape keypoints (hacking cvxpy's fortran indexing)
        x = cp.reshape(self.x.T, shape=(self.orig_shape[1], self.orig_shape[0])).T
        # transform data into low-d space as in scikit learn's _BasePCA.transform()
        reconstruction = self.reproject_cvxpy(x=x)
        # compute the norm of the difference per keypoint
        recon_err = self.compute_recon_error(x=x, reconstruction=reconstruction)
        # constraint applied per keypoint
        return [recon_err <= self.pca_epsilon]
    
    def build_objective(self):
        # build objective
        norm = cp.norm(self.x - self.keypoints_preds_2d, p=2, axis=1)
        objective = cp.Minimize(cp.sum(cp.multiply(self.weights, norm)))
        return objective
    
    def build_problem(self) -> cp.Problem:
        # build problem
        problem = cp.Problem(self.build_objective(), self.build_pca_constraint())
        return problem