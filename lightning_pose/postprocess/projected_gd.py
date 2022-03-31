from typing import Dict
import torch
from torchtyping import TensorType
from typing import Dict


class ProjectedGD(object):
    """ projected gradient descent on an L2 ball subject to constraints"""

    def __init__(
        self,
        data: TensorType["num_obs", "obs_dim"] = None,
        proj_params: Dict[str, dict] = None,
        lr: float = 1e-3,
        max_iter: int = 1000,
        tol: float = 1e-3,
        verbose: bool = False,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.data = data
        self.proj_params = proj_params
        self.optimized_preds = torch.nn.parameter.Parameter(data=data.copy_())
        self.x_list = []

    @staticmethod
    def l2_grad(
        diffs: TensorType["num_obs", "obs_dim"], scalar: float = 1.0
    ) -> TensorType["num_obs", "obs_dim"]:
        # TODO: test
        # TODO check it's a proper L2 norm with a square root
        grad = diffs * scalar * (1.0 / torch.linalg.norm(diffs, dim=1, keepdim=True))
        return grad

    def fit(self, data: TensorType["num_obs", "obs_dim"] = None):
        pass

    def grad_step(
        self, x_curr: TensorType["num_obs", "obs_dim"]
    ) -> TensorType["num_obs", "obs_dim"]:
        # TODO: check dims of x_curr and self.data
        return x_curr - self.lr * self.l2_grad(x_curr - self.data)

    def project(
        self, x_after_step: TensorType["num_obs", "obs_dim"]
    ) -> TensorType["num_obs", "obs_dim"]:
        # PCA rationale:
        # check for the constraints , if small, do nothing
        # if needed, project the result onto the constraints using the projection parameters
        # pca_reproject(x_after_step, self.proj_params) to go back to plausible values (if epsilon = 0)
        # if epsilon non-zero. project onto all evecs (discarded and kept). these are all orthogonal.
        # you go one by one. if your in the kept eigenvecs, do nothing. if you're in discarded evecs, you're outside constraint space
        # you have a sclar proj per dim. so in those held out dims, you manually set the projs to be epsilon instead of that high projection that you may encounter.
        # you modify all the projections onto the discarded evecs. you have a vector which is num_obs x num_evecs. this is the representation of data in the PCA coordinate basis
        # then, you modify that representation, and you send it back to the original space using the transpose of the evecs.
        # Temporal rationale: want x_t - x_t-1 to be small. compute the difference per timepoint. choose one direction, say forward. you have two points in
        # you have 2 points in 2d space. the difference vector is the direction. compute the norm. if norm > epsilon, rescale it so norm is equal to epsilon. diff/epsilon -- now you have a direction and a step size. you define x_t += x_t-1 + diff/epsilon.
        # the next time point has to be inside a ball with radius epsilon. if it's outside, you project onto the exterior of that ball. if it's inside, keep it where it is.
        # the result will be different if you start from the end or from the beggining.
        pass

    def step(
        self, x_curr: TensorType["num_obs", "obs_dim"]
    ) -> TensorType["num_obs", "obs_dim"]:
        x_new = self.grad_step(x_curr)
        x_new = self.project(x_new)  # project the grad onto the constraints
        self.x_list.append(x_new)  # record the new x
        return x_new

