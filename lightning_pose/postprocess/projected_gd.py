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
        # check for the constraints , if small, do nothing
        # if needed, project the result onto the constraints using the projection parameters
        pass

    def step(
        self, x_curr: TensorType["num_obs", "obs_dim"]
    ) -> TensorType["num_obs", "obs_dim"]:
        x_new = self.grad_step(x_curr)
        x_new = self.project(x_new)  # project the grad onto the constraints
        self.x_list.append(x_new)  # record the new x
        return x_new

