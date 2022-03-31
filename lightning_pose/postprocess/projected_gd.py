from typing import Dict
import torch
from torchtyping import TensorType
from typing import Dict
from tqdm import tqdm


class ProjectedGD(object):
    """ projected gradient descent on an L2 ball subject to constraints"""

    def __init__(
        self,
        data: TensorType["num_obs", "obs_dim"] = None,
        proj_params: Dict[str, dict] = None,
        lr: float = 1e-3,
        max_iter: int = 1000,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.data = data
        self.proj_params = proj_params
        self.optimized_preds = data.detach().clone() # + torch.randn_like(data)*1e-4 # torch.nn.parameter.Parameter(data=data.detach().clone())
        self.x_list = []

    @staticmethod
    def l2_grad(
        diffs: TensorType["num_obs", "obs_dim"], scalar: float = 1.0
    ) -> TensorType["num_obs", "obs_dim"]:
        # TODO: test
        if torch.allclose(diffs, torch.zeros_like(diffs)):
            print("diffs are all zeros")
            # don't divide by zero
            return diffs
        else:
            grad = diffs * scalar * (1.0 / torch.linalg.norm(diffs))
            return grad

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
        return self.proj_params["pca_singleview"].reproject(x_after_step)
        # Temporal rationale: want x_t - x_t-1 to be small. compute the difference per timepoint. choose one direction, say forward. you have two points in
        # you have 2 points in 2d space. the difference vector is the direction. compute the norm. if norm > epsilon, rescale it so norm is equal to epsilon. diff/epsilon -- now you have a direction and a step size. you define x_t += x_t-1 + diff/epsilon.
        # the next time point has to be inside a ball with radius epsilon. if it's outside, you project onto the exterior of that ball. if it's inside, keep it where it is.
        # the result will be different if you start from the end or from the beggining.

    def step(
        self, x_curr: TensorType["num_obs", "obs_dim"]
    ) -> TensorType["num_obs", "obs_dim"]:
        x_after_step = self.grad_step(x_curr=x_curr) # gradient descent on the l2 norm objective
        print("x_after_step:", x_after_step)
        x_after_projection = self.project(x_after_step=x_after_step)  # project the current x onto the constraints, get plausible x
        print("x_after_proj:", x_after_projection)
        return x_after_projection
    
    def fit(self) -> TensorType["num_obs", "obs_dim"]:
        # TODO: test that the recurrence is fine
        x_curr = self.optimized_preds.clone()
        for i in tqdm(range(self.max_iter)):
            x_new = self.step(x_curr)
            if self.verbose:
                print(f"iteration {i}")
                print(f"x_curr: {x_curr}")
                print(f"x_new: {x_new}")
            if torch.allclose(x_curr, x_new, atol=self.tol):
                break
            x_curr = x_new.clone()
            self.x_list.append(x_new)  # record the new x
            # TODO: lr_decay
        self.optimized_preds = x_new
        return self.optimized_preds

