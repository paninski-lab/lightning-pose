import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torch.nn import functional as F

patch_typeguard()  # use before @typechecked

# TODO: this won't work unless the inputs are right, not implemented yet.
@typechecked
# what are we doing about NANS?
def MultiviewPCALoss(
    # TODO: y_hat should be already reshaped? if so, change below
    reshaped_maxima_preds: TensorType["Batch_Size", "Num_Keypoints", 2, float],
    discarded_evecs: TensorType["Views_Times_Two", "Num_Discarded_Evecs", float],
    epsilon: TensorType[float],
) -> TensorType[float]:
    """assume that we have keypoints after find_subpixel_maxima
    and that we have discarded confidence here, and that keypoints were reshaped"""
    # # TODO: add conditions regarding epsilon?
    # kernel_size = np.min(self.output_shape)  # change from numpy to torch
    # kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
    # keypoints = find_subpixel_maxima(
    #     y_hat.detach(),  # TODO: why detach? could keep everything on GPU?
    #     torch.tensor(kernel_size, device=self.device),
    #     torch.tensor(self.output_sigma, device=self.device),
    #     self.upsample_factor,  # TODO: these are coming from self, shouldn't be inputs?
    #     self.coordinate_scale,
    #     self.confidence_scale,
    # )
    # keypoints = keypoints[:, :, :2]
    # data_arr = format_mouse_data(keypoints)
    # TODO: consider avoiding the transposes
    abs_proj_discarded = torch.abs(
        torch.matmul(reshaped_maxima_preds.T, discarded_evecs.T)
    )
    epsilon_masked_proj = abs_proj_discarded.masked_fill(
        mask=abs_proj_discarded > epsilon, value=0.0
    )
    assert (epsilon_masked_proj >= 0.0).all()  # every element should be positive
    assert torch.mean(epsilon_masked_proj) <= torch.mean(
        abs_proj_discarded
    )  # the scalar loss should be smaller after zeroing out elements.
    return torch.mean(epsilon_masked_proj)
