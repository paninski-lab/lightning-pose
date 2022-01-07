import torch
import numpy as np
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Tuple, Union, Dict, List, Literal

patch_typeguard()  # use before @typechecked


class EmpiricalEpsilon:
    def __init__(self, percentile: float) -> None:
        self.percentile = percentile

    def __call__(self, loss: Union[torch.Tensor, np.array]) -> float:
        """compute the percentile of some loss, to use as an epsilon for epsilon-insensitive loss.

        Args:
            loss (torch.Tensor): tensor with scalar loss per term (e.g., loss per image,
                or loss per keypoint, etc.)

        Returns:
            float: the percentile of the loss which we use as epsilon

        """
        flattened_loss = loss.flatten()  # applies for both np arrays and torch tensors.
        if type(loss) is torch.Tensor:
            flattened_loss = flattened_loss.clone().detach().cpu().numpy()
        return np.nanpercentile(flattened_loss, self.percentile, axis=0)


# TODO: revisit
@typechecked
def convert_dict_entries_to_tensors(
    loss_params: Dict[str, dict],
    device: Union[str, torch.device],
    losses_to_use: List[str],
    to_parameters: bool = False,
) -> Tuple:
    """Set scalars in loss to torch tensors for use with unsupervised losses.

    Args:
        loss_params: dictionary of loss dictionaries, each containing weight, and other
            args.
        losses_to_use: a list of string with names of losses we'll use for training.
            these names will match the keys of loss_params
        device: device to send floats and ints to
        to_parameters: boolean saying whether we make the values into
            torch.nn.Parameter, allowing them to be recognized as by torch.nn.module as
            module.parameters() (and potentially be trained). if False, keep them as
            tensors.

    Returns:
        dict with updated values

    """
    loss_weights_dict = {}  # for parameters that can be represented as a tensor
    loss_params_dict = {}  # for parameters like a tuple or a string which should not be
    for loss, params in loss_params.items():
        if loss in losses_to_use:
            loss_params_dict[loss] = {}
            for key, val in params.items():
                if key == "log_weight":
                    loss_weights_dict[loss] = torch.tensor(
                        val, dtype=torch.float, device=device
                    )
                elif key == "epsilon" and type(val) != torch.Tensor and val is not None:
                    loss_params_dict[loss][key] = torch.tensor(
                        val, dtype=torch.float, device=device
                    )
                else:
                    loss_params_dict[loss][key] = val
    if to_parameters:
        loss_weights_dict = convert_loss_tensors_to_torch_nn_params(loss_weights_dict)
    print("loss weights at the end of convert_dict_entries_to_tensors:")
    print(loss_weights_dict)
    for key, val in loss_weights_dict.items():
        print(key, val)
    print("loss params dict:")
    print(loss_params_dict)

    return loss_weights_dict, loss_params_dict


@typechecked
def convert_loss_tensors_to_torch_nn_params(
    loss_weights: dict,
) -> torch.nn.ParameterDict:
    loss_weights_params = {}
    for loss, weight in loss_weights.items():  # loop over multiple different losses
        print(loss, weight)
        loss_weights_params[loss] = torch.nn.Parameter(weight, requires_grad=True)
    parameter_dict = torch.nn.ParameterDict(loss_weights_params)
    return parameter_dict


@typechecked
def convert_dict_values_to_tensors(
    param_dict: Dict[str, Union[np.array, float]], device: Literal["cpu", "cuda"]
) -> Dict[str, torch.Tensor]:
    # TODO: currently supporting just floats
    for key, val in param_dict.items():
        param_dict[key] = torch.tensor(val, dtype=torch.float, device=device)
    return param_dict
