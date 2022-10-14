"""High-level loss class that orchestrates the individual losses."""

import pytorch_lightning as pl
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.losses.losses import get_loss_classes

patch_typeguard()  # use before @typechecked


class LossFactory(pl.LightningModule):
    """Factory object that contains an object for each specified loss."""

    @typechecked
    def __init__(
        self,
        losses_params_dict: Dict[str, dict],
        data_module: Union[BaseDataModule, UnlabeledDataModule],
        learn_weights: bool = False,
    ) -> None:

        super().__init__()
        self.losses_params_dict = losses_params_dict
        self.data_module = data_module
        self.learn_weights = learn_weights

        # initialize loss classes
        self._initialize_loss_instances()

        # turn weights to parameters if we want to learn them
        self._initialize_weight_parameter_dict()

    def _initialize_loss_instances(self):
        self.loss_instance_dict = {}
        loss_classes_dict = get_loss_classes()
        for loss, params in self.losses_params_dict.items():
            self.loss_instance_dict[loss] = loss_classes_dict[loss](
                data_module=self.data_module, **params)

    def _initialize_weight_parameter_dict(self):
        if self.learn_weights:
            loss_weights_dict = {}
            for loss, loss_instance in self.loss_instance_dict.items():
                loss_weights_dict[loss] = torch.nn.Parameter(
                    loss_instance.log_weight, requires_grad=True
                )
                # update the loss instance to have a Parameter instead of Tensor
                # TODO: is it ok to do this before initing ParameterDict?
                loss_instance.log_weight = loss_weights_dict[loss]
            self.loss_weights_parameter_dict = torch.nn.ParameterDict(loss_weights_dict)
        else:
            # log_weights are already Tensors, no need to do anything
            # no parameter module optimized
            self.loss_weights_parameter_dict = {}

    @typechecked
    def __call__(
        self,
        stage: Optional[Literal["train", "val", "test"]] = None,
        anneal_weight: Union[float, torch.Tensor] = 1.0,
        **kwargs
    ) -> Tuple[TensorType[()], List[dict]]:

        # loop over losses, compute, sum, log
        # don't log if stage is None
        tot_loss = 0.0
        log_list_all = []
        for loss_name, loss_instance in self.loss_instance_dict.items():

            # kwargs options:
            # - heatmaps_targ
            # - heatmaps_pred
            # - keypoints_targ
            # - keypoints_pred
            #
            # if a Loss class needs to manipulate other objects (e.g. image embedding),
            # the model's `training_step` method must supply that tensor to the loss
            # factory using the correct keyword argument (defined by the new Loss
            # class's `__call__` method)

            # "stage" is used for logging purposes
            curr_loss, log_list = loss_instance(stage=stage, **kwargs)
            current_weighted_loss = loss_instance.weight * curr_loss
            tot_loss += anneal_weight * current_weighted_loss

            # if learning the weights in front of each loss term:
            if self.learn_weights:
                # penalize for the magnitude of the weights:
                # \log(\sigma_i) for each weight i
                # \log(\sigma_1 * \sigma_2 * ...) = \log(\sigma_1) + \log(\sigma_2) + ..
                # 0.5 because log_weight is actually \log(\sigma^2) = 2 * \log(\sigma)
                tot_loss += anneal_weight * 0.5 * loss_instance.log_weight

            # log weighted losses (unweighted losses auto-logged by loss instance)
            log_list += [
                {
                    "name": "%s_%s_loss_weighted" % (stage, loss_name),
                    "value": current_weighted_loss,
                }
            ]

            # append all losses
            log_list_all += log_list

        return tot_loss, log_list_all
