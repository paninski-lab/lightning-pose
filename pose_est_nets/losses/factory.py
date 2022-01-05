# high-level class that orchestrates the individual losses
# individual loss classes -> losses.py
# a base loss parent class -> losses.py
from typing import Any, Dict, List, Union, Callable

import pytorch_lightning as pl
from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule

from pose_est_nets.losses.losses import get_loss_classes, get_losses_dict
import torch


class LossFactory(pl.LightningModule):
    def __init__(
        self,
        losses_params_dict: Dict[str, dict],
        data_module: Union[BaseDataModule, UnlabeledDataModule],
    ) -> None:
        super().__init__()
        self.losses_params_dict = losses_params_dict  # that dictionary is filtered before the LossFactory. Only the relevant losses
        self.data_module = data_module

    @property
    def loss_instance_dict(self) -> Dict[str, Callable]:
        loss_instance_dict = {}
        loss_classes_dict = get_loss_classes()
        for loss, params in self.losses_params_dict.items():  # assumes loss
            loss_instance_dict[loss] = loss_classes_dict[loss](
                data_module=self.data_module, **params
            )  # some losses may need the data_module to compute parameters at initialization time
        return loss_instance_dict

    def __call__(self, **kwargs):
        # loop over losses, compute, sum, log
        # loop over unsupervised losses
        # Question -- should we include the supervised loss?
        tot_loss = 0.0
        tot_loss += supervised_loss
        for loss_name, loss_instance in self.loss_instance_dict.items():
            # Some losses use keypoint_preds, some use heatmap_preds, and some use both.
            # all have **kwargs so are robust to unneeded inputs.

            unsupervised_loss = loss_instance(
                **kwargs
            )  # loss_instance already has all the parameters

            # TODO: eliminate, loss_instance could have this loss_weight as a @property
            loss_weight = (
                1.0
                / (  # weight = \sigma where our trainable parameter is \log(\sigma^2). i.e., we take the parameter as it is in the config and exponentiate it to enforce positivity
                    2.0 * torch.exp(loss_instance.weight)
                )
            )

            current_weighted_loss = loss_weight * unsupervised_loss
            tot_loss += self.total_unsupervised_importance * current_weighted_loss

            if (
                self.learn_weights == True
            ):  # penalize for the magnitude of the weights: \log(\sigma_i) for each weight i
                tot_loss += self.total_unsupervised_importance * (
                    0.5 * self.loss_instance.weight
                )  # recall that \log(\sigma_1 * \sigma_2 * ...) = \log(\sigma_1) + \log(\sigma_2) + ...

            # log individual unsupervised losses
            self.log(loss_name + "_loss", unsupervised_loss, prog_bar=True)
            self.log(
                "weighted_" + loss_name + "_loss", current_weighted_loss, prog_bar=True
            )
            self.log(
                "{}_{}".format(loss_name, "weight"),
                loss_weight,
                prog_bar=True,
            )

        # log other losses
        self.log("total_loss", tot_loss, prog_bar=True)
        self.log("supervised_loss", supervised_loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)


def get_loss(loss: str, params: dict) -> Callable:
    # initialize the right loss class
    pass
