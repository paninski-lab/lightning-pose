"""High-level loss object that orchestrates the individual losses."""

import pytorch_lightning as pl
import torch
from typing import Any, Dict, List, Union, Callable

from pose_est_nets.datasets.datamodules import BaseDataModule, UnlabeledDataModule
from pose_est_nets.losses.helpers import convert_dict_entries_to_tensors
from pose_est_nets.losses.losses import get_loss_classes


class LossFactory(pl.LightningModule):
    """Factory object that contains an object for each specified loss."""

    def __init__(
        self,
        losses_params_dict: Dict[str, dict],
        data_module: Union[BaseDataModule, UnlabeledDataModule],
    ) -> None:
        super().__init__()
        self.losses_params_dict = losses_params_dict
        self.data_module = data_module

        # where to call `convert_dict_entries_to_tensors`?

        # initialize loss classes
        self.loss_instance_dict = {}
        loss_classes_dict = get_loss_classes()
        for loss, params in self.losses_params_dict.items():
            self.loss_instance_dict[loss] = loss_classes_dict[loss](
                data_module=self.data_module, **params
            )  # some losses need the data_module to compute parameters at init time

    # NOTE: this will reinitialize the loss classes on every call to __call__; expensive
    # for e.g. pca losses
    # @property
    # def loss_instance_dict(self) -> Dict[str, Callable]:
    #     loss_instance_dict = {}
    #     loss_classes_dict = get_loss_classes()
    #     for loss, params in self.losses_params_dict.items():
    #         loss_instance_dict[loss] = loss_classes_dict[loss](
    #             data_module=self.data_module, **params
    #         )  # some losses need the data_module to compute parameters at init time
    #     return loss_instance_dict

    def __call__(self, stage=None, anneal_weight=1.0, **kwargs):
        # loop over losses, compute, sum, log
        # loop over unsupervised losses
        # don't log if stage is None
        # Question -- should we include the supervised loss?
        tot_loss = 0.0
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

        return tot_loss
