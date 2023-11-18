"""High-level loss class that orchestrates the individual losses."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from torchtyping import TensorType

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.losses.losses import get_loss_classes

# to ignore imports for sphix-autoapidoc
__all__ = [
    "LossFactory",
]


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

        # initialize loss classes
        self._initialize_loss_instances()

    def _initialize_loss_instances(self):
        self.loss_instance_dict = {}
        loss_classes_dict = get_loss_classes()
        for loss, params in self.losses_params_dict.items():
            self.loss_instance_dict[loss] = loss_classes_dict[loss](
                data_module=self.data_module, **params
            )

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
