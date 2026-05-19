"""High-level loss class that orchestrates the individual losses."""

from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import torch
from jaxtyping import Float
from omegaconf import DictConfig, ListConfig, OmegaConf

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.losses.losses import (
    HeatmapJSLoss,
    HeatmapKLLoss,
    HeatmapMSELoss,
    Loss,
    PairwiseProjectionsLoss,
    PCALoss,
    RegressionMSELoss,
    ReprojectionHeatmapLoss,
    TemporalHeatmapLoss,
    TemporalLoss,
    UnimodalLoss,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    'LossFactory',
    'get_loss_classes',
    'get_loss_factories',
]


def get_loss_classes() -> dict[str, type[Loss]]:
    """Return a mapping from loss name string to loss class.

    Returns:
        dict mapping each registered loss name to its class.
    """
    return {
        RegressionMSELoss.loss_name: RegressionMSELoss,
        HeatmapMSELoss.loss_name: HeatmapMSELoss,
        HeatmapKLLoss.loss_name: HeatmapKLLoss,
        HeatmapJSLoss.loss_name: HeatmapJSLoss,
        PCALoss.LOSS_NAME_MULTIVIEW: PCALoss,
        PCALoss.LOSS_NAME_SINGLEVIEW: PCALoss,
        TemporalLoss.loss_name: TemporalLoss,
        TemporalHeatmapLoss.LOSS_NAME_MSE: TemporalHeatmapLoss,
        TemporalHeatmapLoss.LOSS_NAME_KL: TemporalHeatmapLoss,
        UnimodalLoss.LOSS_NAME_MSE: UnimodalLoss,
        UnimodalLoss.LOSS_NAME_KL: UnimodalLoss,
        UnimodalLoss.LOSS_NAME_JS: UnimodalLoss,
        PairwiseProjectionsLoss.loss_name: PairwiseProjectionsLoss,
        ReprojectionHeatmapLoss.loss_name: ReprojectionHeatmapLoss,
    }


def get_loss_factories(
    cfg: DictConfig | ListConfig,
    data_module: BaseDataModule | UnlabeledDataModule,
) -> dict:
    """Create supervised and unsupervised loss factories from a hydra config.

    Args:
        cfg: hydra config carrying model, data, and loss parameters.
        data_module: data module passed to data-dependent losses such as PCA.

    Returns:
        dict with keys ``'supervised'`` and ``'unsupervised'``, each mapping to a
        :class:`LossFactory` instance.
    """
    cfg_loss_dict = OmegaConf.to_object(cfg.losses)
    assert cfg_loss_dict is not None

    loss_params_dict: dict[str, dict] = {'supervised': {}, 'unsupervised': {}}

    # collect supervised losses; log_weight=0.0 → effective weight = 1/2
    if cfg.model.model_type.find('heatmap') > -1:
        loss_name = 'heatmap_' + cfg.model.heatmap_loss_type
        loss_params_dict['supervised'][loss_name] = {'log_weight': 0.0}
        if cfg.model.model_type.find('multiview') > -1 and cfg.data.get('camera_params_file'):

            log_weight_sp = cfg.losses.get(
                'supervised_pairwise_projections', {}
            ).get('log_weight')
            if log_weight_sp is not None:
                print('Adding supervised pairwise projection loss')
                loss_params_dict['supervised']['supervised_pairwise_projections'] = {
                    'log_weight': log_weight_sp
                }

            log_weight_hr = cfg.losses.get(
                'supervised_reprojection_heatmap_mse', {}
            ).get('log_weight')
            if log_weight_hr is not None:
                print('Adding supervised reprojection heatmap loss')
                height_og = cfg.data.image_resize_dims.height
                width_og = cfg.data.image_resize_dims.width
                height_ds = int(height_og // (2 ** cfg.data.get('downsample_factor', 2)))
                width_ds = int(width_og // (2 ** cfg.data.get('downsample_factor', 2)))
                loss_params_dict['supervised']['supervised_reprojection_heatmap_mse'] = {
                    'log_weight': log_weight_hr,
                    'original_image_height': height_og,
                    'original_image_width': width_og,
                    'downsampled_image_height': height_ds,
                    'downsampled_image_width': width_ds,
                }

    else:
        loss_params_dict['supervised'][cfg.model.model_type] = {'log_weight': 0.0}

    # collect unsupervised losses and their params
    if cfg.model.losses_to_use is not None:
        for loss_name in cfg.model.losses_to_use:
            loss_params_dict['unsupervised'][loss_name] = cfg_loss_dict[loss_name]
            loss_params_dict['unsupervised'][loss_name]['loss_name'] = loss_name
            if loss_name[:8] == 'unimodal' or loss_name[:16] == 'temporal_heatmap':
                if cfg.model.model_type == 'regression':
                    raise NotImplementedError(
                        'unimodal loss can only be used with classes inheriting from '
                        'HeatmapTracker. \nYou specified a RegressionTracker model.'
                    )
                raise Exception(
                    'need to update unimodal and/or temporal heatmap loss to not use '
                    'cfg.data.image_resize_dims, which has been deprecated.'
                )
                height_og = cfg.data.image_resize_dims.height
                width_og = cfg.data.image_resize_dims.width
                loss_params_dict['unsupervised'][loss_name]['original_image_height'] = height_og
                loss_params_dict['unsupervised'][loss_name]['original_image_width'] = width_og
                height_ds = int(height_og // (2 ** cfg.data.get('downsample_factor', 2)))
                width_ds = int(width_og // (2 ** cfg.data.get('downsample_factor', 2)))
                loss_params_dict['unsupervised'][loss_name]['downsampled_image_height'] = height_ds
                loss_params_dict['unsupervised'][loss_name]['downsampled_image_width'] = width_ds
                if loss_name[:8] == 'unimodal':
                    loss_params_dict['unsupervised'][loss_name]['uniform_heatmaps'] = (
                        cfg.training.get('uniform_heatmaps_for_nan_keypoints', False)
                    )
            elif loss_name == 'pca_multiview':
                if cfg.data.get('view_names', None) and len(cfg.data.view_names) > 1:
                    num_keypoints = cfg.data.num_keypoints
                    num_views = len(cfg.data.view_names)
                    if isinstance(cfg.data.mirrored_column_matches[0], int):
                        loss_params_dict['unsupervised'][loss_name][
                            'mirrored_column_matches'
                        ] = [
                            (
                                v * num_keypoints
                                + np.array(cfg.data.mirrored_column_matches, dtype=int)
                            ).tolist()
                            for v in range(num_views)
                        ]
                    else:
                        loss_params_dict['unsupervised'][loss_name][
                            'mirrored_column_matches'
                        ] = cfg.data.mirrored_column_matches
                else:
                    loss_params_dict['unsupervised'][loss_name][
                        'mirrored_column_matches'
                    ] = cfg.data.mirrored_column_matches
            elif loss_name == 'pca_singleview':
                if cfg.data.get('view_names', None) and len(cfg.data.view_names) > 1:
                    raise NotImplementedError(
                        'The Pose PCA loss is currently not implemented for multiview data.'
                    )
                else:
                    loss_params_dict['unsupervised'][loss_name][
                        'columns_for_singleview_pca'
                    ] = cfg.data.get('columns_for_singleview_pca', None)

    loss_factory_sup = LossFactory(
        losses_params_dict=loss_params_dict['supervised'],
        data_module=data_module,
    )
    loss_factory_unsup = LossFactory(
        losses_params_dict=loss_params_dict['unsupervised'],
        data_module=data_module,
    )

    return {'supervised': loss_factory_sup, 'unsupervised': loss_factory_unsup}


class LossFactory(pl.LightningModule):
    """Factory object that contains an object for each specified loss."""

    def __init__(
        self,
        losses_params_dict: dict[str, dict],
        data_module: BaseDataModule | UnlabeledDataModule | None,
    ) -> None:
        """Initialize LossFactory and create all specified loss instances.

        Args:
            losses_params_dict: mapping from loss name to a dict of keyword arguments that will
                be passed to the corresponding loss class constructor.
            data_module: data module passed to each loss; required for data-dependent losses such
                as PCA.
        """
        super().__init__()
        self.losses_params_dict = losses_params_dict
        self.data_module = data_module

        # initialize loss classes
        self._initialize_loss_instances()

    def _initialize_loss_instances(self) -> None:
        """Instantiate each loss class from ``self.losses_params_dict`` and store them."""
        self.loss_instance_dict = {}
        loss_classes_dict = get_loss_classes()
        for loss, params in self.losses_params_dict.items():
            self.loss_instance_dict[loss] = loss_classes_dict[loss](
                data_module=self.data_module, **params
            )

    def __call__(
        self,
        stage: Literal['train', 'val', 'test'] | None = None,
        anneal_weight: float | torch.Tensor | None = 1.0,
        **kwargs: Any,
    ) -> tuple[Float[torch.Tensor, ''], list[dict]]:
        """Compute the total weighted loss and collect logging entries for all registered losses.

        Args:
            stage: training stage used for loss logging (``'train'``, ``'val'``, ``'test'``);
                pass ``None`` to suppress logging.
            anneal_weight: scalar multiplier applied to all non-heatmap losses; typically the
                output of an ``AnnealWeight`` callback.
            **kwargs: tensors forwarded to each individual loss (e.g., ``heatmaps_targ``,
                ``keypoints_pred``).

        Returns:
            Tuple of:
                - scalar total loss tensor.
                - list of logging dicts with ``'name'`` and ``'value'`` keys.
        """
        tot_loss: Float[torch.Tensor, ''] = torch.tensor(0.0)
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

            curr_loss, log_list = loss_instance(stage=stage, **kwargs)
            current_weighted_loss = loss_instance.weight * curr_loss
            if anneal_weight is None or loss_name in ['heatmap_mse', 'heatmap_kl', 'heatmap_js']:
                anneal_weight_ = 1.0
            else:
                anneal_weight_ = anneal_weight
            scaled = anneal_weight_ * current_weighted_loss
            # move accumulator to loss device on first iteration (losses run on GPU at train time)
            tot_loss = tot_loss.to(scaled.device) + scaled

            # log weighted losses (unweighted losses auto-logged by loss instance)
            log_list += [
                {
                    'name': f'{stage}_{loss_name}_loss_weighted',
                    'value': current_weighted_loss,
                }
            ]

            log_list_all += log_list

        return tot_loss, log_list_all
