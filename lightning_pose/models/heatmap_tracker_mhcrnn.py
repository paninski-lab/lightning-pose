"""Models that produce heatmaps of keypoints from images."""

from typing import Any, Tuple, Union

import torch
from omegaconf import DictConfig
from torchtyping import TensorType
from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.datatypes import (
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import convert_bbox_coords, undo_affine_transform_batch
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.backbones import ALLOWED_BACKBONES
from lightning_pose.models.base import (
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)
from lightning_pose.models.heads import HeatmapMHCRNNHead

# to ignore imports for sphix-autoapidoc
__all__ = []


class HeatmapTrackerMHCRNN(BaseSupervisedTracker):
    """Multi-headed Convolutional RNN network that handles context frames."""

    def __init__(
        self,
        num_keypoints: int,
        num_targets: int | None = None,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        **kwargs: Any,
    ):
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError("MHCRNN currently only implements downsample_factor=2")

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            del kwargs["do_context"]

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=True,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        if num_targets is None:
            self.num_targets = num_keypoints * 2
        else:
            self.num_targets = num_targets
        self.downsample_factor = downsample_factor

        self.head = HeatmapMHCRNNHead(
            backbone_arch=backbone,
            in_channels=self.num_fc_input_features,
            out_channels=self.num_keypoints,
            downsample_factor=self.downsample_factor,
            upsampling_factor=1 if "vit" in backbone else 2,
        )

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: Union[
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"],
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "view", "frames", "channels":3, "image_height", "image_width"],
            TensorType["batch", "view", "channels":3, "image_height", "image_width"],
        ],
        is_multiview: bool = False,
    ) -> Tuple[
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "frames", "channels":3, "image_height", "image_width"]
          single view, labeled context batch

        - TensorType["batch", "channels":3, "image_height", "image_width"]
          single view, unlabeled batch from DALI

        - TensorType["batch", "view", "frames", "channels":3, "image_height", "image_width"]
          multivew, labeled context batch

        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview, unlabeled batch from DALI

        """

        shape = images.shape

        # get one representation for each frame
        representations = self.get_representations(images, is_multiview=is_multiview)
        # representations shape is (batch, features, height, width, frames)

        # get two heatmaps for each representation (single frame, multi-frame)
        heatmaps_sf, heatmaps_mf = self.head(representations, shape, is_multiview)

        return heatmaps_sf, heatmaps_mf

    def get_loss_inputs_labeled(
        self,
        batch_dict: Union[
            HeatmapLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints_sf = convert_bbox_coords(batch_dict, pred_keypoints_sf)
        pred_keypoints_mf = convert_bbox_coords(batch_dict, pred_keypoints_mf)
        return {
            "heatmaps_targ": torch.cat([batch_dict["heatmaps"], batch_dict["heatmaps"]], dim=0),
            "heatmaps_pred": torch.cat([pred_heatmaps_sf, pred_heatmaps_mf], dim=0),
            "keypoints_targ": torch.cat([target_keypoints, target_keypoints], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_sf, pred_keypoints_mf], dim=0),
            "confidences": torch.cat([confidence_sf, confidence_mf], dim=0),
        }

    def predict_step(
        self,
        batch_dict: Union[
            HeatmapLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
            UnlabeledBatchDict,
        ],
        batch_idx: int,
        return_heatmaps: bool | None = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Predict heatmaps and keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]

        # images -> heatmaps
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(images)

        # heatmaps -> keypoints
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)

        # reshape keypoints to be (batch, n_keypoints, 2)
        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1, 2)
        pred_keypoints_mf = pred_keypoints_mf.reshape(pred_keypoints_mf.shape[0], -1, 2)

        # find higher confidence indices
        mf_conf_gt = torch.gt(confidence_mf, confidence_sf)

        # select higher confidence indices
        pred_keypoints_sf[mf_conf_gt] = pred_keypoints_mf[mf_conf_gt]
        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1)

        confidence_sf[mf_conf_gt] = confidence_mf[mf_conf_gt]

        # bounding box coords -> original image coords
        pred_keypoints_sf = convert_bbox_coords(batch_dict, pred_keypoints_sf)

        if return_heatmaps:
            pred_heatmaps_sf[mf_conf_gt] = pred_heatmaps_mf[mf_conf_gt]
            return pred_keypoints_sf, confidence_sf, pred_heatmaps_sf
        else:
            return pred_keypoints_sf, confidence_sf

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
        ]
        return params


@typechecked
class SemiSupervisedHeatmapTrackerMHCRNN(SemiSupervisedTrackerMixin, HeatmapTrackerMHCRNN):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory | None = None,
        loss_factory_unsupervised: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        **kwargs: Any,
    ):
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            torch_seed=torch_seed,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )
        self.loss_factory_unsup = loss_factory_unsupervised

        # this attribute will be modified by AnnealWeight callback during training
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))
        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(
        self,
        batch_dict: Union[
            UnlabeledBatchDict,
            MultiviewUnlabeledBatchDict,
        ]
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)"""

        # images -> heatmaps
        pred_heatmaps_crnn, pred_heatmaps_sf = self.forward(
            batch_dict["frames"], is_multiview=batch_dict["is_multiview"],
        )

        # heatmaps -> keypoints
        pred_keypoints_crnn, confidence_crnn = self.head.run_subpixelmaxima(pred_heatmaps_crnn)
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)

        # undo augmentations if needed
        pred_keypoints_crnn = undo_affine_transform_batch(
            keypoints_augmented=pred_keypoints_crnn,
            transforms=batch_dict["transforms"],
            is_multiview=batch_dict["is_multiview"],
        )
        pred_keypoints_sf = undo_affine_transform_batch(
            keypoints_augmented=pred_keypoints_sf,
            transforms=batch_dict["transforms"],
            is_multiview=batch_dict["is_multiview"],
        )

        # keypoints -> original image coords keypoints
        pred_keypoints_crnn = convert_bbox_coords(batch_dict, pred_keypoints_crnn)
        pred_keypoints_sf = convert_bbox_coords(batch_dict, pred_keypoints_sf)

        return {
            "heatmaps_pred": torch.cat([pred_heatmaps_crnn, pred_heatmaps_sf], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_crnn, pred_keypoints_sf], dim=0),
            "confidences": torch.cat([confidence_crnn, confidence_sf], dim=0),
        }
