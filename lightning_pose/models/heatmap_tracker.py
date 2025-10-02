"""Models that produce heatmaps of keypoints from images."""

from typing import Any, Tuple

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
from lightning_pose.models.heads import HeatmapHead

# to ignore imports for sphix-autoapidoc
__all__ = []


class HeatmapTracker(BaseSupervisedTracker):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        num_targets: int | None = None,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a heatmap-based pose estimation model with conv or transformer backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            pretrained: True to load pretrained imagenet weights
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        if num_targets is None:
            self.num_targets = num_keypoints * 2
        else:
            self.num_targets = num_targets
        self.downsample_factor = downsample_factor

        self.head = HeatmapHead(
            backbone_arch=backbone,
            in_channels=self.num_fc_input_features,
            out_channels=self.num_keypoints,
            downsample_factor=self.downsample_factor,
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
        images: (
            TensorType["batch", "channels":3, "image_height", "image_width"]
            | TensorType["batch", "views", "channels":3, "image_height", "image_width"]
        ),
    ) -> TensorType[
        "num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"
    ]:
        """Forward pass through the network."""
        # we get one representation for each desired output.
        shape = images.shape

        # if len(shape) > 4 we assume we have multiple views and need to combine images across
        # batch/views before passing to network, then we reshape
        if len(shape) > 4:
            images = images.reshape(-1, shape[-3], shape[-2], shape[-1])
            # images = [views * batch, channels, image_height, image_width]
            representations = self.get_representations(images)
            # representations = [views * batch, num_features, rep_height, rep_width]
            heatmaps = self.head(representations)
            # heatmaps = [views * batch, num_keypoints, heatmap_height, heatmap_width]
            heatmaps = heatmaps.reshape(shape[0], -1, heatmaps.shape[-2], heatmaps.shape[-1])
            # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]
        else:
            representations = self.get_representations(images)
            heatmaps = self.head(representations)

        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: HeatmapLabeledBatchDict | MultiviewHeatmapLabeledBatchDict
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.head.run_subpixelmaxima(predicted_heatmaps)
        # bounding box coords -> original image coords
        predicted_keypoints = convert_bbox_coords(batch_dict, predicted_keypoints)
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }

    def predict_step(
        self,
        batch_dict: (
            HeatmapLabeledBatchDict
            | MultiviewHeatmapLabeledBatchDict
            | UnlabeledBatchDict
            | MultiviewUnlabeledBatchDict
        ),
        batch_idx: int,
        return_heatmaps: bool | None = False,
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
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
        predicted_heatmaps = self.forward(images)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.head.run_subpixelmaxima(predicted_heatmaps)
        # bounding box coords -> original image coords
        predicted_keypoints = convert_bbox_coords(batch_dict, predicted_keypoints)
        if return_heatmaps:
            return predicted_keypoints, confidence, predicted_heatmaps
        else:
            return predicted_keypoints, confidence

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "lr": 0, "name": "backbone"},
            {"params": self.head.parameters(), "name": "head"},
        ]
        return params


@typechecked
class SemiSupervisedHeatmapTracker(SemiSupervisedTrackerMixin, HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory | None = None,
        loss_factory_unsupervised: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        **kwargs: Any,
    ) -> None:
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
        batch_dict: UnlabeledBatchDict | MultiviewUnlabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict["frames"])
        # heatmaps -> keypoints
        pred_keypoints_augmented, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # undo augmentation if needed
        pred_keypoints = undo_affine_transform_batch(
            keypoints_augmented=pred_keypoints_augmented,
            transforms=batch_dict["transforms"],
            is_multiview=batch_dict["is_multiview"],
        )
        # keypoints -> original image coords keypoints
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        return {
            "heatmaps_pred": pred_heatmaps,  # if augmented, augmented heatmaps
            "keypoints_pred": pred_keypoints,  # if augmented, original keypoints
            "keypoints_pred_augmented": pred_keypoints_augmented,  # match pred_heatmaps
            "confidences": confidence,
        }
