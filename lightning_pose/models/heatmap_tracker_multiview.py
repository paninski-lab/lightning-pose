"""Models that produce heatmaps of keypoints from images on multiview datasets."""

from typing import Any, Tuple

import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torchtyping import TensorType
# from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.datatypes import (
    MultiviewHeatmapLabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.cameras import get_valid_projection_masks, project_camera_pairs_to_3d
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
    convert_bbox_coords,
)
from lightning_pose.models.heads import (
    ALLOWED_MULTIVIEW_HEADS,
    ALLOWED_MULTIVIEW_MULTIHEADS,
    MultiviewFeatureTransformerHead,
    MultiviewFeatureTransformerHeadLearnable, # learned view embeddings
    MultiviewHeatmapCNNHead,
    MultiviewHeatmapCNNMultiHead,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapTrackerMultiview",
    "HeatmapTrackerMultiviewMultihead",
]


class HeatmapTrackerMultiview(BaseSupervisedTracker):
    """Convolutional network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        head: ALLOWED_MULTIVIEW_MULTIHEADS = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            head: architecture used to fuse view information
                - heatmap_cnn / feature_transformer
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError(
                "HeatmapTrackerMultiviewHeatmapCNN currently only implements downsample_factor=2"
            )

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
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        if head == "heatmap_cnn":
            self.head = MultiviewHeatmapCNNHead(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        elif head == "feature_transformer":
            self.head = MultiviewFeatureTransformerHead(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                transformer_d_model=512,
                transformer_nhead=8, # original 8
                transformer_dim_feedforward=512,
                transformer_num_layers=1,
                img_size=image_size,
            )
        elif head == "feature_transformer_learnable":
            # Use the new head with learnable view embeddings
            self.head = MultiviewFeatureTransformerHeadLearnable(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                transformer_d_model=512,
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=2,
                img_size=image_size,
            )
        else:
            raise NotImplementedError(
                f"{head} is not a valid multiview head, choose from {ALLOWED_MULTIVIEW_HEADS}"
            )

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # this attribute will be modified by AnnealWeight callback during training and can be
        # used to weight supervised, non-heatmap losses
        self.total_unsupervised_importance = torch.tensor(1.0)

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: TensorType["batch", "view", "channels":3, "image_height", "image_width"],
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        batch_size, num_views, channels, img_height, img_width = images.shape

        # stack batch and view into first dim to get representations
        images = images.reshape(-1, channels, img_height, img_width) 
        representations = self.get_representations(images)
        # representations shape is (view * batch, num_features, rep_height, rep_width)

        # get heatmaps for each representation
        heatmaps = self.head(representations, num_views)

        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        # project predictions from pairs of views into 3d if calibration data available
        if batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints.shape[1] // 2 // num_views
            keypoints_pred_3d = project_camera_pairs_to_3d(
                points=pred_keypoints.reshape((-1, num_views, num_keypoints, 2)),
                intrinsics=batch_dict["intrinsic_matrix"],
                extrinsics=batch_dict["extrinsic_matrix"],
                dist=batch_dict["distortions"],
            )
            keypoints_targ_3d = batch_dict["keypoints_3d"]
            keypoints_mask_3d = get_valid_projection_masks(
                target_keypoints.reshape((-1, num_views, num_keypoints, 2))
            )
        else:
            keypoints_pred_3d = None
            keypoints_targ_3d = None
            keypoints_mask_3d = None

        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": pred_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": pred_keypoints,
            "confidences": confidence,
            "keypoints_targ_3d": keypoints_targ_3d,  # shape (2*batch, num_keypoints, 3)
            "keypoints_pred_3d": keypoints_pred_3d,  # shape (2*batch, cam_pairs, num_keypoints, 3)
            "keypoints_mask_3d": keypoints_mask_3d,  # shape (2*batch, cam_pairs, num_keypoints)
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pred_heatmaps = self.forward(images)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)

        if return_heatmaps:
            return pred_keypoints, confidence, pred_heatmaps
        else:
            return pred_keypoints, confidence

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
        ]
        return params


class HeatmapTrackerMultiviewMultihead(BaseSupervisedTracker):
    """Multi-headed convolutional network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        head: ALLOWED_MULTIVIEW_MULTIHEADS = "heatmap_cnn",
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
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            head: architecture used to fuse view information
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError(
                "HeatmapTrackerMultiviewHeatmapCNN currently only implements downsample_factor=2"
            )

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
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        if head == "heatmap_cnn":
            self.head = MultiviewHeatmapCNNMultiHead(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                upsampling_factor=1 if "vit" in backbone else 2,
            )
        else:
            raise NotImplementedError(
                f"{head} is not a valid multiview head, choose from {ALLOWED_MULTIVIEW_MULTIHEADS}"
            )

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # this attribute will be modified by AnnealWeight callback during training and can be
        # used to weight supervised, non-heatmap losses
        self.total_unsupervised_importance = torch.tensor(1.0)

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: TensorType["batch", "view", "channels":3, "image_height", "image_width"],
    ) -> Tuple[
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        batch_size, num_views, channels, img_height, img_width = images.shape

        # stack batch and view into first dim to get representations
        images = images.reshape(-1, channels, img_height, img_width)
        representations = self.get_representations(images)
        # representations shape is (view * batch, num_features, rep_height, rep_width)

        # get two heatmaps for each representation (single view, multi-view)
        heatmaps_sv, heatmaps_mv = self.head(representations, num_views)

        return heatmaps_sv, heatmaps_mv

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_sv, pred_heatmaps_mv = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints_sv, confidence_sv = self.head.run_subpixelmaxima(pred_heatmaps_sv)
        pred_keypoints_mv, confidence_mv = self.head.run_subpixelmaxima(pred_heatmaps_mv)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints_sv = convert_bbox_coords(batch_dict, pred_keypoints_sv)
        pred_keypoints_mv = convert_bbox_coords(batch_dict, pred_keypoints_mv)
        # project predictions from pairs of views into 3d if calibration data available
        if batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints_sv.shape[1] // 2 // num_views
            pred_keypoints_3d_sv = project_camera_pairs_to_3d(
                points=pred_keypoints_sv.reshape((-1, num_views, num_keypoints, 2)),
                intrinsics=batch_dict["intrinsic_matrix"],
                extrinsics=batch_dict["extrinsic_matrix"],
                dist=batch_dict["distortions"],
            )
            pred_keypoints_3d_mv = project_camera_pairs_to_3d(
                points=pred_keypoints_mv.reshape((-1, num_views, num_keypoints, 2)),
                intrinsics=batch_dict["intrinsic_matrix"],
                extrinsics=batch_dict["extrinsic_matrix"],
                dist=batch_dict["distortions"],
            )
            keypoints_pred_3d = torch.cat([pred_keypoints_3d_sv, pred_keypoints_3d_mv])
            keypoints_targ_3d = torch.cat([batch_dict["keypoints_3d"], batch_dict["keypoints_3d"]])

            keypoints_mask_3d_ = get_valid_projection_masks(
                target_keypoints.reshape((-1, num_views, num_keypoints, 2))
            )
            keypoints_mask_3d = torch.cat([keypoints_mask_3d_, keypoints_mask_3d_])
        else:
            keypoints_pred_3d = None
            keypoints_targ_3d = None
            keypoints_mask_3d = None

        return {
            "heatmaps_targ": torch.cat([batch_dict["heatmaps"], batch_dict["heatmaps"]], dim=0),
            "heatmaps_pred": torch.cat([pred_heatmaps_sv, pred_heatmaps_mv], dim=0),
            "keypoints_targ": torch.cat([target_keypoints, target_keypoints], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_sv, pred_keypoints_mv], dim=0),
            "confidences": torch.cat([confidence_sv, confidence_mv], dim=0),
            "keypoints_targ_3d": keypoints_targ_3d,  # shape (2*batch, num_keypoints, 3)
            "keypoints_pred_3d": keypoints_pred_3d,  # shape (2*batch, cam_pairs, num_keypoints, 3)
            "keypoints_mask_3d": keypoints_mask_3d,  # shape (2*batch, cam_pairs, num_keypoints)
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pred_heatmaps_sv, pred_heatmaps_mv = self.forward(images)
        # heatmaps -> keypoints
        pred_keypoints_sv, confidence_sv = self.head.run_subpixelmaxima(pred_heatmaps_sv)
        pred_keypoints_mv, confidence_mv = self.head.run_subpixelmaxima(pred_heatmaps_mv)

        # reshape keypoints to be (batch, n_keypoints, 2)
        pred_keypoints_sv = pred_keypoints_sv.reshape(pred_keypoints_sv.shape[0], -1, 2)
        pred_keypoints_mv = pred_keypoints_mv.reshape(pred_keypoints_mv.shape[0], -1, 2)

        # find higher confidence indices
        mv_conf_gt = torch.gt(confidence_mv, confidence_sv)

        # select higher confidence indices
        pred_keypoints_sv[mv_conf_gt] = pred_keypoints_mv[mv_conf_gt]
        pred_keypoints_sv = pred_keypoints_sv.reshape(pred_keypoints_sv.shape[0], -1)

        confidence_sv[mv_conf_gt] = confidence_mv[mv_conf_gt]

        # bounding box coords -> original image coords
        pred_keypoints_sv = convert_bbox_coords(batch_dict, pred_keypoints_sv)

        if return_heatmaps:
            pred_heatmaps_sv[mv_conf_gt] = pred_heatmaps_mv[mv_conf_gt]
            return pred_keypoints_sv, confidence_sv, pred_heatmaps_sv
        else:
            return pred_keypoints_sv, confidence_sv

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
        ]
        return params
