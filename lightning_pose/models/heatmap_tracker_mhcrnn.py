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
from lightning_pose.data.utils import undo_affine_transform_batch
from lightning_pose.losses.factory import LossFactory
from lightning_pose.models import HeatmapTracker
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    SemiSupervisedTrackerMixin,
    convert_bbox_coords,
)
from lightning_pose.models.heads import HeatmapMHCRNNHead

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapTrackerMHCRNN",
    "SemiSupervisedHeatmapTrackerMHCRNN",
]


class HeatmapTrackerMHCRNN(HeatmapTracker):
    """Multi-headed Convolutional RNN network that handles context frames."""

    def __init__(
        self,
        num_keypoints: int,
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
    ):
        """Initialize a DLC-like model with resnet backbone.

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
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError("MHCRNN currently only implements downsample_factor=2")

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            del kwargs["do_context"]

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
            do_context=True,
            **kwargs,
        )

        self.multihead = HeatmapMHCRNNHead(
            single_frame_head=self.head,
            upsampling_factor=1 if "vit" in backbone else 2,
        )

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
        num_frames = shape[0]

        # get one representation for each frame
        representations = self.get_representations(images, is_multiview=is_multiview)
        # representations shape is (batch, features, height, width, frames)

        if len(shape) == 5 and is_multiview:
            # put view info back in batch so we can properly extract heatmaps
            shape_r = representations.shape
            num_frames -= 4  # we lose the first/last 2 frames of unlabeled batch due to context
            representations = representations.reshape(
                num_frames * shape[1], -1, shape_r[-3], shape_r[-2], shape_r[-1],
            )

        # get two heatmaps for each representation (context, non-context)
        heatmaps_crnn, heatmaps_sf = self.multihead(representations, shape, num_frames)

        return heatmaps_crnn, heatmaps_sf

    def get_loss_inputs_labeled(
        self,
        batch_dict: Union[
            HeatmapLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_crnn, pred_heatmaps_sf = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints_crnn, confidence_crnn = self.head.run_subpixelmaxima(pred_heatmaps_crnn)
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        return {
            "heatmaps_targ": torch.cat([batch_dict["heatmaps"], batch_dict["heatmaps"]], dim=0),
            "heatmaps_pred": torch.cat([pred_heatmaps_crnn, pred_heatmaps_sf], dim=0),
            "keypoints_targ": torch.cat([batch_dict["keypoints"], batch_dict["keypoints"]], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_crnn, pred_keypoints_sf], dim=0),
            "confidences": torch.cat([confidence_crnn, confidence_sf], dim=0),
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
        pred_heatmaps_crnn, pred_heatmaps_sf = self.forward(images)
        # heatmaps -> keypoints
        pred_keypoints_crnn, confidence_crnn = self.head.run_subpixelmaxima(pred_heatmaps_crnn)
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        # reshape keypoints to be (batch, n_keypoints, 2)
        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1, 2)
        pred_keypoints_crnn = pred_keypoints_crnn.reshape(pred_keypoints_crnn.shape[0], -1, 2)
        # find higher confidence indices
        crnn_conf_gt = torch.gt(confidence_crnn, confidence_sf)
        # select higher confidence indices
        pred_keypoints_sf[crnn_conf_gt] = pred_keypoints_crnn[crnn_conf_gt]
        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1)
        confidence_sf[crnn_conf_gt] = confidence_crnn[crnn_conf_gt]
        # bounding box coords -> original image coords
        pred_keypoints_sf = convert_bbox_coords(batch_dict, pred_keypoints_sf)

        if return_heatmaps:
            pred_heatmaps_sf[crnn_conf_gt] = pred_heatmaps_crnn[crnn_conf_gt]
            return pred_keypoints_sf, confidence_sf, pred_heatmaps_sf
        else:
            return pred_keypoints_sf, confidence_sf

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {
                "params": self.multihead.head_mf.layers.parameters(),
                "name": "upsampling_rnn",
            },
            {"params": self.multihead.head_sf.parameters(), "name": "upsampling_sf"},
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
