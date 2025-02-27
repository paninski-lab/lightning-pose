"""Models that produce heatmaps of keypoints from images on multiview datasets."""

from typing import Any, Tuple

import torch
from kornia.geometry.subpix import spatial_softmax2d
from omegaconf import DictConfig
from torch import nn, Tensor
from torchtyping import TensorType
# from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.utils import (
    MultiviewHeatmapLabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.models import HeatmapTracker
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    SemiSupervisedTrackerMixin,
    convert_bbox_coords,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapTrackerMultiview",
    "ResidualBlock",
]


import itertools
from kornia.geometry.calibration import undistort_points
from kornia.geometry.epipolar import triangulate_points


def project_camera_pairs_to_3d(
        points: TensorType["batch", "num_views", "num_keypoints", 2],
        intrinsics: TensorType["batch", "num_views", 3, 3],
        extrinsics: TensorType["batch", "num_views", 3, 4],
        dist: TensorType["batch", "num_views", "num_params"],
) -> TensorType["batch", "cam_pair", "num_keypoints", 3]:
    """Project 2D keypoints from each pair of cameras into 3D world space."""

    num_batch, num_views, num_keypoints, _ = points.shape

    points = undistort_points(
        points=points,
        K=intrinsics,
        dist=dist,
        new_K=torch.eye(3, device=points.device).expand(num_batch, num_views, 3, 3),
    )

    p3d = []
    for j1, j2 in itertools.combinations(range(num_views), 2):
        points1 = points[:, j1, ...]
        points2 = points[:, j2, ...]

        # Create a mask for valid keypoints
        # A keypoint is valid if it's not NaN in BOTH views
        valid_mask = ~(
                torch.isnan(points1).any(dim=-1) |
                torch.isnan(points2).any(dim=-1)
        )

        # Prepare points for triangulation
        tri = torch.full(
            (num_batch, num_keypoints, 3),
            float('nan'),
            device=points.device,
            dtype=points.dtype,
        )

        # Triangulate only valid points
        for batch_idx in range(num_batch):
            # Get valid keypoint indices for this batch
            batch_valid_indices = torch.where(valid_mask[batch_idx])[0]

            if len(batch_valid_indices) > 0:
                # Extract valid points for this batch
                batch_points1 = points1[batch_idx][valid_mask[batch_idx]]
                batch_points2 = points2[batch_idx][valid_mask[batch_idx]]

                # Triangulate valid points
                batch_tri = triangulate_points(
                    P1=extrinsics[batch_idx, j1],
                    P2=extrinsics[batch_idx, j2],
                    points1=batch_points1,
                    points2=batch_points2,
                )

                # Place triangulated points back in the full tensor
                tri[batch_idx, valid_mask[batch_idx]] = batch_tri

        p3d.append(tri)

    return torch.stack(p3d, dim=1)


def get_valid_projection_masks(
    points: TensorType["batch", "num_views", "num_keypoints", 2]
) -> TensorType["batch", "cam_pair", "num_keypoints"]:

    num_batch, num_views, num_keypoints, _ = points.shape

    m3d = []
    for j1, j2 in itertools.combinations(range(num_views), 2):
        points1 = points[:, j1, :, 0]
        points2 = points[:, j2, :, 0]
        m3d.append(~torch.isnan(points1 + points2))
    return torch.stack(m3d, dim=1)


class HeatmapTrackerMultiview(HeatmapTracker):

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        downsample_factor: Literal[1, 2, 3] = 2,
        output_shape: tuple | None = None,  # change
        torch_seed: int = 123,
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
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            output_shape: hard-coded image size to avoid dynamic shape computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError(
                "HeatmapTrackerMultiview currently only implements downsample_factor=2"
            )

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            output_shape=output_shape,
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )

        # alias parent upsampling layers for single view
        self.upsampling_layers_sv = self.upsampling_layers

        # create upsampling layers for multiview head
        self.upsampling_layers_mv = self.make_upsampling_layers()
        self.initialize_upsampling_layers(self.upsampling_layers_mv)

        # create multiview head
        self.num_views = num_views
        self.multiview_head = ResidualBlock(
            in_channels=num_views,
            intermediate_channels=32,
            out_channels=num_views,
            final_relu=False,  # we'll use spatial_softmax2d instead later
        )

        # this attribute will be modified by AnnealWeight callback during training
        self.total_unsupervised_importance = torch.tensor(1.0)

    def heatmaps_from_representations(
        self,
        representations: TensorType["batch", "features", "rep_height", "rep_width"],
        num_views: int,
    ) -> Tuple[
            TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
            TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Upsample and run multiview head to get final heatmaps."""

        batch_size_combined = representations.shape[0]
        batch_size = int(batch_size_combined // num_views)

        # ----------------------------------------------------------------------------
        # process single view head (upsampling only)
        # ----------------------------------------------------------------------------
        heatmaps_sv = self.upsampling_layers_sv(representations)
        # heatmaps = [view * batch, num_keypoints, heatmap_height, heatmap_width]
        heat_height = heatmaps_sv.shape[-2]
        heat_width = heatmaps_sv.shape[-1]

        heatmaps_sv = heatmaps_sv.reshape(batch_size, num_views, -1, heat_height, heat_width)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps_sv = heatmaps_sv.reshape(batch_size, -1, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]

        # ----------------------------------------------------------------------------
        # process multi-view head (upsampling + multiview)
        # ----------------------------------------------------------------------------
        heatmaps_mv = self.upsampling_layers_mv(representations)
        # heatmaps = [view * batch, num_keypoints, heatmap_height, heatmap_width]

        # now we will process the heatmaps for each set of corresponding keypoints with the
        # multiview head; this requires lots of reshaping

        heatmaps_mv = heatmaps_mv.reshape(batch_size, num_views, -1, heat_height, heat_width)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.permute(0, 2, 1, 3, 4)
        # heatmaps = [batch, num_keypoints, views, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.reshape(-1, num_views, heat_height, heat_width)
        # heatmaps = [num_keypoints * batch, views, heatmap_height, heatmap_width]
        heatmaps_mv = self.multiview_head(heatmaps_mv)
        # heatmaps = [num_keypoints * batch, views, heatmap_height, heatmap_width]

        # reshape heatmaps back to their original shape
        heatmaps_mv = heatmaps_mv.reshape(batch_size, -1, num_views, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints, views, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.permute(0, 2, 1, 3, 4)
        # heatmaps = [batch, views, num_keypoints, heatmap_height, heatmap_width]
        heatmaps_mv = heatmaps_mv.reshape(batch_size, -1, heat_height, heat_width)
        # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]

        return heatmaps_sv, heatmaps_mv

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

        # we get one representation for each desired output.
        batch_size, num_views, channels, img_height, img_width = images.shape

        # stack batch and view into first dim to get representations
        images = images.reshape(-1, channels, img_height, img_width)
        representations = self.get_representations(images)
        # representations = [view * batch, num_features, rep_height, rep_width]

        # now get heatmaps from the representations
        heatmaps_sv, heatmaps_mv = self.heatmaps_from_representations(
            representations=representations, num_views=num_views,
        )
        # heatmaps_* = [batch, num_keypoints * views, heatmap_height, heatmap_width]

        # normalize heatmaps
        # softmax temp stays 1 here; to modify for model predictions, see constructor
        heatmaps_sv = spatial_softmax2d(heatmaps_sv, temperature=torch.tensor([1.0]))
        heatmaps_mv = spatial_softmax2d(heatmaps_mv, temperature=torch.tensor([1.0]))

        return heatmaps_sv, heatmaps_mv

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_sv, pred_heatmaps_mv = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints_sv, confidence_sv = self.run_subpixelmaxima(pred_heatmaps_sv)
        pred_keypoints_mv, confidence_mv = self.run_subpixelmaxima(pred_heatmaps_mv)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints_sv = convert_bbox_coords(batch_dict, pred_keypoints_sv)
        pred_keypoints_mv = convert_bbox_coords(batch_dict, pred_keypoints_mv)
        # project predictions from pairs of views into 3d if calibration data available
        if batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints_sv.shape[1] // 2 // num_views
            # pred_keypoints_3d_sv = project_camera_pairs_to_3d(
            #     points=pred_keypoints_sv.reshape((-1, num_views, num_keypoints, 2)),
            #     intrinsics=batch_dict["intrinsic_matrix"],
            #     extrinsics=batch_dict["extrinsic_matrix"],
            #     dist=batch_dict["distortions"],
            # )
            pred_keypoints_3d_mv = project_camera_pairs_to_3d(
                points=pred_keypoints_mv.reshape((-1, num_views, num_keypoints, 2)),
                intrinsics=batch_dict["intrinsic_matrix"],
                extrinsics=batch_dict["extrinsic_matrix"],
                dist=batch_dict["distortions"],
            )
            # keypoints_pred_3d = torch.cat([pred_keypoints_3d_sv, pred_keypoints_3d_mv])
            # keypoints_targ_3d = torch.cat([batch_dict["keypoints_3d"], batch_dict["keypoints_3d"]])
            #
            keypoints_mask_3d_ = get_valid_projection_masks(
                target_keypoints.reshape((-1, num_views, num_keypoints, 2))
            )
            # keypoints_mask_3d = torch.cat([keypoints_mask_3d_, keypoints_mask_3d_])
            keypoints_pred_3d = pred_keypoints_3d_mv
            keypoints_targ_3d = batch_dict["keypoints_3d"]
            keypoints_mask_3d = keypoints_mask_3d_
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
            # "heatmaps_targ": batch_dict["heatmaps"],
            # "heatmaps_pred": pred_heatmaps_mv,
            # "keypoints_targ": target_keypoints,
            # "keypoints_pred": pred_keypoints_mv,
            # "confidences": confidence_mv,
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
        pred_keypoints_sv, confidence_sv = self.run_subpixelmaxima(pred_heatmaps_sv)
        pred_keypoints_mv, confidence_mv = self.run_subpixelmaxima(pred_heatmaps_mv)

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
            {"params": self.upsampling_layers_sv.parameters(), "name": "upsampling_sv"},
            {"params": self.upsampling_layers_mv.parameters(), "name": "upsampling_mv"},
            {"params": self.multiview_head.parameters(), "name": "multiview_head"},
        ]
        return params


class ResidualBlock(nn.Module):
    """Resnet residual block module.

    Adapted from:
    https://github.com/pytorch/vision/blob/4249b610811b290ea9ac9e445260be195ce52ae1/torchvision/models/resnet.py#L59  # noqa

    """
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        final_relu: bool = False,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.final_relu = final_relu

        self.initialize_layers()

    def initialize_layers(self):

        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=0.01)
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.constant_(self.bn1.weight, 1.0)
        torch.nn.init.constant_(self.bn1.bias, 0.0)

        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=0.01)
        torch.nn.init.zeros_(self.conv2.bias)

        torch.nn.init.constant_(self.bn2.weight, 1.0)
        torch.nn.init.constant_(self.bn2.bias, 0.0)

        torch.nn.init.xavier_uniform_(self.conv3.weight, gain=0.01)
        torch.nn.init.zeros_(self.conv3.bias)

        torch.nn.init.constant_(self.bn3.weight, 1.0)
        torch.nn.init.constant_(self.bn3.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        # first layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # second layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # third layer
        out = self.conv3(out)
        out = self.bn3(out)
        # residual connection
        out += x
        if self.final_relu:
            out = self.relu(out)
        return out
