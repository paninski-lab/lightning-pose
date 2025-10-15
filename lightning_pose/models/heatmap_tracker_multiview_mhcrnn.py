"""
Multiview transformer with MHCRNN head for context frames.

This module extends the HeatmapTrackerMultiviewTransformer to support context frames
using the MHCRNN head, providing both cross-view information and temporal context.
"""

from typing import Any, Literal
from omegaconf import DictConfig
import math
import torch
import torch.nn as nn

from lightning_pose.models.heads import HeatmapMHCRNNHeadMultiview
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.data.utils import convert_bbox_coords
from lightning_pose.data.datatypes import UnlabeledBatchDict, MultiviewHeatmapLabeledBatchDict
from lightning_pose.data.cameras import project_camera_pairs_to_3d, get_valid_projection_masks


from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)

from typing_extensions import Self
from torchtyping import TensorType


class HeatmapTrackerMultiviewMHCRNN(BaseSupervisedTracker):
    """Multiview transformer with MHCRNN head for context frames."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: Literal["vits_dino", "vitb_dino", "vitb_imagenet", "vitb_sam"] = "vitb_imagenet",
        pretrained: bool = True,
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a multiview transformer with MHCRNN head for context frames."""
        
        # Reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_views = num_views
        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        # Initialize base class (handles backbone creation)
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=True,  # MHCRNN requires context
            **kwargs,
        )

        # Create learnable view embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device=device)
        generator.manual_seed(torch_seed)
        self.view_embeddings = nn.Parameter(
            torch.randn(self.num_views, self.num_fc_input_features,
                       generator=generator, device=device) * 0.02
        )

        # Initialize MHCRNN head
        self.head = HeatmapMHCRNNHeadMultiview(
            backbone_arch=backbone,
            in_channels=self.num_fc_input_features,
            out_channels=self.num_keypoints,
            downsample_factor=self.downsample_factor,
            num_views=num_views,
        )

        self.loss_factory = loss_factory
        
        self.rmse_loss = RegressionRMSELoss()


        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward_vit(
        self,
        images: TensorType["view * batch", "channels":3, "image_height", "image_width"],
    ):
        """Forward pass through vision encoder with view embeddings."""
        
        # Create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.backbone.vision_encoder.embeddings(
            images, bool_masked_pos=None, interpolate_pos_encoding=True,
        )[:, 1:]
        # Shape: (view * batch, num_patches, embedding_dim)

        # Get dimensions
        view_batch_size = embedding_output.shape[0]
        num_patches = embedding_output.shape[1]
        embedding_dim = embedding_output.shape[2]
        batch_size = view_batch_size // self.num_views

        # Create view indices: [0,1,2,3,0,1,2,3,...] for 4 views
        view_indices = torch.arange(self.num_views, device=embedding_output.device)
        view_indices = view_indices.repeat(batch_size)
        
        # Get view embeddings and expand to patches
        view_embeddings_batch = self.view_embeddings[view_indices]
        view_embeddings_expanded = view_embeddings_batch.unsqueeze(1).expand(-1, num_patches, -1)
        embedding_output = embedding_output + view_embeddings_expanded
        
        # Reshape to (batch, view * num_patches, embedding_dim)
        embedding_output = embedding_output.reshape(
            batch_size, self.num_views * num_patches, embedding_dim,
        )

        # Push through ViT encoder
        encoder_outputs = self.backbone.vision_encoder.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        outputs = self.backbone.vision_encoder.layernorm(sequence_output)
        # Shape: (batch, view * num_patches, embedding_dim)

        # Reshape to (view * batch, embedding_dim, height, width)
        patch_size = outputs.shape[1] // self.num_views
        H, W = math.isqrt(patch_size), math.isqrt(patch_size)
        outputs = outputs.reshape(batch_size, self.num_views, patch_size, embedding_dim)
        outputs = outputs.reshape(batch_size, self.num_views, H, W, embedding_dim).permute(
            0, 1, 4, 2, 3
        )
        outputs = outputs.reshape(view_batch_size, embedding_dim, H, W)

        return outputs

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
    ) -> tuple[TensorType, TensorType]:
        """Forward pass with MHCRNN head.
        
        Returns:
            tuple of (heatmaps_sf, heatmaps_mf), both shape: 
            (batch, num_views * num_keypoints, heatmap_height, heatmap_width)
        """
        
        # extract pixel data from batch
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]

        if len(images.shape) != 6:
            raise ValueError(
                f"HeatmapTrackerMultiviewMHCRNN requires context frames (6D input). "
                f"Got shape: {images.shape}, expected: (batch, views, frames, channels, height, width)"
            )
        
        batch_size, num_views, frames, channels, img_height, img_width = images.shape

        batch_shape = torch.tensor(images.shape)
        
        # Process each frame through transformer
        frame_representations = []
        for frame_idx in range(frames):
            # Extract and flatten frame
            frame_images = images[:, :, frame_idx, :, :, :]
            frame_images_flat = frame_images.reshape(-1, channels, img_height, img_width)
            # Pass through ViT
            frame_repr = self.forward_vit(frame_images_flat)
            frame_representations.append(frame_repr)
        
        # Stack and permute: (batch * views, features, height, width, frames)
        stacked = torch.stack(frame_representations, dim=0)
        representations = stacked.permute(1, 2, 3, 4, 0)
        
        # Pass through MHCRNN head
        heatmaps_sf, heatmaps_mf = self.head(representations, batch_shape, is_multiview=True)
        
        # Reshape to (batch, views * keypoints, height, width)
        heatmaps_sf = heatmaps_sf.reshape(
            batch_size, num_views * self.num_keypoints, 
            heatmaps_sf.shape[-2], heatmaps_sf.shape[-1]
        )
        heatmaps_mf = heatmaps_mf.reshape(
            batch_size, num_views * self.num_keypoints,
            heatmaps_mf.shape[-2], heatmaps_mf.shape[-1]
        )
        
        return heatmaps_sf, heatmaps_mf

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and keypoints for loss computation."""
        
        # Get predictions
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(batch_dict)
        
        # Extract keypoints
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)
        # pred_keypoints_*: (batch, views*keypoints*2) e.g., (10, 56)
        # confidence_*: (batch, views*keypoints) e.g., (10, 28)
        
        # Expand confidence for keypoint-level blending (need x,y separately)
        confidence_sf_expanded = confidence_sf.repeat_interleave(2, dim=1)  # (10, 56)
        confidence_mf_expanded = confidence_mf.repeat_interleave(2, dim=1)  # (10, 56)
        
        # Blend keypoints based on expanded confidence
        pred_keypoints = torch.where(
            confidence_mf_expanded > confidence_sf_expanded,
            pred_keypoints_mf,
            pred_keypoints_sf
        )
        # Blend ORIGINAL confidence values (not expanded)
        # This gives you one confidence per keypoint, not per coordinate
        confidence = torch.where(
            confidence_mf > confidence_sf,
            confidence_mf,
            confidence_sf
        )  # (10, 28) - correct shape
        
        # Convert coordinates
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
    
        
        # 3D projection if available
        keypoints_pred_3d = None
        keypoints_targ_3d = None
        keypoints_mask_3d = None
        
        if "keypoints_3d" in batch_dict and batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints.shape[1] // 2 // num_views

            try:
                keypoints_pred_3d = project_camera_pairs_to_3d(
                    points=pred_keypoints.reshape((-1, num_views, num_keypoints, 2)),
                    intrinsics=batch_dict["intrinsic_matrix"].float(),
                    extrinsics=batch_dict["extrinsic_matrix"].float(),
                    dist=batch_dict["distortions"].float(),
                )
                keypoints_targ_3d = batch_dict["keypoints_3d"]
                keypoints_mask_3d = get_valid_projection_masks(
                    target_keypoints.reshape((-1, num_views, num_keypoints, 2))
                )
            except Exception as e:
                print(f"Error in 3D projection: {e}")

        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": pred_heatmaps_sf,  # Use single-frame for loss
            "keypoints_targ": target_keypoints,
            "keypoints_pred": pred_keypoints,
            "confidences": confidence,
            "keypoints_targ_3d": keypoints_targ_3d,
            "keypoints_pred_3d": keypoints_pred_3d,
            "keypoints_mask_3d": keypoints_mask_3d,
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> tuple:
        """Prediction step."""
        
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(batch_dict)
        
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)
        
        # Blend based on confidence
        confidence_sf_expanded = confidence_sf.repeat_interleave(2, dim=1)
        confidence_mf_expanded = confidence_mf.repeat_interleave(2, dim=1)
        
        # Blend keypoints
        pred_keypoints = torch.where(
            confidence_mf_expanded > confidence_sf_expanded,
            pred_keypoints_mf,
            pred_keypoints_sf
        )
        
        # Blend original confidence (NOT expanded)
        confidence = torch.where(
            confidence_mf > confidence_sf,
            confidence_mf,
            confidence_sf
        )
        
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        
        if return_heatmaps:
            return pred_keypoints, confidence, pred_heatmaps_sf
        else:
            return pred_keypoints, confidence

    def get_parameters(self):
        return [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
            {"params": [self.view_embeddings], "name": "view_embeddings"},
        ]

class SemiSupervisedHeatmapTrackerMultiviewMHCRNN(SemiSupervisedTrackerMixin, HeatmapTrackerMultiviewMHCRNN):
    """Semi-supervised model for heatmap tracking with multiview MHCRNN head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_factory_unsup = kwargs.get("loss_factory_unsupervised")

    