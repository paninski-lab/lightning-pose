"""Multiview transformer with efficient MHCRNN head for context frames."""

from typing import Any, Literal, Tuple
from omegaconf import DictConfig
import math
import torch
import torch.nn as nn

from lightning_pose.data.datatypes import UnlabeledBatchDict, MultiviewHeatmapLabeledBatchDict
from lightning_pose.data.utils import convert_bbox_coords
from lightning_pose.data.cameras import project_camera_pairs_to_3d
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss

from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)

from lightning_pose.models.heads import HeatmapMHCRNNHeadMultiview

from torchtyping import TensorType


class HeatmapTrackerMultiviewMHCRNN(BaseSupervisedTracker):
    """Efficient multiview transformer with MHCRNN head for context frames."""

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
        """Initialize efficient multiview transformer with MHCRNN head."""
        
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

        # Initialize efficient MHCRNN head
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
        images: TensorType["view * batch * frames", "channels":3, "image_height", "image_width"],
        num_frames: int,
    ):
        """Optimized ViT forward pass: process frames separately to reduce attention complexity.
        
        Instead of processing all frames together (O((views×frames×patches)²)),
        we process each frame separately (O(frames × (views×patches)²)),
        keeping attention complexity linear in frames instead of quadratic.
        """
        
        # Reshape to separate frames: (batch, views, frames, channels, H, W)
        total_samples = images.shape[0]  # views * batch * frames
        batch_size = total_samples // (self.num_views * num_frames)
        images = images.reshape(batch_size, self.num_views, num_frames, *images.shape[1:])
        # Shape: (batch, views, frames, channels, H, W)
        
        # Process each frame separately through transformer
        # This keeps attention at views × patches per frame (same as original multiview)
        frame_features_list = []
        
        for frame_idx in range(num_frames):
            # Extract one frame: (batch, views, channels, H, W)
            frame_images = images[:, :, frame_idx, :, :, :]
            frame_images_flat = frame_images.reshape(-1, *frame_images.shape[2:])
            # Shape: (batch × views, channels, H, W)
            
            # Create patch embeddings and add position embeddings; remove CLS token
            try:
                embedding_output = self.backbone.vision_encoder.embeddings(
                    frame_images_flat, bool_masked_pos=None, interpolate_pos_encoding=True,
                )[:, 1:]
            except TypeError:
                # DINOv3 doesn't have `interpolate_pos_encoding` arg
                embedding_output = self.backbone.vision_encoder.embeddings(
                    frame_images_flat, bool_masked_pos=None,
                )[:, 1:]
            # Shape: (batch × views, num_patches, embedding_dim)
            
            # Get dimensions
            num_patches = embedding_output.shape[1]
            embedding_dim = embedding_output.shape[2]
            
            # Create view indices: [0,1,2,3,0,1,2,3,...] for this frame
            view_indices = torch.arange(self.num_views, device=embedding_output.device)
            view_indices = view_indices.repeat(batch_size)
            
            # Get view embeddings for each sample
            view_embeddings_batch = self.view_embeddings[view_indices]
            
            # Expand view embeddings to match patch dimensions
            view_embeddings_expanded = view_embeddings_batch.unsqueeze(1).expand(-1, num_patches, -1)
            embedding_output = embedding_output + view_embeddings_expanded
            
            # Reshape for transformer: (batch, views × num_patches, embedding_dim)
            # This is the KEY: attention only over views × patches, not frames
            embedding_output = embedding_output.reshape(
                batch_size, self.num_views * num_patches, embedding_dim,
            )
            
            # Push through ViT encoder (attention over views × patches only)
            encoder_outputs = self.backbone.vision_encoder.encoder(
                embedding_output,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=None,
            )
            sequence_output = encoder_outputs[0]
            outputs = self.backbone.vision_encoder.layernorm(sequence_output)
            # Shape: (batch, views × num_patches, embedding_dim)
            
            # Reshape to (batch × views, embedding_dim, height, width)
            patch_size = outputs.shape[1] // self.num_views
            H, W = math.isqrt(patch_size), math.isqrt(patch_size)
            outputs = outputs.reshape(batch_size, self.num_views, patch_size, embedding_dim)
            outputs = outputs.reshape(batch_size, self.num_views, H, W, embedding_dim).permute(
                0, 1, 4, 2, 3
            )  # Shape: (batch, views, embedding_dim, H, W)
            outputs = outputs.reshape(batch_size * self.num_views, embedding_dim, H, W)
            
            frame_features_list.append(outputs)
        
        # Stack frames: (batch × views, embedding_dim, H, W, frames)
        # This will be reshaped in forward() to (batch × views, embedding_dim, H, W, frames)
        return torch.stack(frame_features_list, dim=-1)

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
    ) -> Tuple[TensorType, TensorType]:
        """Efficient forward pass with single backbone call."""
        
        # Extract pixel data
        if "images" in batch_dict.keys():
            images = batch_dict["images"]
        else:
            images = batch_dict["frames"]

        # Handle different input shapes from DALI vs labeled data
        if len(images.shape) == 5:
            # Could be DALI multiview context: (seq_len, views, channels, H, W)
            # or labeled: (batch, views, frames, channels, H, W)
            # Check if first dim matches num_views (DALI format) or is larger (labeled format)
            seq_len_or_batch, dim2, channels, img_height, img_width = images.shape
            
            if dim2 == self.num_views:
                # DALI format: (seq_len, views, channels, H, W)
                # Reshape to (1, views, seq_len, channels, H, W) for batch processing
                frames = seq_len_or_batch
                images = images.permute(1, 0, 2, 3, 4).unsqueeze(0)  # (1, views, frames, channels, H, W)
            else:
                # Labeled format: (batch, views, frames, channels, H, W) - already correct
                pass
        elif len(images.shape) == 4:
            # Single frame multiview: (batch*views, channels, H, W) or (seq_len*views, channels, H, W)
            # This shouldn't happen for context model - DALI should provide context frames
            # Check if this might be a flattened context batch
            total_samples, channels, img_height, img_width = images.shape
            
            # If total_samples is divisible by (num_views * expected_frames), might be flattened
            expected_frames = 5  # context length
            if total_samples % (self.num_views * expected_frames) == 0:
                # Might be flattened context: reshape and try
                batch_size = total_samples // (self.num_views * expected_frames)
                images = images.reshape(batch_size, self.num_views, expected_frames, channels, img_height, img_width)
            elif total_samples % self.num_views == 0:
                # Single frames per view: (batch*views, channels, H, W)
                # This is a fallback - repeat the frame 5 times to create context
                # This is not ideal but allows prediction to proceed
                batch_size = total_samples // self.num_views
                # Reshape to (batch, views, 1, channels, H, W) then repeat frames
                images = images.reshape(batch_size, self.num_views, 1, channels, img_height, img_width)
                images = images.repeat(1, 1, expected_frames, 1, 1, 1)  # Repeat frame 5 times
                
                # Adjust bbox to match the new batch structure
                # For context models, normalized_to_bbox does bbox[2:-2], so we need extra padding
                # Keypoints will have shape (batch, views*keypoints*2) after processing
                # bbox needs to have shape (batch+4, views*4) to allow [2:-2] slicing
                if "bbox" in batch_dict:
                    original_bbox = batch_dict["bbox"]
                    if len(original_bbox.shape) == 2:
                        # Determine bbox structure
                        if original_bbox.shape[1] == 4:
                            # (seq_len, 4) - one bbox per sample, reshape to (batch, views, 4)
                            bbox_per_view = original_bbox.reshape(batch_size, self.num_views, 4)
                            # Flatten to (batch, views*4)
                            bbox_flat = bbox_per_view.reshape(batch_size, -1)
                        elif original_bbox.shape[1] == self.num_views * 4:
                            # (seq_len, views*4) - already in correct format
                            bbox_flat = original_bbox[:batch_size] if original_bbox.shape[0] >= batch_size else original_bbox
                        else:
                            # Unexpected shape - try to extract
                            bbox_flat = original_bbox[:batch_size, :self.num_views * 4] if original_bbox.shape[0] >= batch_size else original_bbox[:, :self.num_views * 4]
                        
                        # For context models, normalized_to_bbox expects bbox[2:-2], so pad with 2 frames at start and end
                        # Pad with first frame at start and last frame at end
                        first_frame = bbox_flat[0:1].repeat(2, 1)
                        last_frame = bbox_flat[-1:].repeat(2, 1)
                        batch_dict["bbox"] = torch.cat([first_frame, bbox_flat, last_frame], dim=0)
                
                import warnings
                warnings.warn(
                    f"DALI provided single frames instead of context frames. "
                    f"Repeating frames {expected_frames} times as fallback. "
                    f"Performance may be degraded. Check DALI configuration."
                )
            else:
                raise ValueError(
                    f"HeatmapTrackerMultiviewMHCRNN requires context frames, but DALI provided single frames. "
                    f"Got shape: {images.shape} (expected 5D or 6D with context frames). "
                    f"Total samples ({total_samples}) is not divisible by num_views ({self.num_views}). "
                    f"This suggests the DALI pipeline is not configured for context. "
                    f"Check that model_type='context' is set in PrepareDALI for multiview context models."
                )
        elif len(images.shape) != 6:
            raise ValueError(
                f"HeatmapTrackerMultiviewMHCRNN requires context frames. "
                f"Got shape: {images.shape}, expected: (batch, views, frames, channels, height, width) "
                f"or (seq_len, views, channels, height, width) from DALI"
            )
        
        batch_size, num_views, frames, channels, img_height, img_width = images.shape
        batch_shape = torch.tensor(images.shape)

        # ✅ OPTIMIZED: Process frames separately in transformer (reduces attention complexity)
        # Flatten all frames × views for embedding creation
        images_flat = images.reshape(-1, channels, img_height, img_width)
        # Shape: [batch * views * frames, 3, H, W]
        
        # forward_vit now processes frames separately and returns (batch×views, features, H, W, frames)
        features = self.forward_vit(images_flat, frames)
        # Shape: [batch × views, features, fh, fw, frames]
        
        # Prepare for MHCRNN: already in correct shape!
        features_mhcrnn = features
        # Shape: [batch*views, features, fh, fw, frames]
        
        # MHCRNN temporal processing
        heatmaps_sf, heatmaps_mf = self.head(features_mhcrnn, batch_shape, is_multiview=True)
        
        # Reshape to [batch, views * keypoints, height, width]
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
        """Return predictions and targets for loss computation."""
        
        # Get predictions
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(batch_dict)
        
        # Extract keypoints from heatmaps
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)
        
        # Expand confidence for keypoint-level blending (x,y coordinates)
        confidence_sf_expanded = confidence_sf.repeat_interleave(2, dim=1)
        confidence_mf_expanded = confidence_mf.repeat_interleave(2, dim=1)
        
        # Blend keypoints based on confidence
        pred_keypoints = torch.where(
            confidence_mf_expanded > confidence_sf_expanded,
            pred_keypoints_mf,
            pred_keypoints_sf
        )
        
        # Blend confidence values (one per keypoint, not per coordinate)
        confidence = torch.where(
            confidence_mf > confidence_sf,
            confidence_mf,
            confidence_sf
        )
        
        # Convert to original image coordinates
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        
        # 3D projection if calibration data available
        keypoints_pred_3d = None
        keypoints_targ_3d = None
        
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
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> Tuple:
        """Prediction step with confidence-based blending."""
        
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
        
        # Blend confidence (single value per keypoint)
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
    """Semi-supervised version of the efficient multiview MHCRNN."""

    def __init__(self, *args, **kwargs):
        loss_factory_unsupervised = kwargs.get("loss_factory_unsupervised")
        super().__init__(*args, **kwargs)
        self.loss_factory_unsup = kwargs.get("loss_factory_unsupervised")
