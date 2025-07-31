"""Models that produce heatmaps of keypoints from images on multiview datasets."""

import math
from typing import Any, Literal, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torchtyping import TensorType

# from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.cameras import get_valid_projection_masks, project_camera_pairs_to_3d
from lightning_pose.data.datatypes import (
    MultiviewHeatmapLabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import convert_bbox_coords
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)
from lightning_pose.models.heads import (
    ALLOWED_MULTIVIEW_HEADS,
    ALLOWED_MULTIVIEW_MULTIHEADS,
    HeatmapHead,
    MultiviewFeatureTransformerHead,
    MultiviewHeatmapCNNHead,
    MultiviewHeatmapCNNMultiHead,
    MultiviewFeatureTransformerHeadLearnableCrossView,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapTrackerMultiview",
    "HeatmapTrackerMultiviewMultihead",
    "HeatmapTrackerMultiviewTransformer",
]

class CrossViewAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention for cross-view communication
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
          
        # layer normalization and feedforward 
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, num_views: int) -> torch.Tensor:
        """
        Apply cross-view attention.
        
        Args:
            x: Input tensor of shape (batch, view*patches, embed_dim)
            num_views: Number of views
            
        Returns:
            Enhanced tensor with cross-view information exchange
        """
        batch_size, total_patches, embed_dim = x.shape
        patches_per_view = total_patches // num_views

        # Reshape to separate views: (batch, num_views, patches_per_view, embed_dim)
        x_views = x.view(batch_size, num_views, patches_per_view, embed_dim)
        
        # Apply cross-view attention for each spatial position
        enhanced_patches = []
        
        for patch_idx in range(patches_per_view):
            # Extract same spatial position across all views
            # Shape: (batch, num_views, embed_dim)
            patch_across_views = x_views[:, :, patch_idx, :]
            
            # Apply cross-view attention (views attend to each other)
            residual = patch_across_views
            patch_across_views = self.norm1(patch_across_views)
            
            attended_patch, attention_weights = self.cross_attention(
                patch_across_views,  # query
                patch_across_views,  # key  
                patch_across_views   # value
            )
            
            # Residual connection
            attended_patch = residual + attended_patch
            
            # Feed-forward network with residual connection
            residual = attended_patch
            attended_patch = self.norm2(attended_patch)
            attended_patch = residual + self.ffn(attended_patch)
            
            enhanced_patches.append(attended_patch)
        
        # Reconstruct tensor: (batch, num_views, patches_per_view, embed_dim)
        enhanced_views = torch.stack(enhanced_patches, dim=2)
        
        # Reshape back to original format: (batch, view*patches, embed_dim)
        return enhanced_views.view(batch_size, total_patches, embed_dim)

class HeatmapTrackerMultiviewTransformer(BaseSupervisedTracker):
    """Transformer network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: Literal["vits_dino", "vitb_dino", "vitb_imagenet"] = "vitb_imagenet",
        pretrained: bool = True,
        # head: Literal["heatmap_cnn"] = "heatmap_cnn",
        head: Literal["heatmap_cnn", "feature_transformer_learnable_crossview"] = "heatmap_cnn",
        downsample_factor: Literal[1] = 1,  # Changed from 2 to 1 to match head configuration
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a multi-view model with transformer backbone.

        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: transformer variant to be used; cannot use convnets with this model
            pretrained: True to load pretrained imagenet weights
            head: architecture used to project per-view information to 2D heatmaps
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        self.num_views = num_views

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)
        
        # Add step tracking for curriculum learning
        self.current_training_step = 0

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            print("HeatmapTrackerMultiviewTransformer does not currently support context frames")
            # del kwargs["do_context"]

        print(f" we are using the Heatmap Multiview Transformer model with {num_views} views")
        print(f" backbone: {backbone}, pretrained: {pretrained}, head: {head}")
        
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

        #Create learnable view embeddings for each view
        self.view_embeddings = nn.Parameter(
            torch.randn(self.num_views, self.num_fc_input_features) * 0.02
        )  # Small initialization for stability

        
        # add cross-view attention layers 
        cross_view_num_heads = 8
        cross_view_num_layers = 2
        cross_view_dropout = 0.15

        self.cross_view_layers = nn.ModuleList([
            CrossViewAttention(
                embed_dim=self.num_fc_input_features,
                num_heads=cross_view_num_heads,
                dropout=cross_view_dropout
            )
            for _ in range(cross_view_num_layers)
        ])

    
        if head == "heatmap_cnn":
            self.head = HeatmapHead(
                backbone_arch=backbone,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        
        elif head == "feature_transformer_learnable_crossview":
            """Multiview Feature Transformer Head with learnable cross-view embeddings."""
            self.head = MultiviewFeatureTransformerHeadLearnableCrossView(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,  # Use model-level downsample_factor
                transformer_d_model=512,
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=3,
                img_size=image_size,
                view_embed_dim=128,
                dropout=0.1,
            )



        else:
            raise NotImplementedError(f"{head} is not a valid multiview transformer head")

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # this attribute will be modified by AnnealWeight callback during training and can be
        # used to weight supervised, non-heatmap losses
        self.total_unsupervised_importance = torch.tensor(1.0)

        # Initially freeze backbone parameters - will be unfrozen after 400 steps
        print("Freezing backbone parameters initially - will unfreeze after 400 steps")
        for param in self.backbone.parameters():
            param.requires_grad = False

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def apply_random_view_masking(self, images, mask_ratio=None, training_step=0):
        """
        Apply random view masking during training with curriculum learning.
        
        Args:
            images: Input tensor of shape (batch, num_views, channels, height, width)
            mask_ratio: Override masking ratio (if None, uses curriculum learning)
            training_step: Current training step for curriculum learning
            
        Returns:
            tuple: (masked_images, view_mask)
                - masked_images: Images with randomly selected views zeroed out
                - view_mask: Binary mask indicating which views are kept (1) or masked (0)
        """
        if not self.training:
            # During evaluation, don't apply masking
            batch_size = images.shape[0]
            view_mask = torch.ones(batch_size, self.num_views, device=images.device)
            return images, view_mask
            
        batch_size, num_views, channels, height, width = images.shape
        device = images.device
        
        # Curriculum learning: start at 0.1, increase to 0.5 over 5,000 steps
        if mask_ratio is None:
            curriculum_steps = 5000  # Number of steps to reach max masking
            progress = min(training_step / curriculum_steps, 1.0)
            mask_ratio = 0.1 + progress * 0.4  # 0.1 -> 0.5
        
        # Always keep minimum 2 views (never mask more than num_views - 2)
        max_masked_views = max(0, num_views - 2)
        num_views_to_mask = int(min(mask_ratio * num_views, max_masked_views))
        
        # Initialize view mask (1 = keep, 0 = mask)
        view_mask = torch.ones(batch_size, num_views, device=device)
        masked_images = images.clone()
        
        # Apply masking per batch sample
        for batch_idx in range(batch_size):
            if num_views_to_mask > 0:
                # Randomly select views to mask
                view_indices = torch.randperm(num_views, device=device)[:num_views_to_mask]
                view_mask[batch_idx, view_indices] = 0
                
                # Zero out the selected views completely
                for view_idx in view_indices:
                    masked_images[batch_idx, view_idx] = 0
        
        return masked_images, view_mask

    def get_training_schedule_info(self, current_step):
        """Get information about current training schedule progress."""
        curriculum_steps = 5000
        backbone_unfreeze_step = 400
        
        # Masking schedule
        progress = min(current_step / curriculum_steps, 1.0)
        current_mask_ratio = 0.1 + progress * 0.4
        
        # Backbone status
        backbone_frozen = current_step < backbone_unfreeze_step
        
        return {
            "step": current_step,
            "mask_ratio": current_mask_ratio,
            "backbone_frozen": backbone_frozen,
            "curriculum_progress": f"{progress*100:.1f}%",
            "steps_to_unfreeze": max(0, backbone_unfreeze_step - current_step),
            "steps_to_max_masking": max(0, curriculum_steps - current_step)
        }

    def training_step(self, batch_dict, batch_idx):
        """Override training step to track current step for curriculum learning and backbone unfreezing."""
        # Update training step for curriculum learning
        self.current_training_step = self.trainer.global_step
        
        # Unfreeze backbone after 400 steps
        if self.current_training_step == 400:
            print(f"Step {self.current_training_step}: Unfreezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.log("backbone_unfrozen", 1.0, on_step=True, on_epoch=False)
        
        # Log current mask ratio and training milestones
        if self.current_training_step % 100 == 0:  # Log every 100 steps
            curriculum_steps = 5000  # Updated to match 5000 step training
            progress = min(self.current_training_step / curriculum_steps, 1.0)
            current_mask_ratio = 0.1 + progress * 0.4
            self.log("mask_ratio", current_mask_ratio, on_step=True, on_epoch=False, prog_bar=True)
            
            # Log training schedule milestones
            if self.current_training_step in [1000, 2500, 4000, 5000]:
                schedule_info = self.get_training_schedule_info(self.current_training_step)
                print(f"Step {self.current_training_step} Training Schedule:")
                print(f"  - Mask ratio: {schedule_info['mask_ratio']:.3f}")
                print(f"  - Curriculum progress: {schedule_info['curriculum_progress']}")
                print(f"  - Backbone frozen: {schedule_info['backbone_frozen']}")
                if schedule_info['steps_to_max_masking'] > 0:
                    print(f"  - Steps to max masking: {schedule_info['steps_to_max_masking']}")
                else:
                    print(f"  - Maximum masking reached!")
        
        # Log backbone freezing status
        backbone_frozen = any(not p.requires_grad for p in self.backbone.parameters())
        if self.current_training_step % 500 == 0:
            self.log("backbone_frozen", float(backbone_frozen), on_step=True, on_epoch=False)
        
        # Call parent training step
        return super().training_step(batch_dict, batch_idx)

    def validation_step(self, batch_dict, batch_idx):
        """Override validation step to test model performance with different masking levels."""
        # Standard validation without masking
        val_loss = super().validation_step(batch_dict, batch_idx)
        
        # Test with different masking levels during validation
        if batch_idx % 10 == 0:  # Only test on every 10th batch to save compute
            original_training_state = self.training
            self.train()  # Temporarily set to training mode to enable masking
            
            mask_ratios = [0.2, 0.4]
            for mask_ratio in mask_ratios:
                try:
                    # Create a copy of batch_dict for testing
                    test_batch = dict(batch_dict)
                    
                    # Apply masking with specific ratio
                    if "images" in test_batch:
                        original_images = test_batch["images"].clone()
                        masked_images, _ = self.apply_random_view_masking(
                            original_images, mask_ratio=mask_ratio, training_step=self.current_training_step
                        )
                        test_batch["images"] = masked_images
                    
                    # Get predictions with masked views
                    loss_inputs = self.get_loss_inputs_labeled(test_batch)
                    if self.loss_factory is not None:
                        losses_dict = self.loss_factory.compute_loss(loss_inputs)
                        mask_loss = losses_dict["total_loss"] if "total_loss" in losses_dict else 0.0
                        
                        # Log masking performance
                        self.log(
                            f"val_loss_mask_{mask_ratio}",
                            mask_loss,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True
                        )
                
                except Exception as e:
                    # Don't fail validation if masking test fails
                    pass
            
            # Restore original training state
            self.train(original_training_state)
        
        return val_loss

    def forward_vit(
        self,
        images: TensorType["view * batch", "channels":3, "image_height", "image_width"],
        intrinsics: TensorType["view * batch", 3, 3],
        extrinsics: TensorType["view * batch", 3, 4],
        distortions: TensorType["view * batch", "num_dist_params"],
    ):
        """Override forward pass through the vision encoder to add view embeddings."""

        # outputs = self.vision_encoder(
        #     x,
        #     return_dict=True,
        #     output_hidden_states=False,
        #     output_attentions=False,
        #     interpolate_pos_encoding=True,
        # ).last_hidden_state

        # -----------------------------------------------------------------------------------------
        # this block mostly copies self.vision_encoder.forward(), except for addition of view embed

        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.backbone.vision_encoder.embeddings(
            images, bool_masked_pos=None, interpolate_pos_encoding=True,
        )[:, 1:]
        # shape: (view * batch, num_patches, embedding_dim)

        # -----------------------------------------------------------------------------------------
        # IMPLEMENTATION: Learnable View Embeddings
        # -----------------------------------------------------------------------------------------
       

        # -----------------------------------------------------------------------------------------

        # get dims for reshaping
        view_batch_size = embedding_output.shape[0]
        num_patches = embedding_output.shape[1]
        embedding_dim = embedding_output.shape[2]
        batch_size = view_batch_size // self.num_views


        # intrinsics = intrinsics.float() # [batch*views, 3, 3]
        # extrinsics = extrinsics.float() # [batch*views, 3, 4]
        # distortions = distortions.float() # [batch*views, 5] 

        # # CHANGE: Get actual image size instead of hardcoding 256
        # img_height, img_width = images.shape[-2:]  # Get from input images
        
        # # Normalize camera parameters with actual image dimensions
        # intrinsics_norm = intrinsics.clone()
        # intrinsics_norm[:, 0, 0] /= img_width   # fx normalized by width
        # intrinsics_norm[:, 1, 1] /= img_height  # fy normalized by height  
        # intrinsics_norm[:, 0, 2] /= img_width   # cx normalized by width
        # intrinsics_norm[:, 1, 2] /= img_height  # cy normalized by height
        
        # extrinsics_norm = extrinsics.clone()
        # extrinsics_norm[:, :, 3] /= 100.0  # Scale translation
        # distortions_norm = torch.clamp(distortions, -1.0, 1.0)


        # camera_params = torch.cat([
        #     intrinsics_norm.view(-1, 9),      # 3x3 intrinsics flattened
        #     extrinsics_norm.view(-1, 12),     # 3x4 extrinsics flattened  
        #     distortions_norm.view(-1, distortions_norm.shape[-1]),    # distortion parameters
        # ], dim=1)

        
        # Create view indices to map each sample to its corresponding view
        # Shape: (view * batch,)
        view_indices = torch.arange(self.num_views, device=embedding_output.device)
        view_indices = view_indices.repeat(batch_size)  # [0,1,2,3,0,1,2,3,...] for 4 views
        # Get view embeddings for each sample
        # Shape: (view * batch, embedding_dim)
        view_embeddings_batch = self.view_embeddings[view_indices]
        
        # Expand view embeddings to match patch dimensions
        # Shape: (view * batch, 1, embedding_dim) -> (view * batch, num_patches, embedding_dim)
        view_embeddings_expanded = view_embeddings_batch.unsqueeze(1).expand(-1, num_patches, -1)
        embedding_output = embedding_output + view_embeddings_expanded
        # Reshape to (batch, view * num_patches, embedding_dim) so that transformer attention
        # layers process all views simultaneously
        embedding_output = embedding_output.reshape(
            batch_size, self.num_views * num_patches, embedding_dim,
        )

        # push data through vit encoder
        encoder_outputs = self.backbone.vision_encoder.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        outputs = self.backbone.vision_encoder.layernorm(sequence_output)
        # shape: (batch, view * num_patches, embedding_dim)

        for cross_view_layer in self.cross_view_layers:
            outputs = cross_view_layer(outputs, self.num_views)

        # reshape data to (view * batch, embedding_dim, height, width) for head processing
        patch_size = outputs.shape[1] // self.num_views
        H, W = math.isqrt(patch_size), math.isqrt(patch_size)
        outputs = outputs.reshape(batch_size, self.num_views, patch_size, embedding_dim)
        outputs = outputs.reshape(batch_size, self.num_views, H, W, embedding_dim).permute(
            0, 1, 4, 2, 3
        )  # shape: (batch, view, embedding_dim, H, W)
        outputs = outputs.reshape(view_batch_size, embedding_dim, H, W)

        return outputs

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        # extract pixel data from batch
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]
        # CRITICAL FIX: Ensure view ordering for DALI multiview data
        # DALI processes views in arbitrary order, but we need them in view_names order
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'view_names'):
            expected_view_names = self.cfg.data.view_names
            if len(expected_view_names) == images.shape[1]:
                print(f"WARNING: DALI multiview data detected. Expected view order: {expected_view_names}")
                print(f"Current batch shape: {images.shape}")
                # TODO: Add view reordering logic here if needed

        batch_size, num_views, channels, img_height, img_width = images.shape

        # Apply random view masking during training for robustness
        if self.training:
            print(f"Applying random view masking at step {self.current_training_step}")
            images, view_mask = self.apply_random_view_masking(images, training_step=self.current_training_step)

        # extract camera parameters from batch (optional for video prediction)
        if "intrinsic_matrix" in batch_dict:
            intrinsics = batch_dict["intrinsic_matrix"]
            extrinsics = batch_dict["extrinsic_matrix"]
            distortions = batch_dict["distortions"]
            
            # stack batch and view into first dim to pass through transformer
            images = images.reshape(-1, channels, img_height, img_width)
            if intrinsics.shape[1] == 1:  # Placeholder case [batch, 1]
                # First reshape to add the missing dimensions
                intrinsics = intrinsics.reshape(-1, 1, 3, 3)  # [16, 1, 3, 3]
                extrinsics = extrinsics.reshape(-1, 1, 3, 4)  # [16, 1, 3, 4]
                distortions = distortions.reshape(-1, 1, 5)  # [16,1,5]
                
                # Then expand the second dimension
                intrinsics = intrinsics.expand(-1, self.num_views, -1, -1).reshape(-1, 3, 3)
                extrinsics = extrinsics.expand(-1, self.num_views, -1, -1).reshape(-1, 3, 4)
                distortions = distortions.expand(-1, self.num_views, -1).reshape(-1, distortions.shape[-1])
            else:  # Normal case [batch, num_views, ...]
                # Reshape real camera parameters
                intrinsics = intrinsics.reshape(-1, 3, 3)
                extrinsics = extrinsics.reshape(-1, 3, 4)
                distortions = distortions.reshape(-1, distortions.shape[-1])
        else:
            # For video prediction without camera parameters, create dummy parameters
            # These won't be used in the forward pass but are needed for the function signature
            images = images.reshape(-1, channels, img_height, img_width)
            batch_size_total = images.shape[0]
            device = images.device
            
            # Create dummy camera parameters
            intrinsics = torch.eye(3, device=device).unsqueeze(0).expand(batch_size_total, -1, -1)
            extrinsics = torch.eye(3, 4, device=device).unsqueeze(0).expand(batch_size_total, -1, -1)
            distortions = torch.zeros(batch_size_total, 5, device=device)

        # pass through transformer to get base representations
        representations = self.forward_vit(images, intrinsics, extrinsics, distortions)
        # shape: (view * batch, num_features, rep_height, rep_width)

        # get heatmaps for each representation
        if hasattr(self.head, 'forward') and self.head.__class__.__name__ == 'MultiviewFeatureTransformerHeadLearnableCrossView':
            heatmaps = self.head(representations, self.num_views)
        else:
            heatmaps = self.head(representations)

        # reshape to put all views from a single example together
        heatmaps = heatmaps.reshape(batch_size, -1, heatmaps.shape[-2], heatmaps.shape[-1])

        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        # project predictions from pairs of views into 3d if calibration data available
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
                keypoints_pred_3d = None
                keypoints_targ_3d = None
                keypoints_mask_3d = None
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
        # images -> heatmaps
        # if isinstance(batch_dict, dict) and "images" in batch_dict:
        #     # Convert to proper type for labeled data
        #     labeled_batch = MultiviewHeatmapLabeledBatchDict(**batch_dict)
        #     pred_heatmaps = self.forward(labeled_batch)
        # else:
        #     # Handle unlabeled data - this should not happen for this model
        #     raise ValueError("HeatmapTrackerMultiviewTransformer expects labeled data")
        
        pred_heatmaps = self.forward(batch_dict)
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
            {"params": [self.view_embeddings], "name": "view_embeddings"},
            {"params": self.cross_view_layers.parameters(), "name": "cross_view_layers"},
        ]
        
        # params = [
        #     {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
        #     {"params": self.head.parameters(), "name": "head", "lr": 5e-4},  # Middle ground: 5e-4
        #     {"params": [self.view_embeddings], "name": "view_embeddings", "lr": 5e-4},  # Same as head
        # ]
        return params


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
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=4,
                img_size=image_size,
            )
        
        elif head == "feature_transformer_learnable_crossview":
            self.head = MultiviewFeatureTransformerHeadLearnableCrossView(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                transformer_d_model=512,
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=3,
                img_size=image_size,
                view_embed_dim=128,
                dropout=0.1,
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
        pred_heatmaps = self.forward(batch_dict)
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
                intrinsics=batch_dict["intrinsic_matrix"].float(),
                extrinsics=batch_dict["extrinsic_matrix"].float(),
                dist=batch_dict["distortions"].float(),
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
