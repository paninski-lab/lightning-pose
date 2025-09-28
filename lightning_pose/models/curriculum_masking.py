"""Utility classes for curriculum learning and masking in multiview transformer models."""

from typing import Any, Dict, Literal, Tuple

import torch


class CurriculumMasking:
    """Handles curriculum learning and masking for multiview transformer training."""
    
    def __init__(
        self,
        num_views: int,
        patch_mask_config: dict = None,
        backbone_unfreeze_step: int = 400,
        patch_seed: int = None,
    ):
        """Initialize curriculum masking parameters.
        
        Args:
            num_views: Number of camera views
            patch_mask_config: Dictionary containing patch masking configuration
                - init_step: Step to start patch masking
                - final_step: Step when patch masking reaches maximum
                - init_ratio: Initial masking ratio
                - final_ratio: Final masking ratio
            backbone_unfreeze_step: Step to unfreeze backbone
            patch_seed: Seed for deterministic patch masking (None for random)
        """
        self.num_views = num_views
        self.backbone_unfreeze_step = backbone_unfreeze_step
        self.patch_seed = patch_seed
        
        # Parse patch masking configuration
        if patch_mask_config is None:
            patch_mask_config = {}
        
        self.patch_init_step = patch_mask_config.get("init_step", 700)
        self.patch_final_step = patch_mask_config.get("final_step", 5000)
        self.patch_init_ratio = patch_mask_config.get("init_ratio", 0.1)
        self.patch_final_ratio = patch_mask_config.get("final_ratio", 0.5)
        
        # Automatically enable patch masking if final_ratio > 0
        self.use_patch_masking = self.patch_final_ratio > 0.0
    
    def apply_patch_masking(self, images: torch.Tensor, training_step: int = 0, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random patch masking with curriculum learning."""
        if not is_training:  # Not in training mode
            batch_size, num_views, channels, height, width = images.shape
            device = images.device
            patch_size = 16
            num_patches_h = height // patch_size
            num_patches_w = width // patch_size
            total_patches_per_view = num_patches_h * num_patches_w
            patch_mask = torch.ones(batch_size, num_views, total_patches_per_view, device=device)
            return images, patch_mask
            
        batch_size, num_views, channels, height, width = images.shape
        device = images.device
        
        # Calculate current mask ratio using new configuration
        if training_step < self.patch_init_step:
            mask_ratio = 0.0
        else:
            curriculum_steps_for_patch = self.patch_final_step - self.patch_init_step
            progress = min((training_step - self.patch_init_step) / curriculum_steps_for_patch, 1.0)
            mask_ratio = self.patch_init_ratio + progress * (self.patch_final_ratio - self.patch_init_ratio)
        
        # Calculate patch dimensions
        patch_size = 16
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        total_patches_per_view = num_patches_h * num_patches_w
        patches_to_mask_per_view = int(mask_ratio * total_patches_per_view)
        
        # Initialize masks
        patch_mask = torch.ones(batch_size, num_views, total_patches_per_view, device=device)
        masked_images = images.clone()
        
        # Apply patch masking
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                if patches_to_mask_per_view > 0:
                    # Use seed for deterministic patch selection if provided
                    if self.patch_seed is not None:
                        # Simple approach: set global seed before each randperm call
                        torch.manual_seed(self.patch_seed + training_step + batch_idx + view_idx)
                    
                    # Random patch selection (same for both seeded and unseeded)
                    patch_indices = torch.randperm(total_patches_per_view, device=device)[:patches_to_mask_per_view]
                    
                    patch_mask[batch_idx, view_idx, patch_indices] = 0
                    
                    for patch_idx in patch_indices:
                        patch_h = (patch_idx // num_patches_w) * patch_size
                        patch_w = (patch_idx % num_patches_w) * patch_size
                        masked_images[batch_idx, view_idx, :, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = 0
        
        return masked_images, patch_mask
    
    def apply_masking(self, images: torch.Tensor, training_step: int = 0, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply patch masking if enabled, otherwise return original images."""
        if self.use_patch_masking:
            return self.apply_patch_masking(images, training_step, is_training)
        else:
            # No masking - return original images and dummy mask
            batch_size, num_views = images.shape[:2]
            device = images.device
            dummy_mask = torch.ones(batch_size, num_views, device=device)
            return images, dummy_mask
    
    def get_training_schedule_info(self, current_step: int) -> Dict[str, Any]:
        """Get information about current training schedule progress."""
        if self.use_patch_masking:
            if current_step < self.patch_init_step:
                current_mask_ratio = 0.0
                curriculum_progress = "0.0%"
                steps_to_patch_masking = self.patch_init_step - current_step
                steps_to_max_masking = self.patch_final_step - current_step
            else:
                curriculum_steps_for_patch = self.patch_final_step - self.patch_init_step
                progress = min((current_step - self.patch_init_step) / curriculum_steps_for_patch, 1.0)
                current_mask_ratio = self.patch_init_ratio + progress * (self.patch_final_ratio - self.patch_init_ratio)
                curriculum_progress = f"{progress*100:.1f}%"
                steps_to_patch_masking = 0
                steps_to_max_masking = max(0, self.patch_final_step - current_step)
        else:
            current_mask_ratio = 0.0
            curriculum_progress = "0.0%"
            steps_to_max_masking = 0
            steps_to_patch_masking = 0
        
        backbone_frozen = current_step < self.backbone_unfreeze_step
        
        return {
            "step": current_step,
            "mask_ratio": current_mask_ratio,
            "backbone_frozen": backbone_frozen,
            "curriculum_progress": curriculum_progress,
            "steps_to_unfreeze": max(0, self.backbone_unfreeze_step - current_step),
            "steps_to_patch_masking": steps_to_patch_masking,
            "steps_to_max_masking": steps_to_max_masking
        }
    
    def should_unfreeze_backbone(self, current_step: int) -> bool:
        """Check if backbone should be unfrozen at current step."""
        return current_step == self.backbone_unfreeze_step
    
    def should_start_patch_masking(self, current_step: int) -> bool:
        """Check if patch masking should start at current step."""
        return self.use_patch_masking and current_step == self.patch_init_step
