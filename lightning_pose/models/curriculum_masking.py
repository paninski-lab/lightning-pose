"""Utility classes for curriculum learning and masking in multiview transformer models."""

import torch
from typing import Literal, Tuple, Dict, Any


class CurriculumMasking:
    """Handles curriculum learning and masking for multiview transformer training."""
    
    def __init__(
        self,
        num_views: int,
        masking_type: Literal["view", "patch", "none"] = "patch",
        backbone_unfreeze_step: int = 400,
        curriculum_steps: int = 5000,
        patch_masking_delay: int = 300,
        initial_mask_ratio: float = 0.1,
        max_mask_ratio: float = 0.5,
    ):
        """Initialize curriculum masking parameters.
        
        Args:
            num_views: Number of camera views
            masking_type: Type of masking to apply
            backbone_unfreeze_step: Step to unfreeze backbone
            curriculum_steps: Total steps for curriculum learning
            patch_masking_delay: Delay before starting patch masking
            initial_mask_ratio: Starting masking ratio
            max_mask_ratio: Maximum masking ratio
        """
        self.num_views = num_views
        self.masking_type = masking_type
        self.backbone_unfreeze_step = backbone_unfreeze_step
        self.curriculum_steps = curriculum_steps
        self.patch_masking_delay = patch_masking_delay
        self.initial_mask_ratio = initial_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        
        self.use_view_masking = masking_type == "view"
        self.use_patch_masking = masking_type == "patch"
    
    def apply_view_masking(self, images: torch.Tensor, training_step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random view masking with curriculum learning."""
        if not images.requires_grad:  # Not in training mode
            batch_size = images.shape[0]
            view_mask = torch.ones(batch_size, self.num_views, device=images.device)
            return images, view_mask
            
        batch_size, num_views, channels, height, width = images.shape
        device = images.device
        
        # Calculate current mask ratio
        progress = min(training_step / self.curriculum_steps, 1.0)
        mask_ratio = self.initial_mask_ratio + progress * (self.max_mask_ratio - self.initial_mask_ratio)
        
        # Calculate number of views to mask
        max_masked_views = max(0, num_views - 1)
        num_views_to_mask = int(min(mask_ratio * num_views, max_masked_views))
        
        # Initialize masks
        view_mask = torch.ones(batch_size, num_views, device=device)
        masked_images = images.clone()
        
        # Apply masking per batch sample
        for batch_idx in range(batch_size):
            if num_views_to_mask > 0:
                view_indices = torch.randperm(num_views, device=device)[:num_views_to_mask]
                view_mask[batch_idx, view_indices] = 0
                for view_idx in view_indices:
                    masked_images[batch_idx, view_idx] = 0
        
        return masked_images, view_mask
    
    def apply_patch_masking(self, images: torch.Tensor, training_step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random patch masking with curriculum learning."""
        if not images.requires_grad:  # Not in training mode
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
        
        # Calculate current mask ratio
        patch_masking_start_step = self.backbone_unfreeze_step + self.patch_masking_delay
        
        if training_step < patch_masking_start_step:
            mask_ratio = 0.0
        else:
            curriculum_steps_for_patch = self.curriculum_steps - patch_masking_start_step
            progress = min((training_step - patch_masking_start_step) / curriculum_steps_for_patch, 1.0)
            mask_ratio = self.initial_mask_ratio + progress * (self.max_mask_ratio - self.initial_mask_ratio)
        
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
                    patch_indices = torch.randperm(total_patches_per_view, device=device)[:patches_to_mask_per_view]
                    patch_mask[batch_idx, view_idx, patch_indices] = 0
                    
                    for patch_idx in patch_indices:
                        patch_h = (patch_idx // num_patches_w) * patch_size
                        patch_w = (patch_idx % num_patches_w) * patch_size
                        masked_images[batch_idx, view_idx, :, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = 0
        
        return masked_images, patch_mask
    
    def apply_masking(self, images: torch.Tensor, training_step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply appropriate masking based on masking type."""
        if self.use_view_masking:
            return self.apply_view_masking(images, training_step)
        elif self.use_patch_masking:
            return self.apply_patch_masking(images, training_step)
        else:
            # No masking
            batch_size, num_views = images.shape[:2]
            device = images.device
            dummy_mask = torch.ones(batch_size, num_views, device=device)
            return images, dummy_mask
    
    def get_training_schedule_info(self, current_step: int) -> Dict[str, Any]:
        """Get information about current training schedule progress."""
        if self.masking_type == "view":
            progress = min(current_step / self.curriculum_steps, 1.0)
            current_mask_ratio = self.initial_mask_ratio + progress * (self.max_mask_ratio - self.initial_mask_ratio)
            curriculum_progress = f"{progress*100:.1f}%"
            steps_to_max_masking = max(0, self.curriculum_steps - current_step)
            steps_to_patch_masking = 0
        elif self.masking_type == "patch":
            patch_masking_start_step = self.backbone_unfreeze_step + self.patch_masking_delay
            
            if current_step < patch_masking_start_step:
                current_mask_ratio = 0.0
                curriculum_progress = "0.0%"
                steps_to_patch_masking = patch_masking_start_step - current_step
                steps_to_max_masking = self.curriculum_steps - current_step
            else:
                curriculum_steps_for_patch = self.curriculum_steps - patch_masking_start_step
                progress = min((current_step - patch_masking_start_step) / curriculum_steps_for_patch, 1.0)
                current_mask_ratio = self.initial_mask_ratio + progress * (self.max_mask_ratio - self.initial_mask_ratio)
                curriculum_progress = f"{progress*100:.1f}%"
                steps_to_patch_masking = 0
                steps_to_max_masking = max(0, self.curriculum_steps - current_step)
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
        patch_masking_start_step = self.backbone_unfreeze_step + self.patch_masking_delay
        return self.masking_type == "patch" and current_step == patch_masking_start_step
