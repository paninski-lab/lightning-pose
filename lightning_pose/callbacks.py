from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

# to ignore imports for sphix-autoapidoc
__all__ = [
    "AnnealWeight",
    "UnfreezeBackbone",
    "PatchMasking",
]


class AnnealWeight(Callback):
    """Callback to change weight value during training."""

    def __init__(
        self,
        attr_name: str,
        init_val: float = 0.0,
        increase_factor: float = 0.01,
        final_val: float = 1.0,
        freeze_until_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.init_val = init_val
        self.increase_factor = increase_factor
        self.final_val = final_val
        self.freeze_until_epoch = freeze_until_epoch
        self.attr_name = attr_name

    def on_train_start(self, trainer, pl_module) -> None:
        # Dan: removed buffer; seems to complicate checkpoint loading
        # pl_module.register_buffer(self.attr_name, torch.tensor(self.init_val))
        setattr(pl_module, self.attr_name, torch.tensor(self.init_val))

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.current_epoch <= self.freeze_until_epoch:
            pass
        else:
            eff_epoch: int = pl_module.current_epoch - self.freeze_until_epoch
            value: float = min(
                self.init_val + eff_epoch * self.increase_factor, self.final_val
            )
            # Dan: removed buffer; seems to complicate checkpoint loading
            # pl_module.register_buffer(self.attr_name, torch.tensor(value))
            setattr(pl_module, self.attr_name, torch.tensor(value))


class UnfreezeBackbone(Callback):
    """Callback that ramps up the backbone learning rate from 0 to `upsampling_lr` on
    `unfreeze_epoch` or `unfreeze_step`.

    Starts LR at `initial_ratio * upsampling_lr`. Grows lr by a factor of `warm_up_ratio` per
    epoch or step. Once LR reaches `upsampling_lr`, keeps it in sync with `upsampling_lr`.

    Use instead of pl.callbacks.BackboneFinetuning in order to use multi-GPU (DDP). See
    lightning-ai/pytorch-lightning#20340 for context.
    """

    def __init__(
        self,
        unfreeze_epoch: int | None = None,
        unfreeze_step: int | None = None,
        initial_ratio=0.1,
        warm_up_ratio=1.5,
    ):
        assert (unfreeze_epoch is None) != (
            unfreeze_step is None
        ), "Exactly one must be provided."
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_step = unfreeze_step
        self.initial_ratio = initial_ratio
        self.warm_up_ratio = warm_up_ratio
        self._warmed_up = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:

        # Once backbone_lr warms up to upsampling_lr, this callback does nothing.
        # Control of backbone lr is then the sole job of the main lr scheduler.
        if self._warmed_up:
            return

        optimizer = pl_module.optimizers()
        # Check our assumptions about param group indices
        assert optimizer.param_groups[0]["name"] == "backbone"

        head_lr = optimizer.param_groups[1]["lr"]

        optimizer.param_groups[0]["lr"] = self._get_backbone_lr(
            pl_module.global_step, pl_module.current_epoch, head_lr
        )

    def _get_backbone_lr(self, current_step, current_epoch, upsampling_lr):
        """Returns what the backbone LR should be at this point in time.

        Args:
            Only one of `current_step` and `current_epoch` will be used.
            If self.unfreeze_epoch is not None, then we'll use `current_epoch`
            Otherwise, unfreeze_step is not None and we'll use `current_step`.
        """
        assert not self._warmed_up

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # In the code below, the variables are named in terms of epoch,
        # but the same logic applies for steps, conveniently enough.
        # So if we're in "step mode", plug in steps into epoch variables.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        unfreeze_epoch = self.unfreeze_epoch
        if self.unfreeze_step is not None:
            unfreeze_epoch = self.unfreeze_step
            current_epoch = current_step
        # After this point, use `unfreeze_epoch` instead of `self.unfreeze_[epoch|step]`.
        # Main logic begins:

        # Before unfreeze, learning_rate is 0.
        if current_epoch < unfreeze_epoch:
            return 0.0

        # On unfreeze, initialize learning rate.
        # Remember this initial value for warm up.
        if current_epoch == unfreeze_epoch:
            self._initial_lr = self.initial_ratio * upsampling_lr
            return self._initial_lr

        # Warm up: compute inital_ratio * epoch_ratio ** epochs_since_thaw.
        # Use stored initial_ratio rather than recomputing it since
        # upsampling_lr is subject to change via the scheduler.
        if current_epoch > unfreeze_epoch:
            epochs_since_thaw = current_epoch - unfreeze_epoch
            next_lr = min(
                self._initial_lr * self.warm_up_ratio**epochs_since_thaw, upsampling_lr
            )
            if next_lr == upsampling_lr:
                self._warmed_up = True
            return next_lr


class PatchMasking(Callback):
    """Callback to apply curriculum patch masking during training."""

    def __init__(
        self,
        patch_mask_config: dict = None,
        patch_seed: int = 0,
    ):
        super().__init__()

        # Initialize curriculum masking
        self.curriculum_masking = PatchMasker(
            patch_mask_config=patch_mask_config,
            patch_seed=patch_seed,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Apply patch masking to the batch before it goes to the model."""
        if not self.curriculum_masking.use_patch_masking:
            return

        # Extract images from batch
        if isinstance(batch, dict):
            if "images" in batch:
                images = batch["images"]
            elif "frames" in batch:
                images = batch["frames"]
            else:
                return
        else:
            # Handle case where batch is just images
            images = batch

        # Apply masking
        masked_images, patch_mask = self.curriculum_masking.apply_patch_masking(
            images,
            training_step=trainer.global_step,
            is_training=True,
        )

        # Update the batch with masked images
        if isinstance(batch, dict):
            if "images" in batch:
                batch["images"] = masked_images
            elif "frames" in batch:
                batch["frames"] = masked_images
        else:
            # Replace the batch entirely
            batch = masked_images

        # Store patch mask for potential use in loss computation
        pl_module.current_patch_mask = patch_mask

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Log curriculum progress."""
        if not self.curriculum_masking.use_patch_masking:
            return

        schedule_info = self.curriculum_masking.get_training_schedule_info(trainer.global_step)

        # Log curriculum information - only log numeric values
        pl_module.log(
            "patch_mask_ratio", schedule_info['mask_ratio'],
            on_step=False, on_epoch=True, prog_bar=True,
        )


class PatchMasker:
    """Handles curriculum learning and masking for multiview transformer training."""

    def __init__(
        self,
        patch_mask_config: dict | None = None,
        patch_seed: int = 0,
    ):
        """Initialize curriculum masking parameters.

        Args:
            patch_mask_config: Dictionary containing patch masking configuration
                - init_step: Step to start patch masking
                - final_step: Step when patch masking reaches maximum
                - init_ratio: Initial masking ratio
                - final_ratio: Final masking ratio
            patch_seed: Seed for deterministic patch masking to allow reproducibility.
        """
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

        # Validate patch_seed is set for reproducibility
        if self.use_patch_masking and patch_seed is None:
            print(
                "Warning: patch_seed is None but patch masking is enabled. "
                "Results may not be reproducible."
            )

    def apply_patch_masking(
        self,
        images: torch.Tensor,
        training_step: int = 0,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random patch masking with curriculum learning."""

        # during training, apply masking
        batch_size, num_views, channels, height, width = images.shape
        device = images.device

        patch_size = 16
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        if not is_training:  # Not in training mode
            # Calculate patch size (assuming 16x16 patches for ViT)
            total_patches_per_view = num_patches_h * num_patches_w

            # Create patch mask with all patches kept (1)
            patch_mask = torch.ones(batch_size, num_views, total_patches_per_view, device=device)
            return images, patch_mask

        # Calculate current mask ratio
        # start with no masking until patch_init_step
        if training_step < self.patch_init_step:
            mask_ratio = 0.0
        else:
            # start patch masking at patch_init_step, reach max by patch_final_step
            curr_steps_for_patch = self.patch_final_step - self.patch_init_step
            progress = min((training_step - self.patch_init_step) / curr_steps_for_patch, 1.0)
            mask_ratio = (
                self.patch_init_ratio
                + progress * (self.patch_final_ratio - self.patch_init_ratio)
            )

        # Calculate patch dimensions (assuming 16x16 patches for ViT)
        total_patches_per_view = num_patches_h * num_patches_w
        patches_to_mask_per_view = int(mask_ratio * total_patches_per_view)

        # Initialize masks
        patch_mask = torch.ones(batch_size, num_views, total_patches_per_view, device=device)
        masked_images = images.clone()

        # Apply patch masking per batch sample and view
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                if patches_to_mask_per_view > 0:
                    # Create a deterministic seed for this specific combination
                    # Same patches are masked for the same training step, batch, and view
                    # Using multiplication to avoid seed collisions between different combinations
                    deterministic_seed = (
                        self.patch_seed
                        + training_step
                        + batch_idx * 1000
                        + view_idx * 100
                    )
                    # Create a local generator instead of modifying global torch seed
                    local_generator = torch.Generator(device=device)
                    local_generator.manual_seed(deterministic_seed)

                    # Random patch selection with local generator
                    patch_indices = torch.randperm(
                        total_patches_per_view,
                        device=device,
                        generator=local_generator
                    )[:patches_to_mask_per_view]
                    patch_mask[batch_idx, view_idx, patch_indices] = 0
                    # Zero out the selected patches
                    for patch_idx in patch_indices:
                        # Convert patch index to spatial coordinates
                        patch_h = (patch_idx // num_patches_w) * patch_size
                        patch_w = (patch_idx % num_patches_w) * patch_size
                        # Zero out the patch region
                        masked_images[
                            batch_idx,
                            view_idx,
                            :,
                            patch_h:patch_h + patch_size,
                            patch_w:patch_w + patch_size
                        ] = 0

        return masked_images, patch_mask

    def apply_masking(
        self,
        images: torch.Tensor,
        training_step: int = 0,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply patch masking if enabled, otherwise return original images."""
        if self.use_patch_masking:
            return self.apply_patch_masking(images, training_step, is_training)
        else:
            # No masking - return original images and dummy mask
            batch_size, num_views = images.shape[:2]
            device = images.device
            dummy_mask = torch.ones(batch_size, num_views, device=device)
            return images, dummy_mask

    def get_training_schedule_info(
        self,
        current_step: int
    ) -> Dict[str, Any]:
        """Get information about current training schedule progress."""
        if self.use_patch_masking:
            if current_step < self.patch_init_step:
                current_mask_ratio = 0.0
                curriculum_progress = "0.0%"
                steps_to_patch_masking = self.patch_init_step - current_step
                steps_to_max_masking = self.patch_final_step - current_step
            else:
                curr_steps_for_patch = self.patch_final_step - self.patch_init_step
                progress = min((current_step - self.patch_init_step) / curr_steps_for_patch, 1.0)
                current_mask_ratio = (
                    self.patch_init_ratio
                    + progress * (self.patch_final_ratio - self.patch_init_ratio)
                )
                curriculum_progress = f"{progress*100:.1f}%"
                steps_to_patch_masking = 0
                steps_to_max_masking = max(0, self.patch_final_step - current_step)
        else:
            current_mask_ratio = 0.0
            curriculum_progress = "0.0%"
            steps_to_max_masking = 0
            steps_to_patch_masking = 0

        return {
            "step": current_step,
            "mask_ratio": current_mask_ratio,
            "curriculum_progress": curriculum_progress,
            "steps_to_patch_masking": steps_to_patch_masking,
            "steps_to_max_masking": steps_to_max_masking
        }

    def should_start_patch_masking(
        self,
        current_step: int
    ) -> bool:
        """Check if patch masking should start at current step."""
        return self.use_patch_masking and current_step == self.patch_init_step
