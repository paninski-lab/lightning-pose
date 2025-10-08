"""Load vision encoder from Facebook SAM model using HuggingFace."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel

# to ignore imports for sphix-autoapidoc
__all__ = []


class SamVisionEncoder(nn.Module):
    """Wrapper around HuggingFace's SAM Vision Encoder."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        finetune_img_size: int = 1024,
        img_size: int = 1024,
    ):
        super().__init__()

        # Load the full SAM model and extract vision encoder
        full_model = SamModel.from_pretrained(model_name)
        full_model = full_model.cpu()
        self.vision_encoder = full_model.vision_encoder

        # Store size information
        self.img_size = img_size
        self.finetune_img_size = finetune_img_size
        self.patch_size = full_model.config.vision_config.patch_size

        # Store original positional embeddings for potential resizing
        self.original_pos_embed = None
        if hasattr(self.vision_encoder, 'pos_embed'):
            self.original_pos_embed = self.vision_encoder.pos_embed.clone()

        # Check if we need to resize positional embeddings
        if (
            self.finetune_img_size != self.img_size
            and hasattr(self.vision_encoder, 'pos_embed')
            and self.vision_encoder.pos_embed is not None
        ):
            # Resize positional embeddings if needed
            print(
                f"Finetune image size ({finetune_img_size}) does not match model size ({img_size})"
                f" - recomputing position embeddings"
            )
            self._resize_pos_embed()

        # Bypass size check entirely
        self._bypass_size_check()

        # Disable relative positional encoding in SAM
        for layer in self.vision_encoder.layers:
            if hasattr(layer.attn, "use_rel_pos"):
                layer.attn.use_rel_pos = False

    def _bypass_size_check(self):
        """Completely bypass the size check in patch embedding"""

        def no_size_check_forward(pixel_values):
            batch_size, num_channels, height, width = pixel_values.shape

            # Only check channel dimension
            if num_channels != self.vision_encoder.patch_embed.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one "
                    "set in the configuration."
                )

            # Skip size check entirely - just do the convolution
            embeddings = self.vision_encoder.patch_embed.projection(
                pixel_values
            ).permute(0, 2, 3, 1)
            return embeddings

        # Replace the forward method
        self.vision_encoder.patch_embed.forward = no_size_check_forward
        print("Bypassed all size checking in patch_embed")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        This is mostly a copy of SamVisionEncoder.forward(), but without the neck.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Encoded features
        """
        # # Pass through the vision encoder
        # # HuggingFace expects pixel_values as input
        # outputs = self.vision_encoder(pixel_values=x)
        #
        # # Extract the last hidden state (features)
        # # HuggingFace returns a different format than Facebook's implementation
        # features = outputs.last_hidden_state

        # Patch embedding
        hidden_states = self.vision_encoder.patch_embed(x)

        # Add positional embeddings
        if self.vision_encoder.pos_embed is not None:
            hidden_states = hidden_states + self.vision_encoder.pos_embed

        # Transformer layers
        for i, layer_module in enumerate(self.vision_encoder.layers):
            if self.vision_encoder.gradient_checkpointing and self.vision_encoder.training:
                layer_outputs = self.vision_encoder._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=None)
            hidden_states = layer_outputs[0]

        # Reshape to [B, C, H, W]
        features = hidden_states.permute(0, 3, 1, 2)

        return features

    def _resize_pos_embed(self):
        """Resize positional embeddings for different input sizes"""

        if self.original_pos_embed is None:
            return

        # Calculate target size
        old_size = self.img_size // self.patch_size  # 1024 // 16 = 64
        new_size = self.finetune_img_size // self.patch_size  # 128 // 16 = 8

        if old_size == new_size:
            return

        print(f"Resizing pos_embed from {old_size}x{old_size} to {new_size}x{new_size}")

        # HuggingFace stores pos_embed in spatial format [1, H, W, C]
        pos_embed = self.original_pos_embed  # [1, 64, 64, 768]

        # Convert to [1, C, H, W] for interpolation
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, 768, 64, 64]

        # Resize using interpolation
        pos_embed_resized = F.interpolate(
            pos_embed,
            size=(new_size, new_size),  # (8, 8)
            mode='bicubic',
            antialias=True,
        )  # [1, 768, 8, 8]

        # Convert back to spatial format [1, H, W, C]
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1)  # [1, 8, 8, 768]

        # Update the vision encoder's positional embeddings
        self.vision_encoder.pos_embed = nn.Parameter(pos_embed_final)
