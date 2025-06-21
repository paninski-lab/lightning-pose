import math

import torch
import torch.nn as nn
from transformers import ViTMAEConfig, ViTMAEForPreTraining

# to ignore imports for sphix-autoapidoc
__all__ = [
    "ViTVisionEncoder",
]


class ViTVisionEncoder(nn.Module):
    """Wrapper around HuggingFace's ViTMAE Vision Encoder."""

    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        finetune_img_size: int = 256,
    ):
        super().__init__()

        if model_name == "facebook/vit-mae-base":
            img_size = 224
            config = {
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12,
                'intermediate_size': 3072,
                'hidden_act': "gelu",
                'hidden_dropout_prob': 0.0,
                'attention_probs_dropout_prob': 0.0,
                'initializer_range': 0.02,
                'layer_norm_eps': 1.e-12,
                'image_size': img_size,  # usually 224
                'patch_size': 16,   # default is 16, we use large patch size
                'num_channels': 3,  # 3 for RGB
                'qkv_bias': True,
                'decoder_num_attention_heads': 16,
                'decoder_hidden_size': 512,
                'decoder_num_hidden_layers': 8,
                'decoder_intermediate_size': 2048,
                'mask_ratio': 0,  # 0 for no masking, usually 0.75 (MAE)
                'norm_pix_loss': False,
            }
        else:
            raise NotImplementedError(f"{model_name} is not a valid ViTVisionEncoder model name")

        # Load the full ViT model and extract encoder
        self.config = ViTMAEConfig(**config)
        self.vision_encoder = ViTMAE.from_pretrained(model_name)
        del self.vision_encoder.decoder  # remove the decoder from the vit_mae
        self.vision_encoder.config.mask_ratio = 0

        # Store size information
        self.img_size = img_size
        self.finetune_img_size = finetune_img_size
        self.patch_size = self.vision_encoder.config.patch_size

        # Store original positional embeddings for potential resizing
        self.original_pos_embed = None
        if hasattr(self.vision_encoder.vit.embeddings, 'position_embeddings'):
            self.original_pos_embed = \
                self.vision_encoder.vit.embeddings.position_embeddings.clone()

        # Check if we need to resize positional embeddings
        if (
            self.finetune_img_size != img_size
            and hasattr(self.vision_encoder.vit.embeddings, 'position_embeddings')
            and self.vision_encoder.vit.embeddings.position_embeddings is not None
        ):
            # Resize positional embeddings if needed
            print(
                f"Finetune image size ({finetune_img_size}) does not match model size ({img_size})"
                f" - recomputing position embeddings"
            )
            self._resize_pos_embed()

        # Bypass size check entirely
        self._bypass_size_check()

    def _bypass_size_check(self):
        """Completely bypass the size check in patch embedding"""

        def no_size_check_forward(pixel_values, interpolate_pos_encoding: bool = False):
            batch_size, num_channels, height, width = pixel_values.shape

            # Only check channel dimension
            if num_channels != self.vision_encoder.vit.config.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one "
                    "set in the configuration."
                )

            # Skip size check entirely - just do the convolution
            embeddings = self.vision_encoder.vit.embeddings.patch_embeddings.projection(
                pixel_values
            ).flatten(2).transpose(1, 2)
            return embeddings

        # Replace the forward method
        self.vision_encoder.vit.embeddings.patch_embeddings.forward = no_size_check_forward
        print("Bypassed all size checking in embeddings")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        if self.config.num_channels == 1:
            # adjust input channels to 1
            x = x[:, 0, ...].unsqueeze(1)
        outputs = self.vision_encoder(
            pixel_values=x,
            return_latent=True,
        )
        # skip the cls token
        outputs = outputs[:, 1:, ...]  # [N, S, D]
        # change the shape to [N, H, W, D] -> [N, D, H, W]
        S = outputs.shape[1]
        H, W = math.isqrt(S), math.isqrt(S)
        outputs = outputs.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        return outputs

    def _resize_pos_embed(self):
        """Resize positional embeddings for different input sizes"""

        if self.original_pos_embed is None:
            return

        # Calculate target size
        old_size = self.img_size // self.patch_size  # 224 // 16 = 14
        new_size = self.finetune_img_size // self.patch_size  # 128 // 16 = 8

        if old_size == new_size:
            return

        print(f"Resizing pos_embed from {old_size}x{old_size} to {new_size}x{new_size}")

        # Original pos_embed format: [1, H*W + 1, C]
        pos_embed = self.original_pos_embed  # [1, 197, 768] for 224x224 input

        # Separate CLS token embedding from spatial embeddings
        cls_token_embed = pos_embed[:, 0:1, :]  # [1, 1, 768] - CLS token
        spatial_embeddings = pos_embed[:, 1:, :]  # [1, 196, 768] - spatial patches

        # Reshape spatial embeddings to 2D spatial format
        # [1, H*W, C] -> [1, H, W, C]
        batch_size, num_patches, embed_dim = spatial_embeddings.shape
        spatial_2d = spatial_embeddings.reshape(batch_size, old_size, old_size, embed_dim)

        # Convert to [1, C, H, W] for interpolation
        spatial_2d = spatial_2d.permute(0, 3, 1, 2)  # [1, 768, 14, 14]

        # Resize using interpolation
        spatial_resized = nn.functional.interpolate(
            spatial_2d,
            size=(new_size, new_size),  # (8, 8)
            mode='bicubic',
            antialias=True,
        )  # [1, 768, 8, 8]

        # Convert back to sequence format
        # [1, C, H, W] -> [1, H, W, C] -> [1, H*W, C]
        spatial_resized = spatial_resized.permute(0, 2, 3, 1)  # [1, 8, 8, 768]
        spatial_resized = spatial_resized.reshape(batch_size, new_size * new_size, embed_dim)

        # Concatenate CLS token back at the beginning
        pos_embed_final = torch.cat([cls_token_embed, spatial_resized], dim=1)  # [1, 65, 768]

        # Update the position embeddings
        self.vision_encoder.vit.embeddings.position_embeddings = nn.Parameter(pos_embed_final)


class ViTMAE(ViTMAEForPreTraining):

    def forward(
            self,
            pixel_values,
            noise=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            interpolate_pos_encoding=False,
            return_latent=False,
            return_recon=False,
    ):
        # Setting default for return_dict based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (self.training and self.config.mask_ratio > 0) or return_recon:
            outputs = self.vit(
                pixel_values,
                noise=noise,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            latent = outputs.last_hidden_state
        else:
            # use for fine-tuning, or inference
            # mask_ratio = 0
            embedding_output, mask, ids_restore = self.vit.embeddings(pixel_values)
            embedding_output_ = embedding_output[:, 1:, :]  # no cls token
            # unshuffle the embedding output
            embedding_output_ = torch.gather(
                embedding_output_,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(
                    1, 1, embedding_output_.shape[2]
                ).to(embedding_output_.device))
            # add cls token back
            embedding_output = torch.cat((embedding_output[:, :1, :], embedding_output_), dim=1)
            encoder_outputs = self.vit.encoder(
                embedding_output,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            latent = self.vit.layernorm(sequence_output)
            if not return_latent:
                # return the cls token and 0 loss if not return_latent
                return latent[:, 0], 0

        if return_latent:
            return latent

        # extract cls latent
        cls_latent = latent[:, 0]  # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        logits = decoder_outputs.logits
        # print(decoder_outputs.keys())
        loss = self.forward_loss(pixel_values, logits, mask)
        if return_recon:
            return cls_latent, loss, logits

        return cls_latent, loss
