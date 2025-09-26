"""Heads that produce heatmap predictions for heatmap regression."""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.subpix import spatial_softmax2d
from torch import Tensor, nn
from torchtyping import TensorType
from typing_extensions import Literal

from lightning_pose.models.heads import HeatmapHead
from lightning_pose.models.heads.heatmap import run_subpixelmaxima

# to ignore imports for sphix-autoapidoc
__all__ = [
    "MultiviewFeatureTransformerHead",
    "MultiviewFeatureTransformerHeadLearnableCrossView",
]


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_view_embed(n_views, grid_size):

    emb = np.zeros((n_views, grid_size * grid_size, 4))
    for view in range(n_views):
        view_binary = bin(view)[2:].zfill(4)  # [2:] gets rid of "0b" prefix
        for i in range(4):
            emb[view, :, i] = int(view_binary[i])

    return emb


class MultiviewFeatureTransformerHead(nn.Module):
    """Multi-view transformer neural network head that operates on feature maps.

    This head takes a set of 2D feature maps corresponding to different views, and fuses them
    together using a transformer architecture. Each token represents a spatial feature output by
    the backbone for a single view, along with a positional embedding and view embedding.

    """

    def __init__(
        self,
        backbone_arch: str,
        num_views: int,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        transformer_d_model: int = 512,
        transformer_nhead: int = 8,
        transformer_dim_feedforward: int = 512,
        transformer_num_layers: int = 4,
        img_size: int = 256,
    ):
        """

        Args:
            backbone_arch: string denoting backbone architecture; to remove in future release
            num_views: number of camera views in each batch
            in_channels: number of channels in the input feature map
            out_channels: number of channels in the output heatmap (i.e. number of keypoints)
            deconv_out_channels: output channel number for each intermediate deconv layer; defaults
                to number of keypoints
            downsample_factor: make heatmaps smaller than input frames by this factor; subpixel
                operations are performed for increased precision

        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor
        self.temperature = torch.tensor(1000.0)  # soft argmax temp

        self.transformer_d_model = transformer_d_model

        # create tokenizer: map from n_features of backbone to n_features of transformer
        self.tokenize = torch.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=transformer_d_model - 4, 
            kernel_size=1,
        )

        # build position embedding
        if img_size == 128:
            grid_size = 4
        elif img_size == 256:
            grid_size = 8
        elif img_size == 384:
            grid_size = 12

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=transformer_d_model - 4,  # d_model will include concat view embeds later
            grid_size=grid_size, 
            add_cls_token=False,
        )
        # NOTE: fix device for multi-gpu
        self.pos_embed = torch.tensor(pos_embed, dtype=torch.float, device='cuda:0').unsqueeze(0)

        # build view embedding; shape (n_views, grid_size^2, 4)
        view_embed = get_view_embed(n_views=num_views, grid_size=grid_size)
        # NOTE: fix device for multi-gpu
        self.view_embed = torch.tensor(view_embed, dtype=torch.float, device='cuda:0')

        # build transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_d_model,  # embedding dim + view embeddings 
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=0.1,
            layer_norm_eps=1e-05, # originally 1e-05
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers= transformer_num_layers,
        )


        # create upsampling head
        self.upsample = HeatmapHead(
            backbone_arch='resnet',
            in_channels=transformer_d_model,
            out_channels=out_channels,
            downsample_factor=2,
            final_softmax=True,
        )

        print(f" the number of transformer layers is {transformer_num_layers} ")
        

    def forward(
        self,
        features: TensorType["view x batch", "features", "rep_height", "rep_width"],
        num_views: int, # was torch.tensor before
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Upsample and run multiview head to get final heatmaps.

        Args:
            features: outputs of backbone
            num_views: number of camera views for each batch element

        """

        batch_size_combined = features.shape[0]
        n_batch = int(batch_size_combined // num_views)
        n_width = features.shape[-1]
        n_height = features.shape[-2]

        # reduce dimensionality of input features, e.g., 2048->512
        tokens = self.tokenize(features)
        # tokens = [view * batch, num_embed, rep_height, rep_width]
        # assert tokens.shape == (n_views * n_batch, n_embed, n_height, n_width)

        # reshape tokens for position embeddings
        tokens = tokens.reshape(num_views * n_batch, self.transformer_d_model - 4, -1)
        # tokens = [view * batch, num_embed, rep_height * rep_width]
        tokens = tokens.permute((0, 2, 1))
        # tokens = [view * batch, rep_height * rep_width, num_embed]
        # assert tokens.shape == (n_views * n_batch, n_height * n_width, n_embed)

        # add position embeddings to tokens
        tokens = tokens + self.pos_embed
        # tokens = [view * batch, rep_height * rep_width, num_embed]
        # assert tokens.shape == (n_views * n_batch, n_height * n_width, n_embed)

        # add view embeddings to tokens
        # view_embed shape = [n_views, grid_size^2, 4]
        view_embed_batch = self.view_embed.unsqueeze(0).repeat(
            n_batch, 1, 1, 1
        ).reshape((num_views * n_batch, -1, 4))
        tokens = torch.cat([tokens, view_embed_batch], dim=-1)
        # assert tokens.shape == (n_views * n_batch, n_height * n_width, n_embed + 4)  # view embeddings concatenated

        # reshape tokens and feed to transformer
        tokens = tokens.reshape((n_batch, num_views * n_height * n_width, self.transformer_d_model))
        embeddings = self.transformer_encoder(tokens)
        # embeddings = [batch, n_views * n_height * n_width, n_embed + 4]
        # assert embeddings.shape == (n_batch, n_views * n_height * n_width, n_embed + 4)  # view embeddings concatenated

        # reshape embeddings and feed into HeatmapHead
        embeddings = embeddings.reshape((n_batch, num_views, n_height, n_width, self.transformer_d_model))
        embeddings = embeddings.permute((0, 1, 4, 2, 3))
        # embeddings = [batch, n_views, n_embed + 4, n_height, n_width]
        embeddings = embeddings.reshape((-1, self.transformer_d_model, n_height, n_width))
        heatmaps = self.upsample(embeddings)
        # # heatmaps = [batch * n_views, n_keypoints, heat_height, heat_width]
        # assert heatmaps.shape == (n_batch * n_views, n_keypoints, n_height * 8, n_width * 8)
        heatmaps = heatmaps.reshape(n_batch, num_views, self.out_channels, n_height * 8, n_width * 8)
        heatmaps = heatmaps.reshape(n_batch, -1, n_height * 8, n_width * 8)

        return heatmaps

    def run_subpixelmaxima(self, heatmaps):
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


class MultiviewFeatureTransformerHeadLearnableCrossView(nn.Module):
    """Multi-view transformer neural network head with learnable view embeddings and specialized cross-view fusion.
    
    This head takes a set of 2D feature maps corresponding to different views, and fuses them
    together using a transformer architecture optimized for efficient cross-view fusion with a small
    number of layers (2 layers by default). Each token represents a spatial feature output by the backbone 
    for a single view, along with a positional embedding and learnable view embedding.
    """

    def __init__(
        self,
        backbone_arch: str,
        num_views: int,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: int | None = None,
        downsample_factor: int = 2,
        transformer_d_model: int = 512,
        transformer_nhead: int = 8,
        transformer_dim_feedforward: int = 512,  # Moderate increase from 512
        transformer_num_layers: int = 3,          # Keep at 2 as requested
        img_size: int = 256,
        view_embed_dim: int = 128,                # Keep at 64 for compatibility
        dropout: float = 0.1,
    ):
        """
        Args:
            backbone_arch: string denoting backbone architecture; to remove in future release
            num_views: number of camera views in each batch
            in_channels: number of channels in the input feature map
            out_channels: number of channels in the output heatmap (i.e. number of keypoints)
            deconv_out_channels: output channel number for each intermediate deconv layer
            downsample_factor: make heatmaps smaller than input frames by this factor
            transformer_d_model: dimension of transformer model
            transformer_nhead: number of attention heads in transformer
            transformer_dim_feedforward: dimension of feedforward network in transformer
            transformer_num_layers: number of transformer layers
            img_size: input image size
            view_embed_dim: dimension of view embeddings
            dropout: dropout rate
        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor  # Use the parameter value instead of hardcoding
        self.temperature = nn.Parameter(torch.tensor(1000.0))  # Make temperature a parameter to avoid in-place ops
        self.num_views = num_views
        self.view_embed_dim = view_embed_dim
        self.transformer_d_model = transformer_d_model

        # Create improved tokenizer with batch normalization for better gradient flow
        self.tokenize = nn.Sequential(
            nn.Conv2d(in_channels, transformer_d_model, kernel_size=1),
            nn.BatchNorm2d(transformer_d_model),
            nn.ReLU(),
        )

        # Store img_size for dynamic positional embedding creation
        self.img_size = img_size
            
        # Create position embeddings - will be created dynamically in forward pass
        self.pos_embed = None

        # Create learnable view embeddings
        self.view_embed = nn.Parameter(
            torch.randn(num_views, view_embed_dim) * 0.02  # Small initialization for stability
        )

        # View embedding projection
        self.view_projection = nn.Sequential(
            nn.Linear(view_embed_dim, transformer_d_model // 2),  # Smaller projection
            nn.LayerNorm(transformer_d_model // 2),
            nn.ReLU(),
            nn.Linear(transformer_d_model // 2, transformer_d_model),
        )

        # Layer normalization for combining embeddings
        self.embed_norm = nn.LayerNorm(transformer_d_model)
        self.view_projection = nn.Linear(view_embed_dim, transformer_d_model)
        # notice that in the above I removed the embed norm. maybe it will have an influence but we will see later
        
        
        # Create a global view context token for each view
        self.view_context_token = nn.Parameter(
            torch.zeros(1, 1, transformer_d_model)
        )
        nn.init.normal_(self.view_context_token, std=0.02)
        
        # Cross-view attention mechanism (separate from the transformer)
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=transformer_d_model,
            num_heads=transformer_nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_view_norm1 = nn.LayerNorm(transformer_d_model)
        self.cross_view_norm2 = nn.LayerNorm(transformer_d_model)
        
        # Cross-view fusion MLP
        self.cross_view_mlp = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim_feedforward, transformer_d_model),
            nn.Dropout(dropout)
        )

        # Build transformer - make each layer more powerful
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            activation="gelu",  # Use GELU instead of ReLU
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=True,  # Use Pre-LN for better training stability
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_num_layers,
        )

        # Create upsampling head
        self.upsample = HeatmapHead(
            backbone_arch='vit',  # Changed to 'vit' since we're using ViT backbone
            in_channels=transformer_d_model,
            out_channels=out_channels,
            downsample_factor=1,  # Use 1 to get 3 layers: 16->32->64->128
            final_softmax=True,
        )

        print(f"Using CrossView Transformer with {transformer_num_layers} layers")
        print(f"Transformer dimensions: d_model={transformer_d_model}, feedforward={transformer_dim_feedforward}")
        print(f"Using learnable view embeddings with {num_views} views, dimension {view_embed_dim}")
        print(f"Using upsampling head: {self.upsample.__class__.__name__}")

    def forward(
        self,
        features,
        num_views,
    ):
        """Upsample and run multiview head to get final heatmaps.

        Args:
            features: outputs of backbone
            num_views: number of camera views for each batch element
        """
        batch_size_combined = features.shape[0]
        n_batch = int(batch_size_combined // num_views)
        n_width = features.shape[-1]
        n_height = features.shape[-2]

        # Tokenize features (reduce dimensionality)
        tokens = self.tokenize(features)  # [view*batch, d_model, height, width]
        
        # Reshape for adding position embeddings
        tokens = tokens.reshape(num_views * n_batch, self.transformer_d_model, -1)
        tokens = tokens.permute(0, 2, 1)  # [view*batch, height*width, d_model]
        
        # Create positional embeddings dynamically based on actual feature map size
        if self.pos_embed is None or self.pos_embed.shape[1] != n_height * n_width:
            print(f"Creating positional embeddings for feature map size: {n_height}x{n_width} = {n_height * n_width} positions")
            # Create positional embeddings for the actual feature map dimensions
            # Use the same format as the original get_2d_sincos_pos_embed function
            grid_h = np.arange(n_height, dtype=np.float32)
            grid_w = np.arange(n_width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)  # here w goes first
            grid = np.stack(grid, axis=0)
            grid = grid.reshape([2, 1, n_height, n_width])
            
            pos_embed = get_2d_sincos_pos_embed_from_grid(
                embed_dim=self.transformer_d_model,
                grid=grid
            )
            self.pos_embed = torch.tensor(pos_embed, dtype=torch.float, device=tokens.device).unsqueeze(0)
            print(f"Positional embedding shape: {self.pos_embed.shape}")
        
        # Add position embeddings (avoiding in-place operations)
        tokens = tokens + self.pos_embed
        
        # Get view embeddings
        view_indices = torch.arange(num_views, device=tokens.device).repeat_interleave(n_batch)
        view_embeds = self.view_embed[view_indices]  # [view*batch, view_embed_dim]
        
        # Project view embeddings to transformer dimension
        view_embeds = self.view_projection(view_embeds)  # [view*batch, d_model]
        
        # Add context token to each view sequence
        context_tokens = self.view_context_token.expand(num_views * n_batch, -1, -1)
        
        # Expand view embeddings for context token
        view_embeds_context = view_embeds.unsqueeze(1)  # [view*batch, 1, d_model]
        
        # Add view embedding to context token (avoiding in-place operations)
        context_tokens = context_tokens + view_embeds_context
        
        # Expand view embeddings to all spatial positions
        view_embeds = view_embeds.unsqueeze(1).expand(-1, n_height * n_width, -1)
        
        # Add view embeddings to spatial tokens (avoiding in-place operations)
        tokens_with_view = tokens + view_embeds
        tokens_with_view = self.embed_norm(tokens_with_view)
        
        # Combine tokens with context
        tokens_with_context = torch.cat([context_tokens, tokens_with_view], dim=1)
        # tokens_with_context shape: [view*batch, 1+h*w, d_model]
        
        # Reshape for cross-view attention - separate views and batch
        # [batch, views, 1+h*w, d_model]
        reshaped_tokens = tokens_with_context.reshape(
            n_batch, num_views, tokens_with_context.shape[1], self.transformer_d_model
        )
        
        # Extract context tokens from each view for cross-view attention
        context_only = reshaped_tokens[:, :, 0, :]  # [batch, views, d_model]
        
        # Apply cross-view attention on context tokens
        enhanced_contexts = []
        for v in range(num_views):
            # Current view's context as query
            query = context_only[:, v:v+1, :]  # [batch, 1, d_model]
            
            # All views' contexts as keys and values
            key_value = context_only  # [batch, views, d_model]
            
            # Reshape for attention
            query_flat = query.reshape(n_batch, 1, self.transformer_d_model)
            kv_flat = key_value.reshape(n_batch, num_views, self.transformer_d_model)
            
            # Apply cross-view attention
            attn_output, _ = self.cross_view_attn(
                query=query_flat,
                key=kv_flat,
                value=kv_flat,
            )
            
            # Apply residual connection and normalization (avoiding in-place operations)
            enhanced_context = self.cross_view_norm1(query_flat + attn_output)
            
            # Apply MLP
            mlp_output = self.cross_view_mlp(enhanced_context)
            enhanced_context = self.cross_view_norm2(enhanced_context + mlp_output)
            
            enhanced_contexts.append(enhanced_context)
            
        # Stack enhanced contexts
        enhanced_contexts = torch.stack(enhanced_contexts, dim=1)  # [batch, views, 1, d_model]
        
        # Create a new tensor for the updated tokens to avoid in-place operations
        updated_tokens = reshaped_tokens.clone()
        
        # Replace original context tokens with enhanced ones
        updated_tokens[:, :, 0, :] = enhanced_contexts.squeeze(2)
        
        # Reshape back to combined view-batch form for transformer
        tokens_enhanced = updated_tokens.reshape(
            n_batch * num_views, tokens_with_context.shape[1], self.transformer_d_model
        ) # can have the sequence be all of the views - has access to tokens from all other views and have access to the context tokens. 
        


        # Apply transformer (processing each view independently)
        transformer_output = self.transformer_encoder(tokens_enhanced)
        
        # reshape tokens and feed into HeatmapHead
        spatial_tokens = transformer_output[:, 1:, :]
        spatial_tokens = spatial_tokens.reshape(
            num_views * n_batch, n_height, n_width, self.transformer_d_model
        )
        spatial_tokens = spatial_tokens.permute(0, 3, 1, 2)  # [view*batch, d_model, h, w]
        
        # Generate heatmaps
        heatmaps = self.upsample(spatial_tokens)
        print(f"Upsampled heatmaps shape: {heatmaps.shape}")
        
        # Reshape output
        heatmaps = heatmaps.reshape(
            n_batch, num_views, self.out_channels, n_height * 8, n_width * 8
        )
        heatmaps = heatmaps.reshape(n_batch, -1, n_height * 8, n_width * 8)
        print(f"Final heatmaps shape: {heatmaps.shape}")

        return heatmaps

    def run_subpixelmaxima(self, heatmaps):
        """Avoid in-place operations in subpixelmaxima calculation"""
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)



