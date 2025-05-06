"""Heads that produce heatmap predictions for heatmap regression."""

from typing import Tuple
from typing_extensions import Literal

import numpy as np
import torch
from kornia.geometry.subpix import spatial_softmax2d
from torch import nn, Tensor
from torchtyping import TensorType

from lightning_pose.models.heads import HeatmapHead
from lightning_pose.models.heads.heatmap import run_subpixelmaxima


# to ignore imports for sphix-autoapidoc
__all__ = [
    "MultiviewFeatureTransformerHead",
    "MultiviewFeatureTransformerHeadLearnable",
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
        num_views: torch.tensor,
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



class MultiviewFeatureTransformerHeadLearnable(nn.Module):
    """Multi-view transformer neural network head with learnable view embeddings.
    
    This head takes a set of 2D feature maps corresponding to different views, and fuses them
    together using a transformer architecture. Each token represents a spatial feature output by
    the backbone for a single view, along with a positional embedding and learnable view embedding.
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
        view_embed_dim: int = 64,
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
            transformer_d_model: dimension of transformer model
            transformer_nhead: number of attention heads in transformer
            transformer_dim_feedforward: dimension of feedforward network in transformer
            transformer_num_layers: number of transformer layers
            img_size: input image size
        """
        super().__init__()

        self.backbone_arch = backbone_arch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = deconv_out_channels
        self.downsample_factor = downsample_factor
        self.temperature = torch.tensor(1000.0)  # soft argmax temp
        self.num_views = num_views
        self.view_embed_dim = view_embed_dim
        self.transformer_d_model = transformer_d_model

         # create tokenizer: map from n_features of backbone to transformer features
        self.tokenize = torch.nn.Conv2d(
            in_channels=in_channels, 
            out_channels=transformer_d_model, 
            kernel_size=1,
        )

        # build position embedding
        if img_size == 128:
            grid_size = 4
        elif img_size == 256:
            grid_size = 8
        elif img_size == 384:
            grid_size = 12
        else:
            raise ValueError(f"Unsupported image size: {img_size}")
            
        # Create position embeddings with reduced dimension
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=transformer_d_model,  # Match tokenizer output
            grid_size=grid_size, 
            add_cls_token=False,
        )
        self.pos_embed = torch.tensor(pos_embed, dtype=torch.float, device='cuda:0').unsqueeze(0)

        # Create learnable view embeddings - one embedding vector per view
        view_embed = nn.Parameter(
            torch.randn(num_views, view_embed_dim) * 0.02  # Small initialization for stability
        )
        self.view_embed = torch.tensor(view_embed, dtype=torch.float, device='cuda:0')

        # View embedding projection to full transformer dimension
        self.view_projection = nn.Linear(view_embed_dim, transformer_d_model)
        
        # Layer normalization for combining embeddings
        self.embed_norm = nn.LayerNorm(transformer_d_model)

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
        print(f"Using learnable view embeddings with {num_views} views, dimension {view_embed_dim}")
        
    def forward(
        self,
        features: TensorType["view x batch", "features", "rep_height", "rep_width"],
        num_views: torch.tensor,
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
        tokens = tokens.reshape(num_views * n_batch, self.transformer_d_model, -1)
        # tokens = [view * batch, num_embed, rep_height * rep_width]
        tokens = tokens.permute((0, 2, 1))
        # tokens = [view * batch, rep_height * rep_width, num_embed]
        # assert tokens.shape == (n_views * n_batch, n_height * n_width, n_embed)

        # add position embeddings to tokens
        tokens = tokens + self.pos_embed
        # tokens = [view * batch, rep_height * rep_width, num_embed]
        # assert tokens.shape == (n_views * n_batch, n_height * n_width, n_embed)

        # Get view embeddings for each batch item
        view_indices = torch.arange(num_views, device=tokens.device).repeat_interleave(n_batch)
        # view_indices shape: [view * batch]

        # Get the corresponding view embedding for each batch item
        view_embeds = self.view_embed[view_indices]
        # view_embeds shape: [view * batch, view_embed_dim]

        # Expand view embeddings to all spatial positions
        view_embeds = view_embeds.unsqueeze(1).expand(-1, n_height * n_width, -1)
        # view_embeds shape: [view * batch, rep_height * rep_width, view_embed_dim]

        # Project view embeddings to full dimension
        view_embeds = self.view_projection(view_embeds)  # [view*batch, dim]

         # Combine token and view embeddings with normalization
        tokens = self.embed_norm(tokens + view_embeds)

        # Concatenate view embeddings with tokens (as in original code)
        # tokens = torch.cat([tokens, view_embeds], dim=-1)
        # # tokens shape: [view * batch, rep_height * rep_width, transformer_d_model]

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

