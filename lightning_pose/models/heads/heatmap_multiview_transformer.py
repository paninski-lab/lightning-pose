"""Heads that produce heatmap predictions for heatmap regression."""

from typing import Tuple
from typing_extensions import Literal

import numpy as np
import torch
import torch.nn.functional as F
import math
from kornia.geometry.subpix import spatial_softmax2d
from torch import nn, Tensor
from torchtyping import TensorType

from lightning_pose.models.heads import HeatmapHead, HeatmapHeadNoShuffle
from lightning_pose.models.heads.heatmap import run_subpixelmaxima


# to ignore imports for sphix-autoapidoc
__all__ = [
    "MultiviewFeatureTransformerHead",
    "MultiviewFeatureTransformerHeadLearnable",
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

class CameraParameterEncoder(nn.Module):
    """Dedicated encoder for camera parameters to create more expressive view embeddings."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights for better training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.encoder(x)

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
        view_embed_dim: int = 25, # 9 + 12 + num_distortions
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

        # Camera parameter encoder - replaces simple linear projection with dedicated encoder
        self.camera_encoder = CameraParameterEncoder(
            input_dim=view_embed_dim,
            output_dim=transformer_d_model,
            hidden_dim=transformer_d_model // 2
        )

        # Learnable scaling factor for camera embeddings
        self.camera_scaling = nn.Parameter(torch.ones(1))
        
        # Learnable normalization parameters
        self.camera_norm = nn.LayerNorm(transformer_d_model)
        self.token_norm = nn.LayerNorm(transformer_d_model)
        self.embed_norm = nn.LayerNorm(transformer_d_model)

        # Create learnable view embeddings - one embedding vector per view
        # view_embed = nn.Parameter(
        #     torch.randn(num_views, view_embed_dim) * 0.02  # Small initialization for stability
        # )
        # self.view_embed = torch.tensor(view_embed, dtype=torch.float, device='cuda:0')

        # View embedding projection to full transformer dimension
        # self.view_projection = nn.Linear(view_embed_dim, transformer_d_model) # the view_embed_dim should be the 9 +12+num_distrotions 
        
        # # Layer normalization for combining embeddings
        # self.embed_norm = nn.LayerNorm(transformer_d_model)

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


        # this is a try without the shuffle of the pixels! I hope will work well 
        # self.upsample = HeatmapHeadNoShuffle(
        #     backbone_arch='resnet',
        #     in_channels=transformer_d_model,
        #     out_channels=out_channels,
        #     downsample_factor=2,
        #     final_softmax=True,
        # )

        # try using HybridUpsamplingHead
        # self.upsample = HybridUpsamplingHead(
        #     backbone_arch='resnet',
        #     in_channels=transformer_d_model,
        #     out_channels=out_channels,
        #     downsample_factor=2,
        #     final_softmax=True,
        # )
        

        print(f" the number of transformer layers is {transformer_num_layers} ")
        print(f"Using learnable view embeddings with {num_views} views, dimension {view_embed_dim}")
        print(f" using the following upsampling head : {self.upsample.__class__.__name__}")

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training convergence."""
        # Initialize tokenizer
        nn.init.xavier_uniform_(self.tokenize.weight)
        if self.tokenize.bias is not None:
            nn.init.zeros_(self.tokenize.bias)
            
        # Initialize scaling with slight positive bias
        with torch.no_grad():
            self.camera_scaling.fill_(0.5)  # Start with lower influence from camera params
    
    def _normalize_camera_params(self, camera_params):
        """Normalize camera parameters for better training stability."""
        if camera_params is None:
            return None
            
        # Split into components
        intrinsics = camera_params[..., :9]  # First 9 elements (3x3 matrix)
        extrinsics = camera_params[..., 9:21]  # Next 12 elements (3x4 matrix)
        distortions = camera_params[..., 21:]  # Remaining elements
        
        # Normalize each component separately with small epsilon for stability
        eps = 1e-6
        
        # For intrinsics, focal length and principal point have different scales
        intrinsics_mean = intrinsics.mean(dim=[0, 1], keepdim=True)
        intrinsics_std = intrinsics.std(dim=[0, 1], keepdim=True) + eps
        intrinsics_normalized = (intrinsics - intrinsics_mean) / intrinsics_std
        
        # For extrinsics, rotation and translation have different scales
        # Rotation part (first 9 elements)
        rot_mean = extrinsics[..., :9].mean(dim=[0, 1], keepdim=True)
        rot_std = extrinsics[..., :9].std(dim=[0, 1], keepdim=True) + eps
        rot_normalized = (extrinsics[..., :9] - rot_mean) / rot_std
        
        # Translation part (last 3 elements)
        trans_mean = extrinsics[..., 9:].mean(dim=[0, 1], keepdim=True)
        trans_std = extrinsics[..., 9:].std(dim=[0, 1], keepdim=True) + eps
        trans_normalized = (extrinsics[..., 9:] - trans_mean) / trans_std
        
        # For distortions
        dist_mean = distortions.mean(dim=[0, 1], keepdim=True)
        dist_std = distortions.std(dim=[0, 1], keepdim=True) + eps
        dist_normalized = (distortions - dist_mean) / dist_std
        
        # Recombine all normalized parameters
        normalized_params = torch.cat([
            intrinsics_normalized, 
            rot_normalized, 
            trans_normalized, 
            dist_normalized
        ], dim=-1)
        
        return normalized_params
        
        
    def forward(
        self,
        features: TensorType["view x batch", "features", "rep_height", "rep_width"],
        num_views: torch.tensor,
        camera_params: TensorType["batch", "num_views", "camera_params"] | None, # assuming reshaping have been done 
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Upsample and run multiview head to get final heatmaps.

        Args:
            features: outputs of backbone
            num_views: number of camera views for each batch element

        """

        # print(f" the shape of camera_params is {camera_params.shape} ")
        batch_size_combined = features.shape[0]
        n_batch = int(batch_size_combined // num_views)
        n_width = features.shape[-1]
        n_height = features.shape[-2]

        # Verify camera parameters if provided
        if camera_params is not None:
            # Check dimensions match expected view_embed_dim
            assert camera_params.shape[2] == self.view_embed_dim, (
                f"Camera parameter dimension mismatch: got {camera_params.shape[2]}, "
                f"expected {self.view_embed_dim}"
            )

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
        # view_indices = torch.arange(num_views, device=tokens.device).repeat_interleave(n_batch)
        # view_indices shape: [view * batch]

        # 4. Process camera parameters through dedicated encoder
        if camera_params is not None:
            # Normalize camera parameters
            normalized_params = self._normalize_camera_params(camera_params)
            
            # Process through encoder
            view_embeds = normalized_params.float()  # [batch, num_views, view_embed_dim]
            view_embeds = view_embeds.reshape(-1, view_embeds.shape[-1])  # [view*batch, view_embed_dim]
            
            # Encode camera parameters with dedicated encoder
            view_embeds = self.camera_encoder(view_embeds)  # [view*batch, d_model]
            
            # Apply camera embedding normalization
            view_embeds = self.camera_norm(view_embeds)
            
            # Apply learnable scaling factor
            view_embeds = view_embeds * torch.sigmoid(self.camera_scaling)
            
            # Expand to match spatial dimensions
            view_embeds = view_embeds.unsqueeze(1).expand(-1, n_height * n_width, -1)
            # [view*batch, rep_height*rep_width, d_model]
        else:
            # Initialize empty embeddings if no camera parameters
            view_embeds = torch.zeros(
                (num_views * n_batch, n_height * n_width, self.transformer_d_model),
                device=features.device, 
                dtype=features.dtype
            )

        
        # view_embeds = camera_params.float() # we simply get the camera params as the view embeddings
        #[batch, view ,26]
        # make the view embeds a Double

        # # Get the corresponding view embedding for each batch item
        # view_embeds = self.view_embed[view_indices]
        # # view_embeds shape: [view * batch, view_embed_dim]
        # print(f" the shape of view_embeds is {view_embeds.shape} ")
        # # Expand view embeddings to all spatial positions
        # view_embeds = view_embeds.reshape(-1, view_embeds.shape[-1]).unsqueeze(1).expand(-1, n_height * n_width, view_embeds.shape[-1])
        # view_embeds shape: [view * batch, rep_height * rep_width, view_embed_dim]
        # print(f" the shape of view_embeds is {view_embeds.shape} ")

        # Project view embeddings to full dimension
        # view_embeds = self.view_projection(view_embeds)  # [view*batch, dim]

         # Combine token and view embeddings with normalization
        tokens = self.embed_norm(tokens + view_embeds)
        # print(f" the shape of tokens after adding the view embeddings is {tokens.shape} ")

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
        transformer_dim_feedforward: int = 1024,  # Moderate increase from 512
        transformer_num_layers: int = 2,          # Keep at 2 as requested
        img_size: int = 256,
        view_embed_dim: int = 64,                # Keep at 64 for compatibility
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
        self.downsample_factor = downsample_factor
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

        # Determine grid size based on image size
        if img_size == 128:
            grid_size = 4
        elif img_size == 256:
            grid_size = 8
        elif img_size == 384:
            grid_size = 12
        else:
            raise ValueError(f"Unsupported image size: {img_size}")
            
        # Create position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=transformer_d_model,
            grid_size=grid_size, 
            add_cls_token=False,
        )
        self.register_buffer('pos_embed', torch.tensor(pos_embed, dtype=torch.float).unsqueeze(0))

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
            backbone_arch='resnet',
            in_channels=transformer_d_model,
            out_channels=out_channels,
            downsample_factor=2,
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
        )
        
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
        
        # Reshape output
        heatmaps = heatmaps.reshape(
            n_batch, num_views, self.out_channels, n_height * 8, n_width * 8
        )
        heatmaps = heatmaps.reshape(n_batch, -1, n_height * 8, n_width * 8)

        return heatmaps

    def run_subpixelmaxima(self, heatmaps):
        """Avoid in-place operations in subpixelmaxima calculation"""
        return run_subpixelmaxima(heatmaps, self.downsample_factor, self.temperature)


# Helper function for positional encoding
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Size of the grid (height=width assumed)
        add_cls_token: Whether to include embeddings for CLS tokens
        
    Returns:
        Positional embeddings with shape (grid_size*grid_size, embed_dim) or 
        (1+grid_size*grid_size, embed_dim) if add_cls_token is True.
    """
    # Create positional embeddings by grid
    grid_h = grid_size
    grid_w = grid_size
    grid_h_pos = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1)
    grid_w_pos = torch.arange(grid_w, dtype=torch.float32).unsqueeze(1)
    pos_embed_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_pos)  # [H, D/2]
    pos_embed_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_pos)  # [W, D/2]
    
    # Create 2D positional embeddings by outer product
    pos_embed_2d = []
    for h in range(grid_h):
        embed_h = pos_embed_h[h:h+1]  # [1, D/2]
        for w in range(grid_w):
            embed_w = pos_embed_w[w:w+1]  # [1, D/2]
            embed_hw = torch.cat([embed_h.repeat(1, 1), embed_w.repeat(1, 1)], dim=1)  # [1, D]
            pos_embed_2d.append(embed_hw)
    
    pos_embed_2d = torch.cat(pos_embed_2d, dim=0)  # [H*W, D]
    
    # Add CLS token embedding if requested
    if add_cls_token:
        cls_token_embed = torch.zeros([1, embed_dim])
        pos_embed_2d = torch.cat([cls_token_embed, pos_embed_2d], dim=0)
    
    return pos_embed_2d.numpy()


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos_grid):
    """
    Convert 1D position grid to sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        pos_grid: Position grid tensor of shape (length, 1)
        
    Returns:
        Positional embeddings tensor of shape (length, embed_dim)
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    
    pos_grid = pos_grid.reshape(-1)  # [length]
    out = torch.einsum('i,j->ij', pos_grid, omega)  # [length, embed_dim//2]
    
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    
    pos_embed = torch.cat([emb_sin, emb_cos], dim=1)  # [length, embed_dim]
    return pos_embed


class ViewConsistencyLoss(nn.Module):
    """Loss that enforces consistency between 3D projections from different camera pairs."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, keypoints_3d, keypoints_mask):
        """
        Args:
            keypoints_3d: 3D keypoints from different camera pairs [batch, cam_pairs, keypoints, 3]
            keypoints_mask: Validity mask for keypoints [batch, cam_pairs, keypoints]
        """
        if keypoints_3d is None or keypoints_mask is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            
        batch_size, cam_pairs, num_keypoints, _ = keypoints_3d.shape
        
        # Skip if we only have one camera pair
        if cam_pairs <= 1:
            return torch.tensor(0.0, device=keypoints_3d.device)
            
        # Calculate pairwise distances between all camera pair estimates
        total_loss = 0.0
        valid_pairs = 0
        
        for i in range(cam_pairs):
            for j in range(i+1, cam_pairs):
                # Get keypoints from both camera pairs
                points_i = keypoints_3d[:, i]  # [batch, keypoints, 3]
                points_j = keypoints_3d[:, j]  # [batch, keypoints, 3]
                
                # Get validity masks
                mask_i = keypoints_mask[:, i]  # [batch, keypoints]
                mask_j = keypoints_mask[:, j]  # [batch, keypoints]
                
                # Combined mask - only compare valid keypoints from both views
                valid_mask = mask_i & mask_j  # [batch, keypoints]
                
                if valid_mask.sum() == 0:
                    continue
                    
                # Calculate Euclidean distance between corresponding keypoints
                distances = torch.norm(points_i - points_j, dim=-1)  # [batch, keypoints]
                
                # Apply mask and calculate mean
                masked_distances = distances * valid_mask.float()
                pair_loss = masked_distances.sum() / (valid_mask.sum() + 1e-6)
                
                total_loss += pair_loss
                valid_pairs += 1
                
        # Return average loss
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=keypoints_3d.device)


class Reprojection3DLoss(nn.Module):
    """Loss that compares predicted 3D keypoints to ground truth 3D keypoints."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_3d, target_3d, mask=None):
        """
        Args:
            pred_3d: Predicted 3D keypoints [batch, cam_pairs, keypoints, 3] or [batch, keypoints, 3]
            target_3d: Target 3D keypoints [batch, keypoints, 3]
            mask: Optional validity mask [batch, cam_pairs, keypoints] or [batch, keypoints]
        """
        if pred_3d is None or target_3d is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle different prediction shapes (multiple camera pairs vs single prediction)
        if len(pred_3d.shape) == 4:  # [batch, cam_pairs, keypoints, 3]
            batch_size, cam_pairs, num_keypoints, _ = pred_3d.shape
            
            # Calculate loss for each camera pair and take the minimum
            min_loss = None
            
            for i in range(cam_pairs):
                pair_pred = pred_3d[:, i]  # [batch, keypoints, 3]
                
                # Calculate squared Euclidean distance
                diff = pair_pred - target_3d
                squared_dist = torch.sum(diff**2, dim=-1)  # [batch, keypoints]
                
                # Apply mask if available
                if mask is not None:
                    pair_mask = mask[:, i]  # [batch, keypoints]
                    squared_dist = squared_dist * pair_mask.float()
                    loss = squared_dist.sum() / (pair_mask.sum() + 1e-6)
                else:
                    loss = squared_dist.mean()
                
                # Update minimum loss
                if min_loss is None or loss < min_loss:
                    min_loss = loss
            
            return min_loss
        else:
            # Direct comparison for single 3D prediction [batch, keypoints, 3]
            diff = pred_3d - target_3d
            squared_dist = torch.sum(diff**2, dim=-1)  # [batch, keypoints]
            
            # Apply mask if available
            if mask is not None:
                squared_dist = squared_dist * mask.float()
                return squared_dist.sum() / (mask.sum() + 1e-6)
            else:
                return squared_dist.mean()