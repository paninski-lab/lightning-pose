# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything.modeling import ImageEncoderViT
except:
    raise NotImplementedError(
                "segment-anything backbones are not supported in this branch."
                "If you are running the code from a github installation, switch to the branch"
                "features/vit."
                "If you have pip installed lightning pose, there is no access to segment-anything"
                "models due to dependency/installation issues. "
                "For more information please contatct the package maintainers."
            )

from typing import Tuple, Type, List

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT_FT(ImageEncoderViT):
    def __init__(
        self,
        img_size: int = 1024,
        finetune_img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        self.img_size = img_size
        self.finetune_img_size = finetune_img_size
        self.patch_size = patch_size

        ImageEncoderViT.__init__(
            self,
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            out_chans = out_chans,
            qkv_bias = qkv_bias,
            norm_layer = norm_layer,
            act_layer = act_layer,
            use_abs_pos = use_abs_pos,
            use_rel_pos = use_rel_pos,
            rel_pos_zero_init = rel_pos_zero_init,
            window_size = window_size,
            global_attn_indexes = global_attn_indexes
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            if self.img_size != self.finetune_img_size:
                self.pos_embed = resample_abs_pos_embed_nhwc(
                    posemb=self.pos_embed, 
                    new_size=(self.finetune_img_size//self.patch_size, self.finetune_img_size//self.patch_size),
                    verbose=True)
                
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


def resample_abs_pos_embed_nhwc(
        posemb,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    if new_size[0] == posemb.shape[-3] and new_size[1] == posemb.shape[-2]:
        return posemb

    # do the interpolation
    posemb = posemb.reshape(1, posemb.shape[-3], posemb.shape[-2], posemb.shape[-1]).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1)

    return nn.Parameter(posemb)
