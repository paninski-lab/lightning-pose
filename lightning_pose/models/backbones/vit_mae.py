from model.vit_mae.vit_mae import ContrastViTMAE
import math
import torch
import torch.nn as nn
from transformers import (
    ViTMAEConfig, 
    ViTMAEModel, 
    ViTMAEForPreTraining,
)
from segment_anything.modeling import ImageEncoderViT
from typing import Tuple, Type
# to ignore imports for sphix-autoapidoc
__all__ = [
    "ImageEncoderViTMAE",
]

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ImageEncoderViTMAE(nn.Module):
    def __init__(self, config):
        super().__init__() # Initialize parent nn.Module firsts
        self.vit_mae = ContrastViTMAE(config).vit_mae
        del(self.vit_mae.decoder) # remove the decoder from the vit_mae
        self.config = config
        # self.neck = nn.Sequential(
        #     nn.Conv2d(
        #         config['hidden_size'],
        #         config['output_channels'],
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(config['output_channels']),
        #     nn.Conv2d(
        #         config['output_channels'],
        #         config['output_channels'],
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(config['output_channels']),
        # )

    def forward(self, x):
        N = x.shape[0]
        if self.config['num_channels'] == 1:
            print('num_channels is 1, should be 3')
            # adjust input channels to 1
            x = x[:, 0, ...].unsqueeze(1)
            exit()
        outputs = self.vit_mae(
            pixel_values=x,
            return_latent=True
        )
        # skip the cls token
        outputs = outputs[:, 1:, ...] # [N, S, D]
        # change the shape to [N, H, W, D] -> [N, D, H, W]
        S = outputs.shape[1]
        H, W = math.isqrt(S), math.isqrt(S)
        outputs = outputs.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        # outputs = self.neck(outputs)
        return outputs
    