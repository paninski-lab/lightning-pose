# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys

sys.path.append("/data/Projects/lightning-pose/third_party/E-RayZer")
import copy
from huggingface_hub import hf_hub_download
from typing import Tuple
from urllib.parse import urlparse
import numpy as np
import torch
from easydict import EasyDict as edict
from einops import rearrange, repeat
import torch.nn.functional as F
from erayzer_core.model.erayzer import (ERayZer,
                                        cam_info_to_plucker, get_cam_se3)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
def _parse_hf_url(url: str) -> Tuple[str, str, str]:
    """Extract repo id, revision, and file path from a huggingface.co URL."""
    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        raise ValueError(f"Unsupported checkpoint host: {parsed.netloc}")

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 3:
        raise ValueError(f"Malformed Hugging Face URL: {url}")

    repo_id = "/".join(segments[:2])
    pointer = segments[2]
    if pointer in {"blob", "resolve", "raw"}:
        if len(segments) < 5:
            raise ValueError(f"Missing file path in Hugging Face URL: {url}")
        revision = segments[3]
        file_path = "/".join(segments[4:])
    else:
        revision = "main"
        file_path = "/".join(segments[2:])
    return repo_id, revision, file_path


def _get_ckpt_from_hf(url: str) -> str:
    repo_id, revision, file_path = _parse_hf_url(url)
    return hf_hub_download(repo_id=repo_id, filename=file_path, revision=revision)

"""
Camera Pose Evaluation Utilities

Adapted from:
https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/test_co3d.py#L163
"""
def closed_form_inverse_se3(se3):
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]

    R_T = R.transpose(1, 2)
    t_new = -torch.bmm(R_T, T)

    inv = torch.eye(4, device=se3.device, dtype=se3.dtype).unsqueeze(0).repeat(se3.size(0), 1, 1)
    inv[:, :3, :3] = R_T
    inv[:, :3, 3:] = t_new

    return inv


def normalize_extrinsics_sequence(
    extrinsics: torch.Tensor,
    norm_method: str = "mean_dist",
) -> torch.Tensor:
    """
    Normalize camera extrinsics for a single sequence (no batch dimension).
    
    This function follows the same logic as DepthAnything3's _normalize_extrinsics:
    1. Transforms the coordinate system to be centered at the first camera
    2. Scales the scene so the median/mean distance from origin is 1.0
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (S, 4, 4) - homogeneous matrices
                    where S is the number of views in the sequence
        norm_method: Normalization method. Options:
                    - "median_dist": Use median distance (more robust to outliers)
                    - "mean_dist" (default): Use mean distance (more common in CV models)
    
    Returns:
        Normalized camera extrinsics of shape (S, 4, 4)
    """
    if extrinsics.dim() != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics of shape (S, 4, 4), got {extrinsics.shape}")
    
    # Step 1: Get transform from first camera (w2c -> c2w)
    # First camera's extrinsic: extrinsics[0] -> (4, 4)
    # Need to add batch dim for closed_form_inverse_se3: (1, 4, 4)
    first_cam = extrinsics[0:1]  # (1, 4, 4)
    transform = closed_form_inverse_se3(first_cam).squeeze(0)  # (4, 4)
    
    # Step 2: Apply transform to all cameras
    # ex_t_norm = ex_t @ transform
    # (S, 4, 4) @ (4, 4) -> (S, 4, 4)
    ex_t_norm = torch.matmul(extrinsics, transform)  # (S, 4, 4) @ (4, 4) -> (S, 4, 4)
    
    # Step 3: Convert normalized extrinsics to c2w to get translations
    # c2ws = affine_inverse(ex_t_norm)
    # Using closed_form_inverse_se3: reshape to (S, 4, 4) -> (S, 4, 4) after inverse
    c2ws = closed_form_inverse_se3(ex_t_norm)  # (S, 4, 4)
    
    # Step 4: Extract translations and compute distances
    translations = c2ws[:, :3, 3]  # (S, 3)
    dists = translations.norm(dim=-1)  # (S,)
    
    # Step 5: Compute scale factor based on norm_method
    if norm_method == "median_dist":
        scale_dist = torch.median(dists)
    elif norm_method == "mean_dist":
        # Use mean distance, excluding the first camera (distance 0) like VGGT
        valid_distances = dists[dists > 1e-6]
        if len(valid_distances) > 0:
            scale_dist = valid_distances.mean()
        else:
            scale_dist = torch.tensor(1.0, device=dists.device, dtype=dists.dtype)
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}. Must be 'median_dist' or 'mean_dist'")
    
    scale_dist = torch.clamp(scale_dist, min=1e-1)
    
    # Step 6: Scale translation component
    ex_t_norm[:, :3, 3] = ex_t_norm[:, :3, 3] / scale_dist
    
    return ex_t_norm

def update_intrinsics_bbox(K, bbox, img_height, img_width):
    """
    Updates the camera intrinsic matrices for a batch of images that are 
    cropped and then resized.
    
    Args:
        K (torch.Tensor): Original intrinsic matrices, shape [view, 3, 3].
        bbox (torch.Tensor): Tensor of shape [view, 4], formatted as [x_min, y_min, height, width].
        img_height (float/int): The target height the crop is resized to.
        img_width (float/int): The target width the crop is resized to.

    Returns:
        torch.Tensor: The new intrinsic matrices, shape [view, 3, 3].
    """
    
    # Extract batched top-left coordinates and crop dimensions
    x_min = bbox[:, 0]  # Shape: [view]
    y_min = bbox[:, 1]  # Shape: [view]
    crop_height = bbox[:, 2]  # Shape: [view]
    crop_width = bbox[:, 3]  # Shape: [view]
    
    # Calculate scaling factors
    h_factor = img_height / crop_height
    w_factor = img_width / crop_width
    
    # Clone to avoid modifying the original tensor in-place 
    K_new = K.clone()

    # 1. Update the principal points (Shift for the crop, then scale for the resize)
    K_new[:, 0, 2] = (K_new[:, 0, 2] - x_min) * w_factor
    K_new[:, 1, 2] = (K_new[:, 1, 2] - y_min) * h_factor
    
    # 2. Update the focal lengths (Scale for the resize)
    K_new[:, 0, 0] *= w_factor
    K_new[:, 1, 1] *= h_factor

    return K_new

def get_fxfycxcy_from_intrinsics(intrinsics):
    # intrinsics: shape (b, s, 3, 3)
    # return fxfycxcy: shape (b, s, 4)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    return torch.stack([fx, fy, cx, cy], dim=-1)

# class BEAST3D(ERayZer):
#     def __init__(self, config):
#         super().__init__(config.model.model_params)
#         # load pretrained weights
#         if config.model.pretrained_url:
#             checkpoint = torch.load(_get_ckpt_from_hf(config.model.pretrained_url), map_location='cpu')
#             msg = self.load_state_dict(checkpoint['model'], strict=False)
#             print(f"Loaded pretrained weights from {config.model.pretrained_url} with msg: {msg}")
#         else:
#             print(f"No pretrained URL provided for {self.__class__.__name__}")
#         # delete upsamper
#         del self.upsampler
#         # delete renderer
#         del self.renderer
#         # delete image_token_decoder
#         del self.image_token_decoder

#     def inference_render(self, batch):
#         all_images = batch['images']

#         B, V, C, H, W = all_images.shape

#         input_images = all_images
#         input_img_tokens = self.image_tokenizer(input_images)  # [(B*V), N, d]
#         if self.use_pe_embedding_layer:
#             input_img_tokens = self.add_spatial_pe(
#                 input_img_tokens,
#                 B, V,
#                 self.hh,
#                 self.ww,
#                 embedder=self.pe_embedder,
#             )
#         # concanate all tokens together
#         n = input_img_tokens.shape[1]
#         cam_tokens = repeat(self.camera_token, '1 n d -> bv n d', bv=B*V)
#         register_tokens = repeat(self.register_token, '1 n d -> bv n d', bv=B*V)
#         all_tokens = torch.cat([cam_tokens, register_tokens, input_img_tokens], dim=1)
#         all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=B)
#         # stage 1: pose estimation encoder
#         with torch.autocast(device_type=all_tokens.device.type, dtype=torch.bfloat16):
#             all_tokens = self.run_vggt_encoder(all_tokens, B, V)
#         all_tokens = rearrange(all_tokens, 'b (v n) d -> (b v) n d', v=V)
#         cam_tokens_out, _, _ = all_tokens.split([1, self.num_register_tokens, n], dim=1)

#         # predict camera poses from cam tokens
#         cam_tokens_out = cam_tokens_out[:, 0]  # [(B*V), d]
#         cam_info = self.pose_predictor(cam_tokens_out, V)  # [(B*V), num_pose_element+3+4]
#         pred_c2w, pred_fxfycxcy = get_cam_se3(cam_info)  # [(B*V), 4, 4], [(B*V), 4]
#         pred_c2w = rearrange(pred_c2w, '(b v) n d -> b v n d', b=B)
#         pred_fxfycxcy = rearrange(pred_fxfycxcy, '(b v) d -> b v d', b=B).detach()

#         # stage 2: add camera embedding via plucker rays
#         plucker_rays_input = cam_info_to_plucker(pred_c2w, pred_fxfycxcy, self.config.model.target_image, normalized=True, return_moment=True)
#         plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v h w c', b=B, v=V)
#         plucker_rays_input = plucker_rays_input.float()
#         plucker_emb_input = self.input_pose_tokenizer(plucker_rays_input)
#         if self.use_pe_embedding_layer:
#             plucker_emb_input = self.add_spatial_pe(
#                 plucker_emb_input,
#                 B, V,
#                 self.hh, self.ww,
#                 embedder=self.pe_embedder_plucker,
#             )
#         plucker_emb_input = rearrange(plucker_emb_input, '(b v) n d -> b (v n) d', v=V)
#         img_tokens_out = rearrange(input_img_tokens, '(b v) n d -> b (v n) d', b=B, v=V)
#         input_tokens = torch.cat([img_tokens_out, plucker_emb_input], dim=-1)
#         input_tokens = self.mlp_fuse(input_tokens)
#         # geometry encoder
#         with torch.autocast(device_type=input_tokens.device.type, dtype=torch.bfloat16):
#             input_tokens = self.run_vggt_encoder_geom(input_tokens, B, V)
#         input_tokens = rearrange(input_tokens, 'b (v n) d -> b v n d', b=B, v=V)
#         return input_tokens.float()

#     def forward(self, images, intrinsic_matrix, extrinsic_matrix, bbox):
#         batch = edict(
#             images=images,
#         )
#         return self.inference_render(batch)

# class BEAST3D(ERayZer):
#     def __init__(self, config):
#         super().__init__(config.model.model_params)
#         # load pretrained weights
#         if config.model.pretrained_url:
#             checkpoint = torch.load(_get_ckpt_from_hf(config.model.pretrained_url), map_location='cpu')
#             msg = self.load_state_dict(checkpoint['model'], strict=False)
#             print(f"Loaded pretrained weights from {config.model.pretrained_url} with msg: {msg}")
#         else:
#             print(f"No pretrained URL provided for {self.__class__.__name__}")
#         # delete the transformer_encoder_geom
#         del self.transformer_encoder_geom
#         # delete the pose_predictor
#         del self.pose_predictor
#         # delete upsamper
#         del self.upsampler
#         # delete renderer
#         del self.renderer
#         # delete image_token_decoder
#         del self.image_token_decoder

#     def inference_render(self, batch):
#         all_images = batch['images']

#         B, V, C, H, W = all_images.shape

#         input_images = all_images
#         input_img_tokens = self.image_tokenizer(input_images)
#         if self.use_pe_embedding_layer:
#             input_img_tokens = self.add_spatial_pe(
#                 input_img_tokens,
#                 B, V,
#                 self.hh,
#                 self.ww,
#                 embedder=self.pe_embedder,
#             )
#         # concanate all tokens together
#         cam_tokens = repeat(self.camera_token, '1 n d -> bv n d', bv=B*V)
#         register_tokens = repeat(self.register_token, '1 n d -> bv n d', bv=B*V)
#         all_tokens = torch.cat([cam_tokens, register_tokens, input_img_tokens], dim=1)
#         all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=B)
#         # encoder layers, predict depths and feature vectors
#         with torch.autocast(device_type=all_tokens.device.type, dtype=torch.bfloat16):
#             all_tokens = self.run_vggt_encoder(all_tokens, B, V)
#         all_tokens = rearrange(all_tokens, 'b (v n) d -> b v n d', b=B, v=V)
#         # get tokens
#         tokens = all_tokens[:, :, 5:, :]
#         return tokens
    
#     def forward(self, images, intrinsic_matrix, extrinsic_matrix, bbox):
#         batch = edict(
#             images=images,
#         )
#         return self.inference_render(batch)


class BEAST3D(ERayZer):
    def __init__(self, config):
        super().__init__(config.model.model_params)

        # load pretrained weights
        if config.model.pretrained_url:
            checkpoint = torch.load(_get_ckpt_from_hf(config.model.pretrained_url), map_location='cpu')
            msg = self.load_state_dict(checkpoint['model'], strict=False)
            print(f"Loaded pretrained weights from {config.model.pretrained_url} with msg: {msg}")
        else:
            print(f"No pretrained URL provided for {self.__class__.__name__}")
        # delete the transformer_encoder
        del self.transformer_encoder
        # delete the pose_predictor
        del self.pose_predictor
        # delete the camera_token
        del self.camera_token
        # delete the register_token
        del self.register_token
        # delete upsamper
        del self.upsampler
        # delete renderer
        del self.renderer
        # delete image_token_decoder
        del self.image_token_decoder

    def inference_render(self, batch):
        all_images = batch['images']
        c2w = batch['c2w']
        fxfycxcy = batch['fxfycxcy']

        B, V, C, H, W = all_images.shape

        input_images = all_images
        input_img_tokens = self.image_tokenizer(input_images)
        if self.use_pe_embedding_layer:
            input_img_tokens = self.add_spatial_pe(
                input_img_tokens,
                B, V,
                self.hh,
                self.ww,
                embedder=self.pe_embedder,
            )
        # add camera embedding
        plucker_rays_input = cam_info_to_plucker(c2w, fxfycxcy, self.config.model.target_image, normalized=False, return_moment=True)
        plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v h w c', b=B, v=V)
        plucker_rays_input = plucker_rays_input.float()  # Convert to float32 to match Linear layer dtype
        plucker_emb_input = self.input_pose_tokenizer(plucker_rays_input)
        if self.use_pe_embedding_layer:
            plucker_emb_input = self.add_spatial_pe(
                plucker_emb_input,
                B, V,
                self.hh, self.ww,
                embedder=self.pe_embedder_plucker,
            )
        plucker_emb_input = rearrange(plucker_emb_input, '(b v) n d -> b (v n) d', v=V)
        input_img_tokens = rearrange(input_img_tokens, '(b v) n d -> b (v n) d', b=B, v=V)
        input_tokens = torch.cat([input_img_tokens, plucker_emb_input], dim=-1)
        input_tokens = self.mlp_fuse(input_tokens)
        # encoder layers, predict depths and feature vectors
        with torch.autocast(device_type=input_tokens.device.type, dtype=torch.bfloat16):
            input_tokens = self.run_vggt_encoder_geom(input_tokens, B, V)
        input_tokens = rearrange(input_tokens, 'b (v n) d -> b v n d', b=B, v=V)
        return input_tokens.float()
    
    def forward(self, images, intrinsic_matrix, extrinsic_matrix, bbox):
        device = images.device
        batch_size, num_views, channels, img_height, img_width = images.shape
        # Convert extrinsics to homogeneous form: (B, N,4,4)
        extrinsics_homog = torch.cat(
            [
                extrinsic_matrix,
                torch.zeros((batch_size, num_views, 1, 4), device=device),
            ],
            dim=-2,
        )
        extrinsics_homog[:, :, -1, -1] = 1.0
        c2w = extrinsics_homog.clone()
        new_intrinsic_matrix = intrinsic_matrix.clone()
        for b in range(batch_size):
            # show max and min of extrinsics_homog
            c2w[b] = closed_form_inverse_se3(normalize_extrinsics_sequence(extrinsics_homog[b]))
            new_intrinsic_matrix[b] = update_intrinsics_bbox(intrinsic_matrix[b], bbox[b], img_height, img_width)
        # get fxfycxcy from new_intrinsic_matrix
        fxfycxcy = get_fxfycxcy_from_intrinsics(new_intrinsic_matrix)
        # # Undo ImageNet normalization (applied by LitPose dataset) and convert to [-1, 1]
        # # as ERayZer's image_tokenizer expects (pretrained on image * 2.0 - 1.0)
        # imagenet_mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
        # imagenet_std  = IMAGENET_STD.to(device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
        # images = (images * imagenet_std + imagenet_mean) * 2.0 - 1.0

        batch = edict(
            images=images,
            c2w=c2w,
            fxfycxcy=fxfycxcy,
        )
        return self.inference_render(batch)