"""Models that produce heatmaps of keypoints from images on multiview datasets."""

import math
from typing import Any, Literal, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType
from einops import rearrange

from lightning_pose.data.cameras import project_3d_to_2d, project_camera_pairs_to_3d
from lightning_pose.data.datatypes import (
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import (
    convert_bbox_coords,
    convert_original_to_model_coords,
    undo_affine_transform_batch,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.backbones import ALLOWED_TRANSFORMER_BACKBONES
from lightning_pose.models.base import (
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)
from lightning_pose.models.heads import (
    HeatmapHead,
)

from lightning_pose.models.backbones.layers.blocks import Block
from lightning_pose.models.backbones.layers.rope import RotaryPositionEmbedding2D, PositionGetter

# to ignore imports for sphix-autoapidoc
__all__ = []

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

class VGGTAggregator(nn.Module):
    """VGGT-style alternating attention aggregator for multi-view feature fusion.
    
    This aggregator uses alternating frame and global attention blocks to fuse
    features from multiple camera views.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_views: int,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_camera_tokens: int = 1,
        num_register_tokens: int = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 0.01,
        qk_norm: bool = True,
        rope_frequency: int = 100,
        aggregator_checkpoint: str | None = None,
    ):
        """Initialize the VGGT aggregator.
        
        Args:
            embedding_dim: dimension of input embeddings
            num_views: number of camera views
            depth: number of blocks in the frame and global attention layers
            num_heads: number of attention heads
            mlp_ratio: ratio of hidden dimension to embedding dimension
            num_camera_tokens: number of camera tokens
            num_register_tokens: number of register tokens
            qkv_bias: whether to use bias in the query, key, and value projections
            proj_bias: whether to use bias in the projection layer
            ffn_bias: whether to use bias in the feedforward network
            init_values: initial value for the layer scale
            qk_norm: whether to normalize the query and key
            rope_frequency: frequency for rotary position embedding
            aggregator_checkpoint: path to pretrained aggregator checkpoint
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_views = num_views
        self.depth = depth
        self.aa_order = ["frame", "global"]
        self.aa_block_num = depth
        self.num_camera_tokens = num_camera_tokens
        self.num_register_tokens = num_register_tokens
        # patch_start_idx should account for camera and register tokens
        self.patch_start_idx = num_camera_tokens + num_register_tokens
        
        # Initialize camera and register tokens (learnable parameters)
        self.camera_token = nn.Parameter(torch.randn(1, 2, num_camera_tokens, embedding_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embedding_dim))

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        
        # Initialize rotary position embedding
        self.rope = RotaryPositionEmbedding2D(frequency=rope_frequency)
        self.position_getter = PositionGetter()
        
        # Frame attention blocks
        self.frame_blocks = nn.ModuleList(
            [
                Block(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
        # Global attention blocks
        self.global_blocks = nn.ModuleList(
            [
                Block(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
        # Aggregator layernorm
        self.aggregator_layernorm = nn.LayerNorm(embedding_dim * 2)

        # loadd pretrained weights
        self.load_pretrained_weights(VGGT_URL)
    
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        
        tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
        frame_idx += 1

        return tokens, frame_idx

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        tokens = self.global_blocks[global_idx](tokens, pos=pos)
        global_idx += 1

        return tokens, global_idx

    def forward(
        self,
        representations: TensorType["batch * num_views", "embedding_dim", "height", "width"],
    ) -> TensorType["batch * num_views", "embedding_dim", "height", "width"]:
        """Forward pass through the aggregator network."""
        view_batch_size, embedding_dim, H, W = representations.shape
        B = view_batch_size // self.num_views
        representations = representations.reshape(B, self.num_views, embedding_dim, H, W)
        # permute to (batch, num_views, embedding_dim, height, width)
        tokens = representations.permute(0, 1, 3, 4, 2)
        tokens = tokens.reshape(B, self.num_views, H * W, embedding_dim)

        # Expand camera and register tokens to match batch size and sequence length
        camera_tokens = slice_expand_and_flatten(self.camera_token, B, self.num_views).reshape(B, self.num_views, -1, embedding_dim)
        register_tokens = slice_expand_and_flatten(self.register_token, B, self.num_views).reshape(B, self.num_views, -1, embedding_dim)
        tokens = torch.cat([camera_tokens, register_tokens, tokens], dim=2)
        
        # update P because we added special tokens
        B, S, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0

        pos = None
        if self.rope is not None:
            # Get position embeddings for patch tokens only (H*W patches)
            pos = self.position_getter(B * S, H, W, device=tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        for i in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    frame_tokens = tokens
                elif attn_type == "global":
                    tokens, global_idx = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    global_tokens = tokens
            if i == self.aa_block_num - 1:
                tokens = torch.cat([frame_tokens.view(B, S, P, C), global_tokens.view(B, S, P, C)], dim=-1)
        tokens = self.aggregator_layernorm(tokens)
        # Remove register tokens before reshaping back to spatial format
        tokens = tokens[:, :, self.patch_start_idx:, :]
        tokens = tokens.reshape(B, self.num_views, H, W, embedding_dim*2)
        tokens = tokens.permute(0, 1, 4, 2, 3)
        tokens = tokens.reshape(B * self.num_views, embedding_dim*2, H, W)
        return tokens
    
    def _extract_state_dict_by_prefix(self, ckpt_data: dict, prefix: str) -> dict:
        """Extract state dict entries matching a prefix and remove the prefix.
        
        Args:
            ckpt_data: checkpoint dictionary
            prefix: prefix to match and remove from keys
            
        Returns:
            Dictionary with prefix removed from keys
        """
        state_dict = {}
        for name, param in ckpt_data.items():
            if prefix in name:
                key = name.replace(f"{prefix}.", "")
                state_dict[key] = param
        return state_dict
    
    def _load_module_state_dict(self, module: nn.Module, state_dict: dict, module_name: str) -> None:
        """Load state dict into a module and log missing/unexpected keys.
        
        Args:
            module: the module to load weights into
            state_dict: state dictionary to load
            module_name: name for logging purposes
        """
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"{module_name} missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"{module_name} unexpected keys: {unexpected_keys}")
    
    def _load_parameter(self, ckpt_data: dict, key: str, param: nn.Parameter, param_name: str) -> None:
        """Load a parameter directly from checkpoint.
        
        Args:
            ckpt_data: checkpoint dictionary
            key: key in checkpoint dictionary
            param: parameter to load into
            param_name: name for logging purposes
        """
        if key in ckpt_data:
            param.data = ckpt_data[key].clone()
            print(f"Loaded {param_name} with shape {ckpt_data[key].shape}")
        else:
            print(f"Warning: Could not find {key} in checkpoint")
    
    def load_pretrained_weights(self, checkpoint: str):
        """Load pretrained weights from a checkpoint file.
        
        Args:
            checkpoint: path to the checkpoint file (.ckpt or .pth)
        """
        print(f"Loading aggregator weights from {checkpoint}")
        
        # Load checkpoint
        if checkpoint.startswith("http"):
            ckpt_data = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu")
            print(f"Loaded checkpoint from url {checkpoint}")
        else:
            ckpt_data = torch.load(checkpoint, map_location="cpu")
        
        # Load frame and global attention blocks
        frame_blocks_dict = self._extract_state_dict_by_prefix(ckpt_data, "aggregator.frame_blocks")
        self._load_module_state_dict(self.frame_blocks, frame_blocks_dict, "Frame blocks")
        
        global_blocks_dict = self._extract_state_dict_by_prefix(ckpt_data, "aggregator.global_blocks")
        self._load_module_state_dict(self.global_blocks, global_blocks_dict, "Global blocks")
        
        # Load aggregator layernorm
        aggregator_layernorm_dict = self._extract_state_dict_by_prefix(ckpt_data, "track_head.feature_extractor.norm")
        self._load_module_state_dict(self.aggregator_layernorm, aggregator_layernorm_dict, "Aggregator layernorm")
        
        # Load camera and register tokens (these are Parameters, not Modules)
        self._load_parameter(ckpt_data, "aggregator.camera_token", self.camera_token, "camera_token")
        self._load_parameter(ckpt_data, "aggregator.register_token", self.register_token, "register_token")
        
        print(f"Successfully loaded checkpoint from {checkpoint}")


class HeatmapTrackerMultiviewTransformer(BaseSupervisedTracker):
    """Transformer network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_TRANSFORMER_BACKBONES = "vits_dino",
        pretrained: bool = True,
        head: Literal["heatmap_cnn"] = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a multi-view model with transformer backbone.
        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: transformer variant to be used; cannot use convnets with this model
            pretrained: True to load pretrained imagenet weights
            head: architecture used to project per-view information to 2D heatmaps
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma
            image_size: size of input images (height=width for ViT models)
            **kwargs: additional arguments

        """

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        self.num_views = num_views

        # backwards compatibility
        if "do_context" in kwargs.keys():
            raise ValueError(
                "HeatmapTrackerMultiviewTransformer does not currently support context frames"
            )

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            image_size=image_size,
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        # create learnable view embeddings for each view
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device=device)
        generator.manual_seed(torch_seed)
        self.view_embeddings = nn.Parameter(
            torch.randn(
                self.num_views, self.num_fc_input_features, generator=generator, device=device,
            ) * 0.02
        )

        # initialize model head
        if head == "heatmap_cnn":
            self.head = HeatmapHead(
                backbone_arch=backbone,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        else:
            raise NotImplementedError(f"{head} is not a valid multiview transformer head")

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward_vit(
        self,
        images: TensorType["view * batch", "channels":3, "image_height", "image_width"],
    ):
        """Override forward pass through the vision encoder to add view embeddings."""

        # outputs = self.vision_encoder(
        #     x,
        #     return_dict=True,
        #     output_hidden_states=False,
        #     output_attentions=False,
        #     interpolate_pos_encoding=True,
        # ).last_hidden_state

        # this block mostly copies self.vision_encoder.forward(), except for addition of view embed
        ve = self.backbone.vision_encoder
        # CLS + optional register tokens (DINOv2/DINOv3) — not patch tokens
        num_prefix = 1 + getattr(ve.config, "num_register_tokens", 0)

        # create patch embeddings and add position embeddings; strip prefix tokens
        try:
            hidden_states = ve.embeddings(
                images, bool_masked_pos=None, interpolate_pos_encoding=True,
            )
        except TypeError:
            try:
                hidden_states = ve.embeddings(images, bool_masked_pos=None)
            except TypeError:
                hidden_states = ve.embeddings(images)
        embedding_output = hidden_states[:, num_prefix:, :]
        # shape: (view * batch, num_patches, embedding_dim)

        # get dims for reshaping
        view_batch_size = embedding_output.shape[0]
        num_patches = embedding_output.shape[1]
        embedding_dim = embedding_output.shape[2]
        batch_size = view_batch_size // self.num_views

        # Create view indices to map each sample to its corresponding view
        # Shape: (view * batch,)
        view_indices = torch.arange(self.num_views, device=embedding_output.device)
        view_indices = view_indices.repeat(batch_size)  # [0,1,2,3,0,1,2,3,...] for 4 views
        # Get view embeddings for each sample
        # Shape: (view * batch, embedding_dim)
        view_embeddings_batch = self.view_embeddings[view_indices]

        # Expand view embeddings to match patch dimensions
        # Shape: (view * batch, 1, embedding_dim) -> (view * batch, num_patches, embedding_dim)
        view_embeddings_expanded = view_embeddings_batch.unsqueeze(1).expand(-1, num_patches, -1)
        embedding_output = embedding_output + view_embeddings_expanded
        # Reshape to (batch, view * num_patches, embedding_dim) so that transformer attention
        # layers process all views simultaneously
        embedding_output = embedding_output.reshape(
            batch_size, self.num_views * num_patches, embedding_dim,
        )

        # ViT / DINOv2: nn.Encoder stack + layernorm. DINOv3: ModuleList + RoPE + norm (no .encoder).
        if hasattr(ve, "layer") and hasattr(ve, "rope_embeddings") and not hasattr(ve, "encoder"):
            # DINOv3ViTModel — RoPE cos/sin are per spatial patch; tile for fused multiview sequence
            cos, sin = ve.rope_embeddings(images)
            cos = cos.repeat(self.num_views, 1)
            sin = sin.repeat(self.num_views, 1)
            position_embeddings = (cos, sin)
            hidden_states = embedding_output
            for layer_module in ve.layer:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                )
            outputs = ve.norm(hidden_states)
        elif hasattr(ve, "encoder"):
            encoder_outputs = ve.encoder(
                embedding_output,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=None,
            )
            sequence_output = encoder_outputs[0]
            outputs = ve.layernorm(sequence_output)
        else:
            raise NotImplementedError(
                "Multiview ViT forward is only implemented for ViT/DINOv2 (.encoder) "
                "or DINOv3 (.layer + rope_embeddings)."
            )
        # shape: (batch, view * num_patches, embedding_dim)

        # reshape data to (view * batch, embedding_dim, height, width) for head processing
        patch_size = outputs.shape[1] // self.num_views
        H, W = math.isqrt(patch_size), math.isqrt(patch_size)
        outputs = outputs.reshape(batch_size, self.num_views, patch_size, embedding_dim)
        outputs = outputs.reshape(batch_size, self.num_views, H, W, embedding_dim).permute(
            0, 1, 4, 2, 3
        )  # shape: (batch, view, embedding_dim, H, W)
        outputs = outputs.reshape(view_batch_size, embedding_dim, H, W)

        return outputs

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        # extract pixel data from batch
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]

        batch_size, num_views, channels, img_height, img_width = images.shape

        images_flat = images.reshape(-1, channels, img_height, img_width)
        # pass through transformer to get base representations
        representations = self.forward_vit(images_flat)
        # shape: (view * batch, num_features, rep_height, rep_width)

        # get heatmaps for each representation
        heatmaps = self.head(representations)

        # reshape to put all views from a single example together
        heatmaps = heatmaps.reshape(batch_size, -1, heatmaps.shape[-2], heatmaps.shape[-1])

        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""

        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        # project predictions from pairs of views into 3d if calibration data available
        if "keypoints_3d" in batch_dict and batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints.shape[1] // 2 // num_views
            try:
                # project from 2D to 3D
                keypoints_pred_3d = project_camera_pairs_to_3d(
                    points=pred_keypoints.reshape((-1, num_views, num_keypoints, 2)),
                    intrinsics=batch_dict["intrinsic_matrix"].float(),
                    extrinsics=batch_dict["extrinsic_matrix"].float(),
                    dist=batch_dict["distortions"].float(),
                )
                keypoints_targ_3d = batch_dict["keypoints_3d"]
                if "supervised_reprojection_heatmap_mse" in \
                        self.loss_factory.loss_instance_dict.keys():
                    # project from 3D back to 2D in original image coordinates
                    # print(f'intrinsics: {batch_dict["intrinsic_matrix"][0, 0]}')
                    keypoints_pred_2d_reprojected_original = project_3d_to_2d(
                        points_3d=torch.mean(keypoints_pred_3d, dim=1),
                        intrinsics=batch_dict["intrinsic_matrix"].float(),
                        extrinsics=batch_dict["extrinsic_matrix"].float(),
                        dist=batch_dict["distortions"].float(),
                    )
                    # convert from original image coords to model-input coords for heatmaps
                    keypoints_pred_2d_reprojected = convert_original_to_model_coords(
                        batch_dict=batch_dict,
                        original_keypoints=keypoints_pred_2d_reprojected_original,
                    ).reshape(-1, num_views * num_keypoints, 2)
                else:
                    keypoints_pred_2d_reprojected = None

            except Exception as e:
                print(f"Error in 3D projection: {e}")
                keypoints_pred_3d = None
                keypoints_targ_3d = None
                keypoints_pred_2d_reprojected = None
        else:
            keypoints_pred_3d = None
            keypoints_targ_3d = None
            keypoints_pred_2d_reprojected = None

        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": pred_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": pred_keypoints,
            "confidences": confidence,
            "keypoints_targ_3d": keypoints_targ_3d,  # shape (batch, num_keypoints, 3)
            "keypoints_pred_3d": keypoints_pred_3d,  # shape (batch, cam_pairs, num_keypoints, 3)
            "keypoints_pred_2d_reprojected": keypoints_pred_2d_reprojected,
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict heatmaps and keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)

        if return_heatmaps:
            return pred_keypoints, confidence, pred_heatmaps
        else:
            return pred_keypoints, confidence

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
            {"params": [self.view_embeddings], "name": "view_embeddings"},
        ]

        return params

class HeatmapTrackerMultiviewAggregator(BaseSupervisedTracker):
    """Aggregator network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_TRANSFORMER_BACKBONES = "vits_dino",
        pretrained: bool = True,
        head: Literal["heatmap_cnn"] = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a multi-view model with transformer backbone.
        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: transformer variant to be used; cannot use convnets with this model
            pretrained: True to load pretrained imagenet weights
            head: architecture used to project per-view information to 2D heatmaps
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma
            image_size: size of input images (height=width for ViT models)
            **kwargs: additional arguments
                - aggregator_checkpoint: path to pretrained aggregator checkpoint (optional)

        """

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        self.num_views = num_views

        # backwards compatibility
        if "do_context" in kwargs.keys():
            raise ValueError(
                "HeatmapTrackerMultiviewTransformer does not currently support context frames"
            )

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            image_size=image_size,
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        # create learnable view embeddings for each view
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device=device)
        generator.manual_seed(torch_seed)
        # self.view_embeddings = nn.Parameter(
        #     torch.randn(
        #         self.num_views, self.num_fc_input_features, generator=generator, device=device,
        #     ) * 0.02
        # )

        # Initialize VGGT aggregator
        aggregator_checkpoint = kwargs.get("aggregator_checkpoint", None)
        if self.num_fc_input_features != 1024:
            self.proj = nn.Linear(self.num_fc_input_features, 1024)
        else:
            self.proj = None
        self.aggregator = VGGTAggregator(
            embedding_dim=1024,
            num_views=self.num_views,
            aggregator_checkpoint=aggregator_checkpoint,
        )
        # initialize model head
        if head == "heatmap_cnn":
            self.head = HeatmapHead(
                backbone_arch=backbone,
                in_channels=1024 * 2,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        else:
            raise NotImplementedError(f"{head} is not a valid multiview transformer head")

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

        self.num_special_tokens = 1
        if 'dinov3' in backbone:
            self.num_special_tokens = 5

    def forward_vit(
        self,
        images: TensorType["view * batch", "channels":3, "image_height", "image_width"],
    ):
        """Override forward pass through the vision encoder to add view embeddings."""
        batch_size = images.shape[0] // self.num_views
        outputs = self.backbone.vision_encoder(images)[0][:, self.num_special_tokens:]
        view_batch_size, _, embedding_dim = outputs.shape
        outputs = outputs.reshape(batch_size, -1, embedding_dim)
        # shape: (batch, num_views * num_patches, embedding_dim)
        # project to 1024 if needed
        if self.proj is not None:
            outputs = self.proj(outputs)
            embedding_dim = 1024

        # reshape data to (view * batch, embedding_dim, height, width) for head processing
        patch_size = outputs.shape[1] // self.num_views
        H, W = math.isqrt(patch_size), math.isqrt(patch_size)
        outputs = outputs.reshape(batch_size, self.num_views, patch_size, embedding_dim)
        outputs = outputs.reshape(batch_size, self.num_views, H, W, embedding_dim).permute(
            0, 1, 4, 2, 3
        )  # shape: (batch, view, embedding_dim, H, W)
        outputs = outputs.reshape(view_batch_size, embedding_dim, H, W)

        return outputs
    
    def forward_aggregator(
        self,
        representations: TensorType["batch * num_views", "embedding_dim", "height", "width"],
    ) -> TensorType["batch * num_views", "embedding_dim", "height", "width"]:
        """Forward pass through the aggregator network."""
        return self.aggregator(representations)

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        # extract pixel data from batch
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]
        # print(f'images shape: {images.shape}')

        batch_size, num_views, channels, img_height, img_width = images.shape

        images_flat = images.reshape(-1, channels, img_height, img_width)
        # pass through transformer to get base representations
        representations = self.forward_vit(images_flat)
        # shape: (batch, num_views * num_patches, embedding_dim)
        representations = self.forward_aggregator(representations)
        # get heatmaps for each representation
        heatmaps = self.head(representations)
        # reshape to put all views from a single example together
        heatmaps = heatmaps.reshape(batch_size, -1, heatmaps.shape[-2], heatmaps.shape[-1])
        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""

        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        # project predictions from pairs of views into 3d if calibration data available
        if "keypoints_3d" in batch_dict and batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints.shape[1] // 2 // num_views

            try:
                keypoints_pred_3d = project_camera_pairs_to_3d(
                    points=pred_keypoints.reshape((-1, num_views, num_keypoints, 2)),
                    intrinsics=batch_dict["intrinsic_matrix"].float(),
                    extrinsics=batch_dict["extrinsic_matrix"].float(),
                    dist=batch_dict["distortions"].float(),
                )
                keypoints_targ_3d = batch_dict["keypoints_3d"]
                keypoints_mask_3d = get_valid_projection_masks(
                    target_keypoints.reshape((-1, num_views, num_keypoints, 2))
                )

            except Exception as e:
                print(f"Error in 3D projection: {e}")
                keypoints_pred_3d = None
                keypoints_targ_3d = None
                keypoints_mask_3d = None
        else:
            keypoints_pred_3d = None
            keypoints_targ_3d = None
            keypoints_mask_3d = None

        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": pred_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": pred_keypoints,
            "confidences": confidence,
            "keypoints_targ_3d": keypoints_targ_3d,  # shape (2*batch, num_keypoints, 3)
            "keypoints_pred_3d": keypoints_pred_3d,  # shape (2*batch, cam_pairs, num_keypoints, 3)
            "keypoints_mask_3d": keypoints_mask_3d,  # shape (2*batch, cam_pairs, num_keypoints)
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict heatmaps and keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)

        if return_heatmaps:
            return pred_keypoints, confidence, pred_heatmaps
        else:
            return pred_keypoints, confidence

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
            # {"params": [self.view_embeddings], "name": "view_embeddings"},
            {"params": self.aggregator.parameters(), "name": "aggregator"},
        ]

        return params

class HeatmapTracker3DTransformer(BaseSupervisedTracker):
    """Aggregator network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_TRANSFORMER_BACKBONES = "vits_dino",
        pretrained: bool = True,
        head: Literal["heatmap_cnn"] = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a multi-view model with transformer backbone.
        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: transformer variant to be used; cannot use convnets with this model
            pretrained: True to load pretrained imagenet weights
            head: architecture used to project per-view information to 2D heatmaps
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma
            image_size: size of input images (height=width for ViT models)
            **kwargs: additional arguments
                - aggregator_checkpoint: path to pretrained aggregator checkpoint (optional)

        """

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        self.num_views = num_views

        # backwards compatibility
        if "do_context" in kwargs.keys():
            raise ValueError(
                "HeatmapTrackerMultiviewTransformer does not currently support context frames"
            )

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            image_size=image_size,
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        # create learnable view embeddings for each view
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = torch.Generator(device=device)
        generator.manual_seed(torch_seed)
        self.ln = nn.LayerNorm(768)
        # initialize model head
        if head == "heatmap_cnn":
            self.head = HeatmapHead(
                backbone_arch=backbone,
                in_channels=768,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        else:
            raise NotImplementedError(f"{head} is not a valid multiview transformer head")

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

        self.num_special_tokens = 1
        if 'dinov3' in backbone:
            self.num_special_tokens = 5

    def forward_beast3d(
        self,
        images: TensorType["batch", "view", "channels":3, "image_height", "image_width"],
        intrinsic_matrix: TensorType["batch", "view", 3, 3],
        extrinsic_matrix: TensorType["batch", "view", 4, 4],
        bbox: TensorType["batch", "view", 4],
    ):
        """Override forward pass through the vision encoder to add view embeddings."""
        outputs = self.backbone(
            images=images, intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix, bbox=bbox
        )
        outputs = self.ln(outputs)
        p_h, p_w = math.isqrt(outputs.shape[-2]), math.isqrt(outputs.shape[-2])
        outputs = rearrange(outputs, 'b v (h w) d -> (b v) h w d', h=p_h, w=p_w)
        outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """
        # extract pixel data from batch
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]

        batch_size, num_views, channels, img_height, img_width = images.shape
        intrinsic_matrix = batch_dict["intrinsic_matrix"]
        extrinsic_matrix = batch_dict["extrinsic_matrix"]
        bbox = batch_dict["bbox"].reshape(batch_size, num_views, -1)

        # pass through transformer to get base representations
        representations = self.forward_beast3d(
            images=images, intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix, bbox=bbox
        )
        # get heatmaps for each representation
        heatmaps = self.head(representations)
        
        # reshape to put all views from a single example together
        heatmaps = heatmaps.reshape(batch_size, -1, heatmaps.shape[-2], heatmaps.shape[-1])
        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""

        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        # project predictions from pairs of views into 3d if calibration data available
        if "keypoints_3d" in batch_dict and batch_dict["keypoints_3d"].shape[-1] == 3 and False:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints.shape[1] // 2 // num_views

            try:
                keypoints_pred_3d = project_camera_pairs_to_3d(
                    points=pred_keypoints.reshape((-1, num_views, num_keypoints, 2)),
                    intrinsics=batch_dict["intrinsic_matrix"].float(),
                    extrinsics=batch_dict["extrinsic_matrix"].float(),
                    dist=batch_dict["distortions"].float(),
                )
                keypoints_targ_3d = batch_dict["keypoints_3d"]
                keypoints_mask_3d = get_valid_projection_masks(
                    target_keypoints.reshape((-1, num_views, num_keypoints, 2))
                )

            except Exception as e:
                print(f"Error in 3D projection: {e}")
                keypoints_pred_3d = None
                keypoints_targ_3d = None
                keypoints_mask_3d = None
        else:
            keypoints_pred_3d = None
            keypoints_targ_3d = None
            keypoints_mask_3d = None

        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": pred_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": pred_keypoints,
            "confidences": confidence,
            "keypoints_targ_3d": keypoints_targ_3d,  # shape (2*batch, num_keypoints, 3)
            "keypoints_pred_3d": keypoints_pred_3d,  # shape (2*batch, cam_pairs, num_keypoints, 3)
            "keypoints_mask_3d": keypoints_mask_3d,  # shape (2*batch, cam_pairs, num_keypoints)
        }

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict heatmaps and keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # bounding box coords -> original image coords
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)

        if return_heatmaps:
            return pred_keypoints, confidence, pred_heatmaps
        else:
            return pred_keypoints, confidence

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.ln.parameters(), "name": "ln"},
            {"params": self.head.parameters(), "name": "head"},
        ]

        return params



class SemiSupervisedHeatmapTrackerMultiviewTransformer(
    SemiSupervisedTrackerMixin,
    HeatmapTrackerMultiviewTransformer,
):
    """Semi-supervised HeatmapTrackerMultiviewTransformer that supports unsupervised losses."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        loss_factory_unsupervised: LossFactory | None = None,
        backbone: ALLOWED_TRANSFORMER_BACKBONES = "vits_dino",
        pretrained: bool = True,
        head: Literal["heatmap_cnn"] = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a semi-supervised multi-view model with transformer backbone.

        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss computation
            backbone: transformer variant to be used; cannot use convnets with this model
            pretrained: True to load pretrained imagenet weights
            head: architecture used to project per-view information to 2D heatmaps
            downsample_factor: make heatmap smaller than original frames to save memory
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
            image_size: size of input images (height=width for ViT models)

        """

        # initialize the parent class (HeatmapTrackerMultiviewTransformer)
        super().__init__(
            num_keypoints=num_keypoints,
            num_views=num_views,
            loss_factory=loss_factory,
            backbone=backbone,
            pretrained=pretrained,
            head=head,
            downsample_factor=downsample_factor,
            torch_seed=torch_seed,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            image_size=image_size,
            **kwargs,
        )

        self.loss_factory_unsup = loss_factory_unsupervised

        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(
        self,
        batch_dict: UnlabeledBatchDict | MultiviewUnlabeledBatchDict,
    ) -> dict:
        """
        Return predicted heatmaps and keypoints for unlabeled data
        (required by SemiSupervisedTrackerMixin).
        """

        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict)
        # heatmaps -> keypoints
        pred_keypoints_augmented, confidence = self.head.run_subpixelmaxima(pred_heatmaps)

        # undo augmentation if needed
        # Fix transforms shape: squeeze extra dimension if present
        transforms = batch_dict["transforms"]

        # Handle different possible transform shapes
        if len(transforms.shape) == 4:
            # Shape [num_views, 1, 2, 3] -> squeeze to [num_views, 2, 3]
            if transforms.shape[1] == 1:
                transforms = transforms.squeeze(1)
            # Shape [1, num_views, 2, 3] -> squeeze to [num_views, 2, 3]
            elif transforms.shape[0] == 1:
                transforms = transforms.squeeze(0)

        # Ensure transforms have the expected shape for multiview: [num_views, 2, 3]
        if batch_dict["is_multiview"] and len(transforms.shape) != 3:
            print(
                "WARNING: Expected transforms shape [num_views, 2, 3] for multiview, "
                f"got {transforms.shape}"
            )

        pred_keypoints = undo_affine_transform_batch(
            keypoints_augmented=pred_keypoints_augmented,
            transforms=transforms,
            is_multiview=batch_dict["is_multiview"],
        )

        # keypoints -> original image coords keypoints
        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)

        result = {
            "heatmaps_pred": pred_heatmaps,  # if augmented, augmented heatmaps
            "keypoints_pred": pred_keypoints,  # if augmented, original keypoints
            "keypoints_pred_augmented": pred_keypoints_augmented,  # match pred_heatmaps
            "confidences": confidence,
        }

        return result