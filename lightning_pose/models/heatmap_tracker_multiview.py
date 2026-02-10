"""Models that produce heatmaps of keypoints from images on multiview datasets."""

import math
from typing import Any, Literal, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType

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

# to ignore imports for sphix-autoapidoc
__all__ = []


class PluckerRayEmbedding(nn.Module):
    """Compute 3D geometry-aware positional embeddings using Plücker ray coordinates.

    For each patch center in each camera view, computes the 3D ray from the camera
    origin through that patch's pixel location. Rays are represented as 6D Plücker
    coordinates (direction d, moment m = origin × d), Fourier-encoded, and projected
    to the transformer's embedding dimension via a small MLP.

    This gives the transformer attention geometric awareness: patches from different
    views whose rays nearly intersect in 3D (i.e. observe the same body part) receive
    related positional embeddings, enabling cross-view correspondence to emerge
    naturally through the dot-product attention mechanism.

    Computational cost is negligible (<0.1% of ViT forward pass FLOPs).
    """

    def __init__(
        self,
        embed_dim: int,
        num_freq_bands: int = 8,
        hidden_dim: int = 128,
    ):
        """Initialize Plücker ray embedding module.

        Args:
            embed_dim: output dimension (must match transformer embedding dim)
            num_freq_bands: number of Fourier frequency bands for positional encoding
            hidden_dim: hidden dimension of the projection MLP

        """
        super().__init__()
        self.num_freq_bands = num_freq_bands
        # Plücker coords are 6D; Fourier encoding expands to 6 * (2L + 1)
        fourier_dim = 6 * (2 * num_freq_bands + 1)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Initialize MLP to produce small outputs so ray embeddings start near zero
        # and don't disrupt pretrained ViT representations at the beginning of training
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal Fourier encoding (NeRF-style) to input coordinates.

        Args:
            x: (..., D) input coordinates

        Returns:
            (..., D * (2 * num_freq_bands + 1)) encoded coordinates

        """
        freqs = (
            2.0
            ** torch.arange(self.num_freq_bands, device=x.device, dtype=x.dtype)
            * math.pi
        )
        # x: (..., D), freqs: (L,)
        x_freq = x.unsqueeze(-1) * freqs  # (..., D, L)
        encoded = torch.cat(
            [
                x,                             # raw coordinates:  (..., D)
                x_freq.sin().flatten(-2),      # sin components:   (..., D*L)
                x_freq.cos().flatten(-2),      # cos components:   (..., D*L)
            ],
            dim=-1,
        )
        return encoded

    @staticmethod
    def compute_plucker_rays(
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        patch_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Plücker ray coordinates for patch centers in each camera view.

        The Plücker representation has a key property: two rays intersect iff
        d1·m2 + d2·m1 = 0, which is a dot product — exactly what transformer
        attention computes. This lets the attention learn cross-view correspondences
        based on 3D geometry.

        Args:
            intrinsics: (num_views, 3, 3) camera intrinsic matrices
            extrinsics: (num_views, 3, 4) camera extrinsic matrices [R|t]
            patch_centers: (num_patches, 2) pixel coordinates of patch centers

        Returns:
            plucker: (num_views, num_patches, 6) Plücker coordinates [direction, moment]

        """
        R = extrinsics[:, :3, :3]   # (V, 3, 3) rotation
        t = extrinsics[:, :3, 3:]   # (V, 3, 1) translation

        # Camera center in world coordinates: c = -R^T @ t
        cam_origins = -torch.bmm(R.transpose(1, 2), t).squeeze(-1)  # (V, 3)

        # Homogeneous pixel coordinates for all patch centers
        ones = torch.ones(
            patch_centers.shape[0], 1,
            device=patch_centers.device, dtype=patch_centers.dtype,
        )
        pixels_h = torch.cat([patch_centers, ones], dim=-1)  # (P, 3)

        # Ray directions in camera coordinates: K^{-1} @ [u, v, 1]^T
        K_inv = torch.inverse(intrinsics)  # (V, 3, 3)
        rays_cam = torch.einsum('vij,pj->vpi', K_inv, pixels_h)  # (V, P, 3)

        # Transform to world coordinates: R^T @ ray_cam
        rays_world = torch.einsum(
            'vij,vpj->vpi', R.transpose(1, 2), rays_cam,
        )  # (V, P, 3)

        # Normalize ray directions
        rays_world = rays_world / (rays_world.norm(dim=-1, keepdim=True) + 1e-8)

        # Plücker moment: m = origin × direction
        cam_origins_exp = cam_origins.unsqueeze(1).expand_as(rays_world)  # (V, P, 3)
        moments = torch.cross(cam_origins_exp, rays_world, dim=-1)       # (V, P, 3)

        # Concatenate [direction, moment] for 6D Plücker representation
        plucker = torch.cat([rays_world, moments], dim=-1)  # (V, P, 6)

        return plucker

    def forward(self, plucker_coords: torch.Tensor) -> torch.Tensor:
        """Project Plücker coordinates to embedding space via Fourier encoding + MLP.

        Args:
            plucker_coords: (num_views, num_patches, 6)

        Returns:
            embeddings: (num_views, num_patches, embed_dim)

        """
        encoded = self.fourier_encode(plucker_coords)
        return self.mlp(encoded)


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
        camera_intrinsics: torch.Tensor | None = None,
        camera_extrinsics: torch.Tensor | None = None,
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
            camera_intrinsics: (num_views, 3, 3) camera intrinsic matrices for 3D ray
                positional embeddings. If None, will be auto-extracted from first labeled batch.
            camera_extrinsics: (num_views, 3, 4) camera extrinsic matrices [R|t] for 3D ray
                positional embeddings. If None, will be auto-extracted from first labeled batch.
            **kwargs: additional arguments

        """

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        self.num_views = num_views
        self.image_size = image_size

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

        # ---------------------------------------------------------------
        # 3D-aware Plücker ray positional embeddings
        # ---------------------------------------------------------------
        # These encode the 3D ray direction + camera position for each
        # patch in each view, giving the transformer geometric awareness
        # of cross-view correspondences via its dot-product attention.
        self.ray_embedding = PluckerRayEmbedding(
            embed_dim=self.num_fc_input_features,
            num_freq_bands=8,
            hidden_dim=128,
        )

        # Buffer for cached Plücker coordinates (fixed camera geometry).
        # The MLP projection runs every forward pass to stay differentiable.
        self.register_buffer('_cached_plucker_coords', None)

        # If camera params provided at init, precompute ray embeddings now
        if camera_intrinsics is not None and camera_extrinsics is not None:
            self._precompute_ray_embeddings(
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
            )

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        # camera_intrinsics/extrinsics are stored as the derived _cached_plucker_coords buffer
        self.save_hyperparameters(
            ignore=["loss_factory", "loss_factory_unsupervised",
                    "camera_intrinsics", "camera_extrinsics"]
        )

    def _scale_intrinsics_to_model_input(
        self,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Scale camera intrinsics from original image coordinates to model input coordinates.

        Uses the principal point as a proxy for the original image dimensions
        (assuming cx ≈ W_orig/2 and cy ≈ H_orig/2), then scales focal lengths and
        principal point to match the model's input image size.

        Args:
            intrinsics: (num_views, 3, 3) camera intrinsic matrices in original image coords

        Returns:
            (num_views, 3, 3) intrinsic matrices scaled to model input coordinates

        """
        K = intrinsics.clone().float()
        cx = K[:, 0, 2].clamp(min=1.0)
        cy = K[:, 1, 2].clamp(min=1.0)
        scale_x = self.image_size / (2.0 * cx)
        scale_y = self.image_size / (2.0 * cy)
        K[:, 0, 0] *= scale_x           # fx scaled
        K[:, 0, 2] = self.image_size / 2.0   # cx at image center
        K[:, 1, 1] *= scale_y           # fy scaled
        K[:, 1, 2] = self.image_size / 2.0   # cy at image center
        return K

    def _precompute_ray_embeddings(
        self,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> None:
        """Compute and cache Plücker ray coordinates for all patch centers.

        The Plücker coordinates encode fixed camera geometry and are cached as a buffer.
        The trainable MLP projection runs every forward pass to remain differentiable.

        Args:
            camera_intrinsics: (num_views, 3, 3) camera intrinsic matrices
            camera_extrinsics: (num_views, 3, 4) camera extrinsic matrices [R|t]

        """
        patch_size = 16  # standard for all supported ViT backbones
        grid_size = self.image_size // patch_size

        # Compute patch center pixel coordinates in model input space
        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                u = patch_size * j + patch_size / 2.0  # x (column)
                v = patch_size * i + patch_size / 2.0  # y (row)
                coords.append([u, v])

        patch_centers = torch.tensor(
            coords, dtype=torch.float32, device=camera_intrinsics.device,
        )  # (num_patches, 2)

        # Scale intrinsics to model input coordinate system
        K_scaled = self._scale_intrinsics_to_model_input(camera_intrinsics)

        # Compute Plücker ray coordinates
        plucker_coords = PluckerRayEmbedding.compute_plucker_rays(
            intrinsics=K_scaled,
            extrinsics=camera_extrinsics.float(),
            patch_centers=patch_centers,
        )  # (num_views, num_patches, 6)

        # Cache as buffer (survives .to(device), .cuda(), checkpoint save/load)
        self.register_buffer('_cached_plucker_coords', plucker_coords)

    def set_camera_params(
        self,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> None:
        """Set or update camera parameters and recompute ray embeddings.

        Call this if cameras change after model initialization.

        Args:
            camera_intrinsics: (num_views, 3, 3) camera intrinsic matrices
            camera_extrinsics: (num_views, 3, 4) camera extrinsic matrices [R|t]

        """
        self._precompute_ray_embeddings(camera_intrinsics, camera_extrinsics)

    def forward_vit(
        self,
        images: TensorType["view * batch", "channels":3, "image_height", "image_width"],
    ):
        """Override forward pass through the vision encoder to add view and ray embeddings."""

        # this block mostly copies self.vision_encoder.forward(), except for addition of view embed

        # create patch embeddings and add position embeddings; remove CLS token
        try:
            embedding_output = self.backbone.vision_encoder.embeddings(
                images, bool_masked_pos=None, interpolate_pos_encoding=True,
            )[:, 1:]
        except TypeError:
            # DINOv3 doesn't have `interpolate_pos_encoding` arg, does this by default
            embedding_output = self.backbone.vision_encoder.embeddings(
                images, bool_masked_pos=None,
            )[:, 1:]
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

        # Add 3D Plücker ray positional embeddings (geometry-aware, per-patch per-view)
        if self._cached_plucker_coords is not None:
            # MLP forward is differentiable; Plücker coords are fixed geometry
            ray_pe = self.ray_embedding(self._cached_plucker_coords)
            # ray_pe shape: (num_views, num_patches, embedding_dim)
            # Expand for batch: repeat pattern matches images_flat view ordering
            ray_pe_expanded = ray_pe.repeat(batch_size, 1, 1)
            # ray_pe_expanded shape: (view * batch, num_patches, embedding_dim)
            embedding_output = embedding_output + ray_pe_expanded

        # Reshape to (batch, view * num_patches, embedding_dim) so that transformer attention
        # layers process all views simultaneously
        embedding_output = embedding_output.reshape(
            batch_size, self.num_views * num_patches, embedding_dim,
        )

        # push data through vit encoder
        encoder_outputs = self.backbone.vision_encoder.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        outputs = self.backbone.vision_encoder.layernorm(sequence_output)
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

        # Lazy-init ray embeddings from batch camera params if not already initialized.
        # This handles the case where camera params weren't passed at model construction.
        if (
            self._cached_plucker_coords is None
            and "intrinsic_matrix" in batch_dict
            and batch_dict["intrinsic_matrix"].dim() >= 3
        ):
            self._precompute_ray_embeddings(
                camera_intrinsics=batch_dict["intrinsic_matrix"][0],
                camera_extrinsics=batch_dict["extrinsic_matrix"][0],
            )

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
            {"params": self.ray_embedding.parameters(), "name": "ray_embedding"},
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
