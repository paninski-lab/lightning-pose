"""Models that produce heatmaps of keypoints from images on multiview datasets."""

import math
from typing import Any, Literal, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torchtyping import TensorType

# from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.cameras import get_valid_projection_masks, project_camera_pairs_to_3d
from lightning_pose.data.datatypes import (
    MultiviewHeatmapLabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import convert_bbox_coords
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)
from lightning_pose.models.heads import (
    ALLOWED_MULTIVIEW_HEADS,
    ALLOWED_MULTIVIEW_MULTIHEADS,
    HeatmapHead,
    MultiviewFeatureTransformerHead,
    MultiviewHeatmapCNNHead,
    MultiviewHeatmapCNNMultiHead,
    MultiviewFeatureTransformerHeadLearnableCrossView,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "HeatmapTrackerMultiview",
    "HeatmapTrackerMultiviewMultihead",
    "HeatmapTrackerMultiviewTransformer",
]


class HeatmapTrackerMultiviewTransformer(BaseSupervisedTracker):
    """Transformer network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: Literal["vits_dino", "vitb_dino", "vitb_imagenet"] = "vitb_imagenet",
        pretrained: bool = True,
        # head: Literal["heatmap_cnn"] = "heatmap_cnn",
        head: Literal["heatmap_cnn", "feature_transformer_learnable_crossview"] = "heatmap_cnn",
        downsample_factor: Literal[1] = 1,  # Changed from 2 to 1 to match head configuration
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

        """

        self.num_views = num_views

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            print("HeatmapTrackerMultiviewTransformer does not currently support context frames")
            # del kwargs["do_context"]

        print(f" we are using the Heatmap Multiview Transformer model with {num_views} views")
        print(f" backbone: {backbone}, pretrained: {pretrained}, head: {head}")
        
        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        #Create learnable view embeddings for each view
        self.view_embeddings = nn.Parameter(
            torch.randn(self.num_views, self.num_fc_input_features) * 0.02
        )  # Small initialization for stability

        # self.ray_encoder = nn.Sequential(
        #     nn.Linear(3, self.num_fc_input_features, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.num_fc_input_features, self.num_fc_input_features, bias=False),
        # )
    
        if head == "heatmap_cnn":
            self.head = HeatmapHead(
                backbone_arch=backbone,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        
        elif head == "feature_transformer_learnable_crossview":
            """Multiview Feature Transformer Head with learnable cross-view embeddings."""
            self.head = MultiviewFeatureTransformerHeadLearnableCrossView(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,  # Use model-level downsample_factor
                transformer_d_model=512,
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=3,
                img_size=image_size,
                view_embed_dim=128,
                dropout=0.1,
            )



        else:
            raise NotImplementedError(f"{head} is not a valid multiview transformer head")

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # this attribute will be modified by AnnealWeight callback during training and can be
        # used to weight supervised, non-heatmap losses
        self.total_unsupervised_importance = torch.tensor(1.0)

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward_vit(
        self,
        images: TensorType["view * batch", "channels":3, "image_height", "image_width"],
        intrinsics: TensorType["view * batch", 3, 3],
        extrinsics: TensorType["view * batch", 3, 4],
        distortions: TensorType["view * batch", "num_dist_params"],
    ):
        """Override forward pass through the vision encoder to add view embeddings."""

        # outputs = self.vision_encoder(
        #     x,
        #     return_dict=True,
        #     output_hidden_states=False,
        #     output_attentions=False,
        #     interpolate_pos_encoding=True,
        # ).last_hidden_state

        # -----------------------------------------------------------------------------------------
        # this block mostly copies self.vision_encoder.forward(), except for addition of view embed

        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.backbone.vision_encoder.embeddings(
            images, bool_masked_pos=None, interpolate_pos_encoding=True,
        )[:, 1:]
        # shape: (view * batch, num_patches, embedding_dim)

        # -----------------------------------------------------------------------------------------
        # IMPLEMENTATION: Learnable View Embeddings
        # -----------------------------------------------------------------------------------------
       

        # -----------------------------------------------------------------------------------------

        # get dims for reshaping
        view_batch_size = embedding_output.shape[0]
        num_patches = embedding_output.shape[1]
        embedding_dim = embedding_output.shape[2]
        batch_size = view_batch_size // self.num_views


        # intrinsics = intrinsics.float() # [batch*views, 3, 3]
        # extrinsics = extrinsics.float() # [batch*views, 3, 4]
        # distortions = distortions.float() # [batch*views, 5] 

        # # CHANGE: Get actual image size instead of hardcoding 256
        # img_height, img_width = images.shape[-2:]  # Get from input images
        
        # # Normalize camera parameters with actual image dimensions
        # intrinsics_norm = intrinsics.clone()
        # intrinsics_norm[:, 0, 0] /= img_width   # fx normalized by width
        # intrinsics_norm[:, 1, 1] /= img_height  # fy normalized by height  
        # intrinsics_norm[:, 0, 2] /= img_width   # cx normalized by width
        # intrinsics_norm[:, 1, 2] /= img_height  # cy normalized by height
        
        # extrinsics_norm = extrinsics.clone()
        # extrinsics_norm[:, :, 3] /= 100.0  # Scale translation
        # distortions_norm = torch.clamp(distortions, -1.0, 1.0)


        # camera_params = torch.cat([
        #     intrinsics_norm.view(-1, 9),      # 3x3 intrinsics flattened
        #     extrinsics_norm.view(-1, 12),     # 3x4 extrinsics flattened  
        #     distortions_norm.view(-1, distortions_norm.shape[-1]),    # distortion parameters
        # ], dim=1)

        
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
        # Add view embeddings to patch embeddings
        embedding_output = embedding_output + view_embeddings_expanded
        
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

        # extract pixel data from batch
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]
            
            # CRITICAL FIX: Ensure view ordering for DALI multiview data
            # DALI processes views in arbitrary order, but we need them in view_names order
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'view_names'):
                expected_view_names = self.cfg.data.view_names
                if len(expected_view_names) == images.shape[1]:
                    print(f"WARNING: DALI multiview data detected. Expected view order: {expected_view_names}")
                    print(f"Current batch shape: {images.shape}")
                    # TODO: Add view reordering logic here if needed

        batch_size, num_views, channels, img_height, img_width = images.shape

        # extract camera parameters from batch (optional for video prediction)
        if "intrinsic_matrix" in batch_dict:
            intrinsics = batch_dict["intrinsic_matrix"]
            extrinsics = batch_dict["extrinsic_matrix"]
            distortions = batch_dict["distortions"]
            
            # stack batch and view into first dim to pass through transformer
            images = images.reshape(-1, channels, img_height, img_width)
            if intrinsics.shape[1] == 1:  # Placeholder case [batch, 1]
                # First reshape to add the missing dimensions
                intrinsics = intrinsics.reshape(-1, 1, 3, 3)  # [16, 1, 3, 3]
                extrinsics = extrinsics.reshape(-1, 1, 3, 4)  # [16, 1, 3, 4]
                distortions = distortions.reshape(-1, 1, 5)  # [16,1,5]
                
                # Then expand the second dimension
                intrinsics = intrinsics.expand(-1, self.num_views, -1, -1).reshape(-1, 3, 3)
                extrinsics = extrinsics.expand(-1, self.num_views, -1, -1).reshape(-1, 3, 4)
                distortions = distortions.expand(-1, self.num_views, -1).reshape(-1, distortions.shape[-1])
            else:  # Normal case [batch, num_views, ...]
                # Reshape real camera parameters
                intrinsics = intrinsics.reshape(-1, 3, 3)
                extrinsics = extrinsics.reshape(-1, 3, 4)
                distortions = distortions.reshape(-1, distortions.shape[-1])
        else:
            # For video prediction without camera parameters, create dummy parameters
            # These won't be used in the forward pass but are needed for the function signature
            images = images.reshape(-1, channels, img_height, img_width)
            batch_size_total = images.shape[0]
            device = images.device
            
            # Create dummy camera parameters
            intrinsics = torch.eye(3, device=device).unsqueeze(0).expand(batch_size_total, -1, -1)
            extrinsics = torch.eye(3, 4, device=device).unsqueeze(0).expand(batch_size_total, -1, -1)
            distortions = torch.zeros(batch_size_total, 5, device=device)

        # pass through transformer to get base representations
        representations = self.forward_vit(images, intrinsics, extrinsics, distortions)
        # shape: (view * batch, num_features, rep_height, rep_width)

        # get heatmaps for each representation
        if hasattr(self.head, 'forward') and self.head.__class__.__name__ == 'MultiviewFeatureTransformerHeadLearnableCrossView':
            heatmaps = self.head(representations, self.num_views)
        else:
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
        # if isinstance(batch_dict, dict) and "images" in batch_dict:
        #     # Convert to proper type for labeled data
        #     labeled_batch = MultiviewHeatmapLabeledBatchDict(**batch_dict)
        #     pred_heatmaps = self.forward(labeled_batch)
        # else:
        #     # Handle unlabeled data - this should not happen for this model
        #     raise ValueError("HeatmapTrackerMultiviewTransformer expects labeled data")
        
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
        
        # params = [
        #     {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
        #     {"params": self.head.parameters(), "name": "head", "lr": 5e-4},  # Middle ground: 5e-4
        #     {"params": [self.view_embeddings], "name": "view_embeddings", "lr": 5e-4},  # Same as head
        # ]
        return params


class HeatmapTrackerMultiview(BaseSupervisedTracker):
    """Convolutional network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        head: ALLOWED_MULTIVIEW_MULTIHEADS = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        image_size: int = 256,
        **kwargs: Any,
    ):
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            head: architecture used to fuse view information
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError(
                "HeatmapTrackerMultiviewHeatmapCNN currently only implements downsample_factor=2"
            )

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            del kwargs["do_context"]

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        if head == "heatmap_cnn":
            self.head = MultiviewHeatmapCNNHead(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
        elif head == "feature_transformer":
            self.head = MultiviewFeatureTransformerHead(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                transformer_d_model=512,
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=4,
                img_size=image_size,
            )
        
        elif head == "feature_transformer_learnable_crossview":
            self.head = MultiviewFeatureTransformerHeadLearnableCrossView(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                transformer_d_model=512,
                transformer_nhead=8,
                transformer_dim_feedforward=512,
                transformer_num_layers=3,
                img_size=image_size,
                view_embed_dim=128,
                dropout=0.1,
            )

        else:
            raise NotImplementedError(
                f"{head} is not a valid multiview head, choose from {ALLOWED_MULTIVIEW_HEADS}"
            )

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # this attribute will be modified by AnnealWeight callback during training and can be
        # used to weight supervised, non-heatmap losses
        self.total_unsupervised_importance = torch.tensor(1.0)

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: TensorType["batch", "view", "channels":3, "image_height", "image_width"],
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        batch_size, num_views, channels, img_height, img_width = images.shape

        # stack batch and view into first dim to get representations
        images = images.reshape(-1, channels, img_height, img_width)
        representations = self.get_representations(images)
        # representations shape is (view * batch, num_features, rep_height, rep_width)

        # get heatmaps for each representation
        heatmaps = self.head(representations, num_views)

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
        if batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints.shape[1] // 2 // num_views
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
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]

        # images -> heatmaps
        pred_heatmaps = self.forward(images)
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
        ]
        return params


class HeatmapTrackerMultiviewMultihead(BaseSupervisedTracker):
    """Multi-headed convolutional network that handles multi-view datasets."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        head: ALLOWED_MULTIVIEW_MULTIHEADS = "heatmap_cnn",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | dict | None = None,
        **kwargs: Any,
    ):
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            num_views: number of camera views
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            head: architecture used to fuse view information
                - heatmap_cnn
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                - multisteplr: milestones, gamma

        """

        if downsample_factor != 2:
            raise NotImplementedError(
                "HeatmapTrackerMultiviewHeatmapCNN currently only implements downsample_factor=2"
            )

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            del kwargs["do_context"]

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=False,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.downsample_factor = downsample_factor

        if head == "heatmap_cnn":
            self.head = MultiviewHeatmapCNNMultiHead(
                backbone_arch=backbone,
                num_views=num_views,
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
                upsampling_factor=1 if "vit" in backbone else 2,
            )
        else:
            raise NotImplementedError(
                f"{head} is not a valid multiview head, choose from {ALLOWED_MULTIVIEW_MULTIHEADS}"
            )

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # this attribute will be modified by AnnealWeight callback during training and can be
        # used to weight supervised, non-heatmap losses
        self.total_unsupervised_importance = torch.tensor(1.0)

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: TensorType["batch", "view", "channels":3, "image_height", "image_width"],
    ) -> Tuple[
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Forward pass through the network.

        Batch options
        -------------
        - TensorType["batch", "view", "channels":3, "image_height", "image_width"]
          multiview labeled batch or unlabeled batch from DALI

        """

        batch_size, num_views, channels, img_height, img_width = images.shape

        # stack batch and view into first dim to get representations
        images = images.reshape(-1, channels, img_height, img_width)
        representations = self.get_representations(images)
        # representations shape is (view * batch, num_features, rep_height, rep_width)

        # get two heatmaps for each representation (single view, multi-view)
        heatmaps_sv, heatmaps_mv = self.head(representations, num_views)

        return heatmaps_sv, heatmaps_mv

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_sv, pred_heatmaps_mv = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints_sv, confidence_sv = self.head.run_subpixelmaxima(pred_heatmaps_sv)
        pred_keypoints_mv, confidence_mv = self.head.run_subpixelmaxima(pred_heatmaps_mv)
        # bounding box coords -> original image coords
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints_sv = convert_bbox_coords(batch_dict, pred_keypoints_sv)
        pred_keypoints_mv = convert_bbox_coords(batch_dict, pred_keypoints_mv)
        # project predictions from pairs of views into 3d if calibration data available
        if batch_dict["keypoints_3d"].shape[-1] == 3:
            num_views = batch_dict["images"].shape[1]
            num_keypoints = pred_keypoints_sv.shape[1] // 2 // num_views
            pred_keypoints_3d_sv = project_camera_pairs_to_3d(
                points=pred_keypoints_sv.reshape((-1, num_views, num_keypoints, 2)),
                intrinsics=batch_dict["intrinsic_matrix"],
                extrinsics=batch_dict["extrinsic_matrix"],
                dist=batch_dict["distortions"],
            )
            pred_keypoints_3d_mv = project_camera_pairs_to_3d(
                points=pred_keypoints_mv.reshape((-1, num_views, num_keypoints, 2)),
                intrinsics=batch_dict["intrinsic_matrix"],
                extrinsics=batch_dict["extrinsic_matrix"],
                dist=batch_dict["distortions"],
            )
            keypoints_pred_3d = torch.cat([pred_keypoints_3d_sv, pred_keypoints_3d_mv])
            keypoints_targ_3d = torch.cat([batch_dict["keypoints_3d"], batch_dict["keypoints_3d"]])

            keypoints_mask_3d_ = get_valid_projection_masks(
                target_keypoints.reshape((-1, num_views, num_keypoints, 2))
            )
            keypoints_mask_3d = torch.cat([keypoints_mask_3d_, keypoints_mask_3d_])
        else:
            keypoints_pred_3d = None
            keypoints_targ_3d = None
            keypoints_mask_3d = None

        return {
            "heatmaps_targ": torch.cat([batch_dict["heatmaps"], batch_dict["heatmaps"]], dim=0),
            "heatmaps_pred": torch.cat([pred_heatmaps_sv, pred_heatmaps_mv], dim=0),
            "keypoints_targ": torch.cat([target_keypoints, target_keypoints], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_sv, pred_keypoints_mv], dim=0),
            "confidences": torch.cat([confidence_sv, confidence_mv], dim=0),
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
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]

        # images -> heatmaps
        pred_heatmaps_sv, pred_heatmaps_mv = self.forward(images)
        # heatmaps -> keypoints
        pred_keypoints_sv, confidence_sv = self.head.run_subpixelmaxima(pred_heatmaps_sv)
        pred_keypoints_mv, confidence_mv = self.head.run_subpixelmaxima(pred_heatmaps_mv)

        # reshape keypoints to be (batch, n_keypoints, 2)
        pred_keypoints_sv = pred_keypoints_sv.reshape(pred_keypoints_sv.shape[0], -1, 2)
        pred_keypoints_mv = pred_keypoints_mv.reshape(pred_keypoints_mv.shape[0], -1, 2)

        # find higher confidence indices
        mv_conf_gt = torch.gt(confidence_mv, confidence_sv)

        # select higher confidence indices
        pred_keypoints_sv[mv_conf_gt] = pred_keypoints_mv[mv_conf_gt]
        pred_keypoints_sv = pred_keypoints_sv.reshape(pred_keypoints_sv.shape[0], -1)

        confidence_sv[mv_conf_gt] = confidence_mv[mv_conf_gt]

        # bounding box coords -> original image coords
        pred_keypoints_sv = convert_bbox_coords(batch_dict, pred_keypoints_sv)

        if return_heatmaps:
            pred_heatmaps_sv[mv_conf_gt] = pred_heatmaps_mv[mv_conf_gt]
            return pred_keypoints_sv, confidence_sv, pred_heatmaps_sv
        else:
            return pred_keypoints_sv, confidence_sv

    def get_parameters(self):
        params = [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
        ]
        return params
