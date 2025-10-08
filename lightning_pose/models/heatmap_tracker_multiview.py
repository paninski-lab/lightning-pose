"""Models that produce heatmaps of keypoints from images on multiview datasets."""

import math
from typing import Any, Literal, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType

from lightning_pose.data.cameras import get_valid_projection_masks, project_camera_pairs_to_3d
from lightning_pose.data.datatypes import (
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import convert_bbox_coords, undo_affine_transform_batch
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

        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.backbone.vision_encoder.embeddings(
            images, bool_masked_pos=None, interpolate_pos_encoding=True,
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
            {"params": [self.view_embeddings], "name": "view_embeddings"},
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
