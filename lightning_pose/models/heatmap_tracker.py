"""Models that produce heatmaps of keypoints from images."""

from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d
from kornia.geometry.transform import pyrup
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union
from typing_extensions import Literal
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from lightning_pose.data.utils import (
    evaluate_heatmaps_at_location, undo_affine_transform,
    BaseLabeledBatchDict, HeatmapLabeledBatchDict, UnlabeledBatchDict
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import BaseSupervisedTracker, SemiSupervisedTrackerMixin

patch_typeguard()  # use before @typechecked


@typechecked
class HeatmapTracker(BaseSupervisedTracker):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnet50_3d", "resnet50_contrastive", "resnet50_animal_apose", "resnet50_animal_ap10k", 
            "resnet50_human_jhmdb", "resnet50_human_res_rle", "resnet50_human_top_res"
        ] = "resnet50",
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,  # change
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = False,
    ) -> None:
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of backbone model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=do_context,
        )
        self.num_keypoints = num_keypoints
        self.num_targets = num_keypoints * 2
        self.loss_factory = loss_factory
        # TODO: downsample_factor may be in mismatch between datamodule and model.
        self.downsample_factor = downsample_factor
        self.upsampling_layers = self.make_upsampling_layers()
        self.initialize_upsampling_layers()
        self.output_shape = output_shape
        # TODO: temp=1000 works for 64x64 heatmaps, need to generalize to other shapes
        self.temperature = torch.tensor(1000.0, device=self.device)  # soft argmax temp
        self.torch_seed = torch_seed
        self.do_context = do_context
        if self.mode == "2d":
            self.unnormalized_weights = nn.parameter.Parameter(
                torch.Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]), requires_grad=False)
            self.representation_fc = lambda x: x @ torch.transpose(
                nn.functional.softmax(self.unnormalized_weights, dim=1), 0, 1)
        elif self.mode == "3d":
            self.unnormalized_weights = nn.parameter.Parameter(
                torch.Tensor([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]),
                requires_grad=False
            )
            self.representation_fc = lambda x: x @ torch.transpose(
                nn.functional.softmax(self.unnormalized_weights, dim=1), 0, 1)

        # use this to log auxiliary information: rmse on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters(ignore="loss_factory")  # cannot be pickled

    @property
    def num_filters_for_upsampling(self) -> int:
        return self.num_fc_input_features

    @property
    def coordinate_scale(self) -> TensorType[(), int]:
        return torch.tensor(2**self.downsample_factor, device=self.device)

    def run_subpixelmaxima(
        self,
        heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ) -> Tuple[TensorType["batch", "num_targets"], TensorType["batch", "num_keypoints"]]:
        """Use soft argmax on heatmaps.

        Args:
            heatmaps: output of upsampling layers

        Returns:
            tuple
                - soft argmax of shape (batch, num_targets)
                - confidences of shape (batch, num_keypoints)

        """
        # upsample heatmaps
        for _ in range(self.downsample_factor):
            heatmaps = pyrup(heatmaps)

        # find soft argmax
        softmaxes = spatial_softmax2d(heatmaps, temperature=self.temperature)
        preds = spatial_expectation2d(softmaxes, normalized_coordinates=False)

        # compute confidences as softmax value pooled around prediction
        confidences = evaluate_heatmaps_at_location(heatmaps=softmaxes, locs=preds)

        return preds.reshape(-1, self.num_targets), confidences

    def initialize_upsampling_layers(self) -> None:
        """Intialize the Conv2DTranspose upsampling layers."""
        # TODO: test that running this method changes the weights and biases
        for index, layer in enumerate(self.upsampling_layers):
            if index > 0:  # we ignore the PixelShuffle
                if isinstance(layer, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    torch.nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    torch.nn.init.constant_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)

    def make_upsampling_layers(self) -> torch.nn.Sequential:
        # Note:
        # https://github.com/jgraving/DeepPoseKit/blob/
        # cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/DeepLabCut.py#L131
        # in their model, the pixel shuffle happens only for downsample_factor=2
        upsampling_layers = []
        upsampling_layers.append(nn.PixelShuffle(2))
        # upsampling_layers.append(nn.BatchNorm2d(self.num_filters_for_upsampling // 4))
        # upsampling_layers.append(nn.ReLU(inplace=True))
        upsampling_layers.append(
            self.create_double_upsampling_layer(
                in_channels=self.num_filters_for_upsampling // 4,
                out_channels=self.num_keypoints,
            )
        )  # up to here results in downsample_factor=3
        if self.downsample_factor == 2:
            # upsampling_layers.append(nn.BatchNorm2d(self.num_keypoints))
            # upsampling_layers.append(nn.ReLU(inplace=True))
            upsampling_layers.append(
                self.create_double_upsampling_layer(
                    in_channels=self.num_keypoints,
                    out_channels=self.num_keypoints,
                )
            )
        return nn.Sequential(*upsampling_layers)

    @staticmethod
    def create_double_upsampling_layer(
        in_channels: int,
        out_channels: int,
    ) -> torch.nn.ConvTranspose2d:
        """Perform ConvTranspose2d to double the output shape."""
        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )

    def heatmaps_from_representations(
        self,
        representations: Union[
            TensorType["batch", "features", "rep_height", "rep_width"],
            TensorType["batch", "features", "rep_height", "rep_width", "frames"],
        ],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Handle context frames then upsample to get final heatmaps."""

        # handle context frames first
        if (self.mode == "2d" and self.do_context) or self.mode == "3d":
            # push through a linear layer to get the final representation
            # input shape (batch, features, rep_height, rep_width, frames)
            representations: TensorType[
                "batch", "features", "rep_height", "rep_width", "frames"
            ] = self.representation_fc(representations)
            # final squeeze
            representations: TensorType[
                "batch", "features", "rep_height", "rep_width"
            ] = torch.squeeze(representations, 4)

        # upsample representations to get heatmaps
        heatmaps = self.upsampling_layers(representations)

        return heatmaps

    def forward(
        self,
        images: Union[
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"]
        ],
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network."""
        # we get one representation for each desired output. 
        # in the case of unsupervised sequences + context, we have outputs for all images but the
        # first two and last two.
        # this is all handled internally by get_representations()
        representations = self.get_representations(images)
        heatmaps = self.heatmaps_from_representations(representations)
        # softmax temp stays 1 here; to modify for model predictions, see constructor
        return spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))

    def get_loss_inputs_labeled(self, batch_dict: HeatmapLabeledBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_targ": batch_dict["keypoints"],
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }
    
    def predict_step(
        self,
        batch: Union[HeatmapLabeledBatchDict, UnlabeledBatchDict],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict heatmaps and keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        if "images" in batch.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch["frames"]
        # images -> heatmaps
        predicted_heatmaps = self.forward(images)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        return predicted_keypoints, confidence


@typechecked
class SemiSupervisedHeatmapTracker(SemiSupervisedTrackerMixin, HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        loss_factory_unsupervised: LossFactory,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnet50_3d", "resnet50_contrastive", "resnet50_animal_apose", "resnet50_animal_ap10k", 
            "resnet50_human_jhmdb", "resnet50_human_res_rle", "resnet50_human_top_res"
        ] = "resnet50",
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = False,
    ) -> None:
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of original model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            output_shape=output_shape,
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=do_context,
        )
        self.loss_factory_unsup = loss_factory_unsupervised.to(self.device)

        # this attribute will be modified by AnnealWeight callback during training
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))
        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(self, batch: UnlabeledBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch["frames"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        # undo augmentation if needed
        if batch["transforms"].shape[-1] == 3:
            # reshape to (seq_len, n_keypoints, 2)
            pred_kps = torch.reshape(predicted_keypoints, (predicted_keypoints.shape[0], -1, 2))
            # undo
            pred_kps = undo_affine_transform(pred_kps, batch["transforms"])
            # reshape to (seq_len, n_keypoints * 2)
            predicted_keypoints = torch.reshape(pred_kps, (pred_kps.shape[0], -1))
        return {
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }
