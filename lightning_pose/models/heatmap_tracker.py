"""Models that produce heatmaps of keypoints from images."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from kornia.filters import filter2d
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d
from kornia.geometry.transform.pyramid import _get_pyramid_gaussian_kernel
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.utils import (
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
    evaluate_heatmaps_at_location,
    undo_affine_transform_batch,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
    convert_bbox_coords,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "upsample",
    "HeatmapTracker",
    "SemiSupervisedHeatmapTracker",
]


def upsample(
    inputs: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
) -> TensorType["batch", "num_keypoints", "two_x_heatmap_height", "two_x_heatmap_width"]:
    """Upsample batch of heatmaps using interpolation (no learned weights).

    This is a copy of kornia's pyrup function but with better defaults.
    """
    kernel = _get_pyramid_gaussian_kernel()
    _, _, height, width = inputs.shape
    # align_corners=False is important!! otherwise the offsets below don't hold
    inputs_up = nn.functional.interpolate(
        inputs, size=(height * 2, width * 2), mode="bicubic", align_corners=False,
    )
    inputs_up = filter2d(inputs_up, kernel, border_type="constant")
    return inputs_up


class HeatmapTracker(BaseSupervisedTracker):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        num_targets: int = None,
        loss_factory: Optional[LossFactory] = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        output_shape: Optional[tuple] = None,  # change
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            pretrained: True to load pretrained imagenet weights
            output_shape: hard-coded image size to avoid dynamic shape computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )
        self.num_keypoints = num_keypoints
        if num_targets is None:
            self.num_targets = num_keypoints * 2
        else:
            self.num_targets = num_targets
        self.loss_factory = loss_factory
        # TODO: downsample_factor may be in mismatch between datamodule and model.
        self.downsample_factor = downsample_factor
        self.upsampling_layers = self.make_upsampling_layers()
        self.initialize_upsampling_layers()
        self.output_shape = output_shape
        # TODO: temp=1000 works for 64x64 heatmaps, need to generalize to other shapes
        self.temperature = torch.tensor(1000.0, device=self.device)  # soft argmax temp
        self.torch_seed = torch_seed

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    @property
    def num_filters_for_upsampling(self) -> int:
        return self.num_fc_input_features

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
            heatmaps = upsample(heatmaps)
        # find soft argmax
        softmaxes = spatial_softmax2d(heatmaps, temperature=self.temperature)
        preds = spatial_expectation2d(softmaxes, normalized_coordinates=False)
        # compute confidences as softmax value pooled around prediction
        confidences = evaluate_heatmaps_at_location(heatmaps=softmaxes, locs=preds)
        # fix grid offsets from upsampling
        if self.downsample_factor == 1:
            preds -= 0.5
        elif self.downsample_factor == 2:
            preds -= 1.5
        elif self.downsample_factor == 3:
            preds -= 2.5

        # NOTE: we cannot use
        # `preds.reshape(-1, self.num_targets)`
        # This works fine for the non-multiview case
        # This works fine for multiview training
        # This fails during multiview inference when we might have an arbitrary number of views
        # that we are processing (self.num_targets is tied to the labeled data)
        return preds.reshape(-1, heatmaps.shape[1] * 2), confidences

    def run_hard_argmax(
        self,
        heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ) -> Tuple[TensorType["batch", "num_targets"], TensorType["batch", "num_keypoints"]]:
        """Use hard argmax on heatmaps.

        Args:
            heatmaps: output of upsampling layers

        Returns:
            tuple
                - hard argmax of shape (batch, num_targets)
                - confidences of shape (batch, num_keypoints)

        """

        # upsample heatmaps
        for _ in range(self.downsample_factor):
            heatmaps = upsample(heatmaps)
        # find hard argmax
        softmaxes = spatial_softmax2d(heatmaps, temperature=self.temperature)
        preds = self._spatial_argmax2d(softmaxes)
        # compute confidences as softmax value pooled around prediction
        confidences = evaluate_heatmaps_at_location(heatmaps=softmaxes, locs=preds)
        # fix grid offsets from upsampling
        if self.downsample_factor == 1:
            preds -= 0.5
        elif self.downsample_factor == 2:
            preds -= 1.5
        elif self.downsample_factor == 3:
            preds -= 2.5

        return preds.reshape(-1, self.num_targets), confidences

    @staticmethod
    def _spatial_argmax2d(heatmaps):
        flat_indexes = heatmaps.flatten(start_dim=-2).argmax(-1)
        B = heatmaps.shape[0]
        N = heatmaps.shape[1]
        peaks = torch.zeros(B, N, 2, device=heatmaps.device, dtype=torch.float32)
        for i in range(B):
            for j in range(N):
                idxs_ = divmod(flat_indexes[i, j].item(), heatmaps.shape[-1])
                peaks[i, j, 0] = idxs_[1]  # x coords
                peaks[i, j, 1] = idxs_[0]  # y coords
        return peaks

    def initialize_upsampling_layers(self) -> None:
        """Intialize the Conv2DTranspose upsampling layers."""
        for index, layer in enumerate(self.upsampling_layers):
            if index > 0:  # we ignore the PixelShuffle
                if isinstance(layer, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
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
        n_layers_to_build = 4 - self.downsample_factor - 1
        if self.backbone_arch in ["vit_h_sam", "vit_b_sam"]:
            n_layers_to_build = -1
        for _ in range(n_layers_to_build):
            # add upsampling layer to account for heatmap downsampling
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
        representations: TensorType["batch", "features", "rep_height", "rep_width"],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Upsample representations to get final heatmaps."""
        heatmaps = self.upsampling_layers(representations)
        return heatmaps

    def forward(
        self,
        images: Union[
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "views", "channels":3, "image_height", "image_width"],
        ]
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network."""
        # we get one representation for each desired output.
        shape = images.shape

        # if len(shape) > 4 we assume we have multiple views and need to combine images across
        # batch/views before passing to network, then we reshape
        if len(shape) > 4:
            images = images.reshape(-1, shape[-3], shape[-2], shape[-1])
            representations = self.get_representations(images)
            heatmaps = self.heatmaps_from_representations(representations)
            heatmaps = heatmaps.reshape(shape[0], -1, heatmaps.shape[-2], heatmaps.shape[-1])
        else:
            representations = self.get_representations(images)
            heatmaps = self.heatmaps_from_representations(representations)

        # softmax temp stays 1 here; to modify for model predictions, see constructor
        return spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))

    def get_loss_inputs_labeled(
        self,
        batch_dict: Union[
            HeatmapLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        # bounding box coords -> original image coords
        predicted_keypoints = convert_bbox_coords(batch_dict, predicted_keypoints)
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }

    def predict_step(
        self,
        batch_dict: Union[
            HeatmapLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
            UnlabeledBatchDict,
        ],
        batch_idx: int,
        return_heatmaps: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
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
        predicted_heatmaps = self.forward(images)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        # bounding box coords -> original image coords
        predicted_keypoints = convert_bbox_coords(batch_dict, predicted_keypoints)
        if return_heatmaps:
            return predicted_keypoints, confidence, predicted_heatmaps
        else:
            return predicted_keypoints, confidence


@typechecked
class SemiSupervisedHeatmapTracker(SemiSupervisedTrackerMixin, HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: Optional[LossFactory] = None,
        loss_factory_unsupervised: Optional[LossFactory] = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        output_shape: Optional[tuple] = None,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        **kwargs: Any,
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
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            output_shape=output_shape,
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )
        self.loss_factory_unsup = loss_factory_unsupervised

        # this attribute will be modified by AnnealWeight callback during training
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))
        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(
        self,
        batch_dict: Union[
            UnlabeledBatchDict,
            MultiviewUnlabeledBatchDict,
        ],
    ) -> Dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["frames"])
        # heatmaps -> keypoints
        predicted_keypoints_augmented, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        # undo augmentation if needed
        predicted_keypoints = undo_affine_transform_batch(
            keypoints_augmented=predicted_keypoints_augmented,
            transforms=batch_dict["transforms"],
            is_multiview=batch_dict["is_multiview"],
        )
        # keypoints -> original image coords keypoints
        predicted_keypoints = convert_bbox_coords(batch_dict, predicted_keypoints)
        return {
            "heatmaps_pred": predicted_heatmaps,  # if augmented, augmented heatmaps
            "keypoints_pred": predicted_keypoints,  # if augmented, original keypoints
            "keypoints_pred_augmented": predicted_keypoints_augmented,  # match heatmaps_pred
            "confidences": confidence,
        }
