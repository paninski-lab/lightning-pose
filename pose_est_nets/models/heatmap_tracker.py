"""Models that produce heatmaps of keypoints from images."""

from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d
from kornia.geometry.transform import pyrup
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Union, Optional, Tuple, TypedDict
from typing_extensions import Literal
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from pose_est_nets.losses.factory import LossFactory
from pose_est_nets.losses.losses import MaskedHeatmapLoss, MaskedRMSELoss
from pose_est_nets.models.base import (
    BaseBatchDict,
    BaseSupervisedTracker,
    HeatmapBatchDict,
    SemiSupervisedTrackerMixin,
)

patch_typeguard()  # use before @typechecked


@typechecked
class HeatmapTracker(BaseSupervisedTracker):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        resnet_version: Literal[18, 34, 50, 101, 152] = 18,
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,  # change
        torch_seed: int = 123,
    ) -> None:
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of backbone model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(  # execute BaseFeatureExtractor.__init__()
            resnet_version=resnet_version,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
        )
        self.num_keypoints = num_keypoints
        self.num_targets = num_keypoints * 2
        self.loss_factory = loss_factory
        # TODO: downsample_factor may be in mismatch between datamodule and model.
        # consider adding support for more types
        self.downsample_factor = downsample_factor
        self.upsampling_layers = self.make_upsampling_layers()
        self.initialize_upsampling_layers()
        self.output_shape = output_shape
        self.temperature = torch.tensor(100, device=self.device)  # soft argmax temp
        self.torch_seed = torch_seed

        # Necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters()

        # use this to log auxiliary information: rmse on labeled data
        self.rmse_loss = RegressionRMSELoss()

    @property
    def num_filters_for_upsampling(self):
        return self.num_fc_input_features

    @property
    def coordinate_scale(self):
        return torch.tensor(2 ** self.downsample_factor, device=self.device)

    @typechecked
    def run_subpixelmaxima(
        self,
        heatmaps: TensorType[
            "batch", "num_keypoints", "heatmap_height", "heatmap_width", float
        ],
    ) -> Tuple[
        TensorType["batch", "num_targets", float],
        TensorType["batch", "num_keypoints", float],
    ]:
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
        # compute predictions as softmax value at argmax
        confidences = torch.amax(softmaxes, dim=(2, 3))
        return preds.reshape(-1, self.num_targets), confidences

    def initialize_upsampling_layers(self) -> None:
        """Intialize the Conv2DTranspose upsampling layers."""
        # TODO: test that running this method changes the weights and biases
        for index, layer in enumerate(self.upsampling_layers):
            if index > 0:  # we ignore the PixelShuffle
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    @typechecked
    def make_upsampling_layers(self) -> torch.nn.Sequential:
        # Note:
        # https://github.com/jgraving/DeepPoseKit/blob/cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/DeepLabCut.py#L131
        # in their model, the pixel shuffle happens only for the
        # downsample_factor=2
        upsampling_layers = [nn.PixelShuffle(2)]
        upsampling_layers.append(
            self.create_double_upsampling_layer(
                in_channels=self.num_filters_for_upsampling // 4,
                out_channels=self.num_keypoints,
            )
        )  # up to here results in downsample_factor=3 for [384,384] images
        if self.downsample_factor == 2:
            upsampling_layers.append(
                self.create_double_upsampling_layer(
                    in_channels=self.num_keypoints,
                    out_channels=self.num_keypoints,
                )
            )
        return nn.Sequential(*upsampling_layers)

    @staticmethod
    @typechecked
    def create_double_upsampling_layer(
        in_channels: int, out_channels: int
    ) -> torch.nn.ConvTranspose2d:
        """Perform ConvTranspose2d to double the output shape.

        Args:
            in_channels: TODO
            out_channels: TODO

        Returns:
            upsampling layer

        """
        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )

    @typechecked
    def heatmaps_from_representations(
        self,
        representations: TensorType[
            "batch",
            "features",
            "rep_height",
            "rep_width",
            float,
        ],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width", float]:
        """Wrapper around self.upsampling_layers for type and shape assertion.

        Args:
            representations: the output of the Resnet feature extractor.

        Returns:
            the result of applying the upsampling layers to the representations
        """
        return self.upsampling_layers(representations)

    @typechecked
    def forward(
        self,
        images: TensorType["batch", "channels":3, "image_height", "image_width", float],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width", float]:
        """Forward pass through the network.

        Args:
            images: images

        Returns:
            heatmap per keypoint

        """
        representations = self.get_representations(images)
        heatmaps = self.heatmaps_from_representations(representations)
        # B = heatmaps.shape[0]
        # valid_probability_heatmaps = self.softmax(
        #     heatmaps.reshape(B, self.num_keypoints, -1)
        # )
        # valid_probability_heatmaps = valid_probability_heatmaps.reshape(
        #     B, self.num_keypoints, self.output_shape[0], self.output_shape[1]
        # )
        return heatmaps

    @typechecked
    def get_loss_inputs_labeled(self, batch_dict: HeatmapBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""

        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)

        return {
            "heatmaps_targ": data_batch["heatmaps"],
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_targ": data_batch["keypoints"],
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }


class SemiSupervisedHeatmapTracker(HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        loss_factory_unsupervised: LossFactory,
        resnet_version: Literal[18, 34, 50, 101, 152] = 18,
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,  # change
        torch_seed: int = 123,
    ):
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of original model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            resnet_version=resnet_version,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            output_shape=output_shape,
            torch_seed=torch_seed,
        )
        self.loss_factory_unsup = loss_factory_unsupervised

        # this attribute will be modified by AnnealWeight callback during training
        self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))

        # Necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters()

    @typechecked
    def get_loss_inputs_unlabeled(self, batch: torch.Tensor) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""

        # images -> heatmaps
        predicted_heatmaps = self.forward(batch)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)

        return {
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }
