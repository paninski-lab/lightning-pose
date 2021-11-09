"""Models that produce heatmaps of keypoints from images."""

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing_extensions import Literal

from pose_est_nets.losses.losses import (
    convert_dict_entries_to_tensors,
    get_losses_dict,
    MaskedMSEHeatmapLoss,
    MaskedRMSELoss,
)
from pose_est_nets.models.base_resnet import BaseFeatureExtractor
from pose_est_nets.utils.heatmap_tracker_utils import (
    find_subpixel_maxima,
    largest_factor,
    SubPixelMaxima,
)

patch_typeguard()  # use before @typechecked


@typechecked
class HeatmapTracker(BaseFeatureExtractor):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_targets: int,
        resnet_version: Literal[18, 34, 50, 101, 152] = 18,
        downsample_factor: Literal[
            2, 3
        ] = 2,  # TODO: downsample_factor may be in mismatch between datamodule and model. consider adding support for more types
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,  # change
        output_sigma: float = 1.25,  # check value
        upsample_factor: int = 100,
        confidence_scale: float = 1.0,
        threshold: Optional[float] = None,
        torch_seed: int = 123,
    ) -> None:
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_targets: number of body parts times 2 (x,y) coords
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of backbone model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            output_sigma: TODO
            upsample_factor: TODO
            confidence_scale: TODO
            threshold: TODO
            torch_seed: make weight initialization reproducible

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(  # execute BaseFeatureExtractor.__init__()
            resnet_version=resnet_version,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
        )
        self.num_targets = num_targets
        self.downsample_factor = downsample_factor
        self.upsampling_layers = self.make_upsampling_layers()
        self.initialize_upsampling_layers()
        self.output_shape = output_shape
        self.output_sigma = torch.tensor(output_sigma, device=self.device)
        self.upsample_factor = torch.tensor(upsample_factor, device=self.device)
        self.confidence_scale = torch.tensor(confidence_scale, device=self.device)
        self.threshold = threshold
        self.torch_seed = torch_seed
        # Necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters()
        # self.device = device  # done automatically by pytorch lightning?

    @property
    def num_keypoints(self):
        return self.num_targets // 2

    @property
    def num_filters_for_upsampling(self):
        return self.base.fc.in_features

    @property
    def coordinate_scale(self):
        return torch.tensor(2 ** self.downsample_factor, device=self.device)

    @property
    def SubPixMax(self):
        return SubPixelMaxima(
            output_shape=self.output_shape,
            output_sigma=self.output_sigma,
            upsample_factor=self.upsample_factor,
            coordinate_scale=self.coordinate_scale,
            confidence_scale=self.confidence_scale,
            threshold=self.threshold,
            device=self.device,
        )

    def run_subpixelmaxima(self, heatmaps1):
        return self.SubPixMax.run(heatmaps1)

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
        )  #  up to here results in downsample_factor=3 for [384,384] images
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
        self, images: TensorType["batch", "channels":3, "image_height", "image_width"]
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network.

        Args:
            images: images

        Returns:
            heatmap per keypoint

        """
        representations = self.get_representations(images)
        heatmaps = self.heatmaps_from_representations(representations)
        return heatmaps

    @typechecked
    def training_step(self, data_batch: Dict, batch_idx: int) -> Dict:

        # forward pass
        predicted_heatmaps = self.forward(data_batch["images"])  # images -> heatmaps
        predicted_keypoints, confidence = self.run_subpixelmaxima(
            predicted_heatmaps
        )  # heatmaps -> keypoints

        # compute loss
        heatmap_loss = MaskedMSEHeatmapLoss(data_batch["heatmaps"], predicted_heatmaps)
        supervised_rmse = MaskedRMSELoss(data_batch["keypoints"], predicted_keypoints)

        # log training loss + rmse
        self.log("train_loss", heatmap_loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)

        return {"loss": heatmap_loss}

    @typechecked
    def evaluate(
        self,
        data_batch: Dict,
        stage: Optional[Literal["val", "test"]] = None
    ) -> None:
        predicted_heatmaps = self.forward(data_batch["images"])  # images -> heatmaps
        predicted_keypoints, confidence = self.run_subpixelmaxima(
            predicted_heatmaps
        )  # heatmaps -> keypoints
        loss = MaskedMSEHeatmapLoss(data_batch["heatmaps"], predicted_heatmaps)
        supervised_rmse = MaskedRMSELoss(data_batch["keypoints"], predicted_keypoints)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
            self.log(
                f"{stage}_supervised_rmse", supervised_rmse, prog_bar=True, logger=True
            )

    def validation_step(self, validation_batch: Dict, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: Dict, batch_idx):
        self.evaluate(test_batch, "test")


class SemiSupervisedHeatmapTracker(HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_targets: int,
        resnet_version: Literal[18, 34, 50, 101, 152] = 18,
        downsample_factor: Literal[
            2, 3
        ] = 2,  # TODO: downsample_factor may be in mismatch between datamodule and model. consider adding support for more types
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,  # change
        output_sigma: float = 1.25,  # check value,
        upsample_factor: int = 100,
        confidence_scale: float = 1.0,
        threshold: Optional[float] = None,
        torch_seed: int = 123,
        loss_params: Optional[
            dict
        ] = None,  # optional so we can initialize a model without passing that in
        semi_super_losses_to_use: Optional[list] = None,
    ):
        """

        Args:
            num_targets: number of body parts times 2 (x,y) coords
            resnet_version: ResNet variant to be used (e.g. 18, 34, 50, 101,
                or 152); essentially specifies how large the resnet will be
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of original model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            output_sigma: TODO
            upsample_factor: TODO
            confidence_scale: TODO
            threshold: TODO
            torch_seed: make weight initialization reproducible
            loss_params: TODO
            semi_super_losses_to_use: TODO

        """
        super().__init__(
            num_targets=num_targets,
            resnet_version=resnet_version,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            output_shape=output_shape,
            output_sigma=output_sigma,
            upsample_factor=upsample_factor,
            confidence_scale=confidence_scale,
            threshold=threshold,
            torch_seed=torch_seed,
        )
        print(semi_super_losses_to_use)
        self.loss_function_dict = get_losses_dict(semi_super_losses_to_use)
        self.loss_params = loss_params

    @typechecked
    def training_step(self, data_batch: Dict, batch_idx: int) -> Dict:

        # forward pass labeled
        predicted_heatmaps = self.forward(data_batch["labeled"]['images'])
        predicted_keypoints, confidence = self.run_subpixelmaxima(
            predicted_heatmaps
        )  # heatmaps -> keypoints

        # compute loss labeled
        supervised_loss = MaskedMSEHeatmapLoss(
            data_batch["labeled"]["heatmaps"],
            predicted_heatmaps
        )
        supervised_rmse = MaskedRMSELoss(
            data_batch["labeled"]["keypoints"],
            predicted_keypoints
        )

        # forward pass unlabeled
        unlabeled_predicted_heatmaps = self.forward(data_batch["unlabeled"])
        predicted_us_keypoints, confidence = self.run_subpixelmaxima(
            unlabeled_predicted_heatmaps
        )

        # loop over unsupervised losses
        tot_loss = 0.0
        tot_loss += supervised_loss
        self.loss_params = convert_dict_entries_to_tensors(
            self.loss_params, self.device
        )
        for loss_name, loss_func in self.loss_function_dict.items():
            # Some losses use keypoint_preds, some use heatmap_preds, and some use both.
            # all have **kwargs so are robust to unneeded inputs."
            add_loss = self.loss_params[loss_name]["weight"] * loss_func(
                keypoint_preds = predicted_us_keypoints, 
                heatmap_preds = unlabeled_predicted_heatmaps,
                **self.loss_params[loss_name]
            )
            tot_loss += add_loss
            # log individual unsupervised losses
            self.log(loss_name + "_loss", add_loss, prog_bar=True)

        # log other losses
        self.log("total_loss", tot_loss, prog_bar=True)
        self.log("supervised_loss", supervised_loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)

        return {"loss": tot_loss}
