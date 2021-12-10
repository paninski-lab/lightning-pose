"""Models that produce heatmaps of keypoints from images."""

from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d
from kornia.geometry.transform import pyrup
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
from typing_extensions import Literal
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from pose_est_nets.losses.losses import (
    convert_dict_entries_to_tensors,
    convert_loss_tensors_to_torch_nn_modules,
    get_losses_dict,
    MaskedHeatmapLoss,
    MaskedRMSELoss,
)
from pose_est_nets.models.base_resnet import BaseFeatureExtractor
from pose_est_nets.models.regression_tracker import BaseBatchDict


class HeatmapBatchDict(BaseBatchDict):
    """Inherets key-value pairs from BaseExampleDict and adds "heatmaps"

    Args:
        BaseExampleDict (TypedDict): a dict containing a single example.
    """

    heatmaps: TensorType[
        "batch", "num_keypoints", "heatmap_height", "heatmap_width", float
    ]


class SemiSupervisedHeatmapBatchDict(TypedDict):
    labeled: HeatmapBatchDict
    unlabeled: TensorType[
        "sequence_length", "RGB":3, "image_height", "image_width", float
    ]


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
        supervised_loss: str = "mse",
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
        self.supervised_loss = supervised_loss
        self.confidence_scale = torch.tensor(confidence_scale, device=self.device)
        self.threshold = threshold
        self.softmax = nn.Softmax(dim=2)
        self.temperature = torch.tensor(100, device=self.device)  # soft argmax temp
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
    def training_step(self, batch_dict: HeatmapBatchDict, batch_idx: int) -> Dict:

        # forward pass
        predicted_heatmaps = self.forward(batch_dict["images"])  # images -> heatmaps
        predicted_keypoints, confidence = self.run_subpixelmaxima(
            predicted_heatmaps
        )  # heatmaps -> keypoints

        # compute loss
        heatmap_loss = MaskedHeatmapLoss(
            batch_dict["heatmaps"], predicted_heatmaps, self.supervised_loss
        )

        supervised_rmse = MaskedRMSELoss(batch_dict["keypoints"], predicted_keypoints)

        # log training loss + rmse
        self.log("train_loss", heatmap_loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)

        return {"loss": heatmap_loss}

    @typechecked
    def evaluate(
        self,
        batch_dict: HeatmapBatchDict,
        stage: Optional[Literal["val", "test"]] = None,
    ) -> None:
        predicted_heatmaps = self.forward(batch_dict["images"])  # images -> heatmaps
        predicted_keypoints, confidence = self.run_subpixelmaxima(
            predicted_heatmaps
        )  # heatmaps -> keypoints
        loss = MaskedHeatmapLoss(
            batch_dict["heatmaps"], predicted_heatmaps, self.supervised_loss
        )
        supervised_rmse = MaskedRMSELoss(batch_dict["keypoints"], predicted_keypoints)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
            self.log(
                f"{stage}_supervised_rmse", supervised_rmse, prog_bar=True, logger=True
            )

    def validation_step(self, validation_batch: HeatmapBatchDict, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: HeatmapBatchDict, batch_idx):
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
        supervised_loss: str = "mse",
        confidence_scale: float = 1.0,
        threshold: Optional[float] = None,
        torch_seed: int = 123,
        loss_params: Optional[
            dict
        ] = None,  # TODO: specify a dictionary of dictionaries. is it Optional?
        semi_super_losses_to_use: Optional[list] = None,
        learn_weights: bool = True,  # whether to use multitask weight learning
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
            supervised_loss=supervised_loss,
            confidence_scale=confidence_scale,
            threshold=threshold,
            torch_seed=torch_seed,
        )
        print(semi_super_losses_to_use)
        self.loss_function_dict = get_losses_dict(semi_super_losses_to_use)
        self.learn_weights = learn_weights
        (
            self.loss_params_tensor,
            self.loss_params_dict,
        ) = convert_dict_entries_to_tensors(
            loss_params=loss_params,
            device=self.device,
            losses_to_use=semi_super_losses_to_use,
            to_parameters=self.learn_weights,
        )
        if (
            self.learn_weights == True
        ):  # for each unsupervised loss we convert the "log_weight" in the config into a learnable parameter
            self.loss_params_tensor = convert_loss_tensors_to_torch_nn_modules(
                self.loss_params_tensor
            )
            print(self.loss_params_tensor)  # TODO: remove
        # self.save_hyperparameters()

    @typechecked
    def training_step(
        self, data_batch: SemiSupervisedHeatmapBatchDict, batch_idx: int
    ) -> Dict:

        # forward pass labeled
        predicted_heatmaps = self.forward(data_batch["labeled"]["images"])
        predicted_keypoints, confidence = self.run_subpixelmaxima(
            predicted_heatmaps
        )  # heatmaps -> keypoints

        # compute loss labeled
        supervised_loss = MaskedHeatmapLoss(
            data_batch["labeled"]["heatmaps"], predicted_heatmaps, self.supervised_loss
        )

        supervised_rmse = MaskedRMSELoss(
            data_batch["labeled"]["keypoints"], predicted_keypoints
        )

        # forward pass unlabeled
        unlabeled_predicted_heatmaps = self.forward(data_batch["unlabeled"])
        predicted_us_keypoints, confidence = self.run_subpixelmaxima(
            unlabeled_predicted_heatmaps
        )

        # loop over unsupervised losses
        tot_loss = 0.0
        tot_loss += supervised_loss
        for loss_name, loss_func in self.loss_function_dict.items():
            # Some losses use keypoint_preds, some use heatmap_preds, and some use both.
            # all have **kwargs so are robust to unneeded inputs.

            unsupervised_loss = loss_func(
                keypoint_preds=predicted_us_keypoints,
                heatmap_preds=unlabeled_predicted_heatmaps,
                **self.loss_params_tensor[loss_name],
                **self.loss_params_dict[loss_name],
            )

            loss_weight = (
                1.0
                / (  # weight = \sigma where our trainable parameter is \log(\sigma^2). i.e., we take the parameter as it is in the config and exponentiate it to enforce positivity
                    2.0 * torch.exp(self.loss_params_tensor[loss_name]["log_weight"])
                )
            )

            current_weighted_loss = loss_weight * unsupervised_loss
            tot_loss += current_weighted_loss

            if (
                self.learn_weights == True
            ):  # penalize for the magnitude of the weights: \log(\sigma_i) for each weight i
                # tot_loss += -0.5 * torch.log((2.0 * loss_weight))
                tot_loss += (
                    0.5 * self.loss_params_tensor[loss_name]["log_weight"]
                )  # recall that \log(\sigma_1 * \sigma_2 * ...) = \log(\sigma_1) + \log(\sigma_2) + ...
            # log individual unsupervised losses
            self.log(loss_name + "_loss", unsupervised_loss, prog_bar=True)
            self.log(
                "weighted_" + loss_name + "_loss", current_weighted_loss, prog_bar=True
            )

        # log other losses
        self.log("total_loss", tot_loss, prog_bar=True)
        self.log("supervised_loss", supervised_loss, prog_bar=True)
        self.log("supervised_rmse", supervised_rmse, prog_bar=True)

        # log weights of losses (we do it always, but it is interesting only when self.learn_weights=True)
        # for each unsupervised loss we convert the "log_weight" in the config into a learnable parameter
        # the quantity being logged is \sigma, where the weight is 1/ 2 *\sigma^2. Ideally, \sigma should decrease in training.
        for loss_name in self.loss_function_dict.keys():
            self.log(
                "{}_{}".format(loss_name, "weight"),
                loss_weight,
                prog_bar=True,
            )
        return {"loss": tot_loss}

    @staticmethod
    def anneal_unsupervised_weight(epoch: int, increase_factor: float) -> float:
        return max(epoch * increase_factor, 1.0)

    # single optimizer with different learning rates
    def configure_optimizers(self):
        optimizer = Adam(
            [
                {"params": self.backbone.parameters()},
                {"params": self.upsampling_layers.parameters()},
                {"params": self.loss_params_tensor.parameters(), "lr": 1e-1},
            ],
            lr=1e-3,
        )

        scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
