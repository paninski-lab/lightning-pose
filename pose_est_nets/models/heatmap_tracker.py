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
from pose_est_nets.models.base_resnet import BaseFeatureExtractor
from pose_est_nets.models.regression_tracker import BaseBatchDict


class HeatmapBatchDict(BaseBatchDict):
    """Inherets key-value pairs from BaseExampleDict and adds "heatmaps"."""
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
    def training_step(self, batch_dict: HeatmapBatchDict, batch_idx: int) -> Dict:

        # TODO: return loss from evaluate method and set "train" stage?
        # forward pass
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)

        # compute and log loss
        loss = self.loss_factory(
            heatmaps_targ=data_batch["heatmaps"],
            heatmaps_pred=predicted_heatmaps,
            stage="train",  # for logging purposes
        )
        self.log("train_loss", loss, prog_bar=True)

        # for additional info: compute and log supervised rmse
        rmse_loss = RegressionRMSELoss()
        supervised_rmse = rmse_loss(
            keypoints_targ=data_batch["keypoints"],
            keypoints_pred=predicted_keypoints,
            logging=False,
        )
        self.log("train_rmse_supervised", supervised_rmse, prog_bar=True)

        return {"loss": loss}

    @typechecked
    def evaluate(
        self,
        batch_dict: HeatmapBatchDict,
        stage: Optional[Literal["val", "test"]] = None,
    ) -> None:

        # forward pass
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)

        # compute loss
        loss = self.loss_factory(
            heatmaps_targ=data_batch["heatmaps"],
            heatmaps_pred=predicted_heatmaps,
            stage=stage,  # for logging purposes
        )

        if stage:
            rmse_loss = RegressionRMSELoss()
            supervised_rmse = rmse_loss(
                keypoints_targ=data_batch["keypoints"],
                keypoints_pred=predicted_keypoints,
                logging=False,
            )
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_rmse_supervised", supervised_rmse, prog_bar=True)

    def validation_step(self, validation_batch: HeatmapBatchDict, batch_idx):
        self.evaluate(validation_batch, "val")

    def test_step(self, test_batch: HeatmapBatchDict, batch_idx):
        self.evaluate(test_batch, "test")

    # single optimizer with different learning rates
    def configure_optimizers(self):
        params = [
            # {"params": self.backbone.parameters()},
            #  don't uncomment above line; the BackboneFinetuning callback should add
            # backbone to the params.
            {
                "params": self.upsampling_layers.parameters()
            },  # important this is the 0th element, for BackboneFinetuning callback
        ]
        optimizer = Adam(params, lr=1e-3)
        scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
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

    @typechecked
    def training_step(
        self, data_batch: SemiSupervisedHeatmapBatchDict, batch_idx: int
    ) -> Dict:

        # on each epoch, self.total_unsupervised_importance is modified by the
        # AnnealWeight callback
        self.log(
            "total_unsupervised_importance",
            self.total_unsupervised_importance,
            prog_bar=True,
        )

        # forward pass labeled
        # --------------------
        predicted_heatmaps = self.forward(data_batch["labeled"]["images"])
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)

        # compute and log loss
        loss_super = self.loss_factory(
            heatmaps_targ=data_batch["labeled"]["heatmaps"],
            heatmaps_pred=predicted_heatmaps,
            stage="train",  # for logging purposes
        )
        self.log("train_loss_supervised", loss_super, prog_bar=True)

        # for additional info: compute and log supervised rmse
        rmse_loss = RegressionRMSELoss()
        supervised_rmse = rmse_loss(
            keypoints_targ=data_batch["labeled"]["keypoints"],
            keypoints_pred=predicted_keypoints,
            logging=False,
        )
        self.log("train_rmse_supervised", supervised_rmse, prog_bar=True)

        # forward pass unlabeled
        # ----------------------
        predicted_heatmaps_ul = self.forward(data_batch["unlabeled"])
        predicted_keypoints_ul, confidence = self.run_subpixelmaxima(
            predicted_heatmaps_ul
        )

        # compute and log unsupervised loss
        loss_unsuper = self.loss_factory_unsup(
            keypoints_pred=predicted_keypoints_ul,
            heatmaps_pred=predicted_heatmaps_ul,
            anneal_weight=self.total_unsupervised_importance,
            stage="train",  # for logging purposes
        )

        # log total loss
        total_loss = loss_super + loss_unsuper
        self.log("total_loss", total_loss, prog_bar=True)

        return {"loss": tot_loss}

    # single optimizer with different learning rates
    def configure_optimizers(self):
        params = [
            # {"params": self.backbone.parameters()}, # don't uncomment this line; the BackboneFinetuning callback should add backbone to the params.
            {
                "params": self.upsampling_layers.parameters()
            },  # important that this is the 0th element, for BackboneFineTuning
        ]
        if self.learn_weights:
            params.append({"params": self.loss_weights_dict.parameters(), "lr": 1e-2})
        optimizer = Adam(params, lr=1e-3)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200, 300], gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # # single optimizer with different learning rates
    # def configure_optimizers(self):
    #     params_net = [
    #         # {"params": self.backbone.parameters()}, # don't uncomment this line; the BackboneFinetuning callback should add backbone to the params.
    #         {
    #             "params": self.upsampling_layers.parameters()
    #         },  # important that this is the 0th element, for BackboneFineTuning
    #     ]
    #     optimizer = Adam(params_net, lr=1e-3)
    #     scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.5)

    #     optimizers = [optimizer]
    #     lr_schedulers = [scheduler]

    #     if self.learn_weights:
    #         params_weights = [{"params": self.loss_weights_dict.parameters()}]
    #         optimizer_weights = Adam(params_weights, lr=1e-3)
    #         optimizers.append(optimizer_weights)
    #         scheduler_weights = MultiStepLR(
    #             optimizer, milestones=[100, 200, 300], gamma=0.5
    #         )
    #         lr_schedulers.append(scheduler_weights)

    #     return optimizers, lr_schedulers
