"""Models that produce heatmaps of keypoints from images."""

from typing import Dict, Optional, Tuple, Union

import torch
from kornia.geometry.subpix import spatial_softmax2d
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked
from typing_extensions import Literal

from lightning_pose.data.utils import (
    HeatmapLabeledBatchDict,
    UnlabeledBatchDict,
    undo_affine_transform,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.models import ALLOWED_BACKBONES
from lightning_pose.models.base import SemiSupervisedTrackerMixin
from lightning_pose.models.heatmap_tracker import HeatmapTracker


class HeatmapTrackerMHCRNN(HeatmapTracker):
    """Multi-headed Convolutional RNN network that handles context frames."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: Optional[LossFactory] = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        output_shape: Optional[tuple] = None,  # change
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        **kwargs,
    ):
        """Initialize a DLC-like model with resnet backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
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

        if downsample_factor != 2:
            raise NotImplementedError("MHCRNN currently only implements downsample_factor=2")

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        if "do_context" in kwargs.keys():
            del kwargs["do_context"]
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
            do_context=True,
            **kwargs,
        )

        # create upsampling layers for crnn
        self.crnn = UpsamplingCRNN(
            num_filters_for_upsampling=self.num_filters_for_upsampling,
            num_keypoints=self.num_keypoints,
            upsampling_factor=1 if "vit" in backbone else 2,
        )
        self.upsampling_layers_rnn = self.crnn.layers

        # alias parent upsampling layers for single frame
        self.upsampling_layers_sf = self.upsampling_layers

    def heatmaps_from_representations(
        self,
        representations: TensorType["batch", "features", "rep_height", "rep_width", "frames"],
    ) -> Tuple[
            TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
            TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Handle context frames then upsample to get final heatmaps."""
        # permute to shape (frames, batch, features, rep_height, rep_width)
        representations = torch.permute(representations, (4, 0, 1, 2, 3))
        heatmaps_crnn = self.crnn(representations)
        heatmaps_sf = self.upsampling_layers_sf(representations[2])  # index 2 == middle frame

        return heatmaps_crnn, heatmaps_sf

    def forward(
        self,
        images: Union[
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"]
        ],
    ) -> Tuple[
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
            TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"],
    ]:
        """Forward pass through the network."""

        # we get one representation for each desired output.
        representations = self.get_representations(images)
        heatmaps_crnn, heatmaps_sf = self.heatmaps_from_representations(representations)

        # normalize heatmaps
        # softmax temp stays 1 here; to modify for model predictions, see constructor
        heatmaps_crnn_norm = spatial_softmax2d(heatmaps_crnn, temperature=torch.tensor([1.0]))
        heatmaps_sf_norm = spatial_softmax2d(heatmaps_sf, temperature=torch.tensor([1.0]))

        return heatmaps_crnn_norm, heatmaps_sf_norm

    def get_loss_inputs_labeled(self, batch_dict: HeatmapLabeledBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_crnn, pred_heatmaps_sf = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        pred_keypoints_crnn, confidence_crnn = self.run_subpixelmaxima(pred_heatmaps_crnn)
        pred_keypoints_sf, confidence_sf = self.run_subpixelmaxima(pred_heatmaps_sf)
        return {
            "heatmaps_targ": torch.cat([batch_dict["heatmaps"], batch_dict["heatmaps"]], dim=0),
            "heatmaps_pred": torch.cat([pred_heatmaps_crnn, pred_heatmaps_sf], dim=0),
            "keypoints_targ": torch.cat([batch_dict["keypoints"], batch_dict["keypoints"]], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_crnn, pred_keypoints_sf], dim=0),
            "confidences": torch.cat([confidence_crnn, confidence_sf], dim=0),
        }

    def predict_step(
        self,
        batch: Union[HeatmapLabeledBatchDict, UnlabeledBatchDict],
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
        if "images" in batch.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch["frames"]

        # images -> heatmaps
        pred_heatmaps_crnn, pred_heatmaps_sf = self.forward(images)
        # heatmaps -> keypoints
        pred_keypoints_crnn, confidence_crnn = self.run_subpixelmaxima(pred_heatmaps_crnn)
        pred_keypoints_sf, confidence_sf = self.run_subpixelmaxima(pred_heatmaps_sf)
        # reshape keypoints to be (batch, n_keypoints, 2)
        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1, 2)
        pred_keypoints_crnn = pred_keypoints_crnn.reshape(pred_keypoints_crnn.shape[0], -1, 2)
        # find higher confidence indices
        crnn_conf_gt = torch.gt(confidence_crnn, confidence_sf)
        # select higher confidence indices
        pred_keypoints_sf[crnn_conf_gt] = pred_keypoints_crnn[crnn_conf_gt]
        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1)
        confidence_sf[crnn_conf_gt] = confidence_crnn[crnn_conf_gt]

        if return_heatmaps:
            pred_heatmaps_sf[crnn_conf_gt] = pred_heatmaps_crnn[crnn_conf_gt]
            return pred_keypoints_sf, confidence_sf, pred_heatmaps_sf
        else:
            return pred_keypoints_sf, confidence_sf

    def get_parameters(self):
        params = [
            # don't uncomment line below
            # the BackboneFinetuning callback should add backbone to the params.
            # {"params": self.backbone.parameters()},
            # important this is the 0th element, for BackboneFinetuning callback
            {"params": self.upsampling_layers_rnn.parameters()},
            {"params": self.upsampling_layers_sf.parameters()},
        ]
        return params


@typechecked
class SemiSupervisedHeatmapTrackerMHCRNN(SemiSupervisedTrackerMixin, HeatmapTrackerMHCRNN):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: Optional[LossFactory] = None,
        loss_factory_unsupervised: Optional[LossFactory] = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        output_shape: Optional[tuple] = None,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        **kwargs,
    ):
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
        if loss_factory_unsupervised:
            self.loss_factory_unsup = loss_factory_unsupervised.to(self.device)
        else:
            self.loss_factory_unsup = None

        # this attribute will be modified by AnnealWeight callback during training
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))
        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(self, batch: UnlabeledBatchDict) -> Dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps_crnn, pred_heatmaps_sf = self.forward(batch["frames"])
        # heatmaps -> keypoints
        pred_keypoints_crnn, confidence_crnn = self.run_subpixelmaxima(pred_heatmaps_crnn)
        pred_keypoints_sf, confidence_sf = self.run_subpixelmaxima(pred_heatmaps_sf)

        # undo augmentation if needed
        if batch["transforms"].shape[-1] == 3:
            # reshape to (seq_len, n_keypoints, 2)
            pred_kps = torch.reshape(pred_keypoints_crnn, (pred_keypoints_crnn.shape[0], -1, 2))
            # undo
            pred_kps = undo_affine_transform(pred_kps, batch["transforms"])
            # reshape to (seq_len, n_keypoints * 2)
            pred_keypoints_crnn = torch.reshape(pred_kps, (pred_kps.shape[0], -1))

            # reshape to (seq_len, n_keypoints, 2)
            pred_kps = torch.reshape(pred_keypoints_sf, (pred_keypoints_sf.shape[0], -1, 2))
            # undo
            pred_kps = undo_affine_transform(pred_kps, batch["transforms"])
            # reshape to (seq_len, n_keypoints * 2)
            pred_keypoints_sf = torch.reshape(pred_kps, (pred_kps.shape[0], -1))

        return {
            "heatmaps_pred": torch.cat([pred_heatmaps_crnn, pred_heatmaps_sf], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_crnn, pred_keypoints_sf], dim=0),
            "confidences": torch.cat([confidence_crnn, confidence_sf], dim=0),
        }

    def get_parameters(self):
        params = [
            # don't uncomment line below
            # the BackboneFinetuning callback should add backbone to the params.
            # {"params": self.backbone.parameters()},
            # important this is the 0th element, for BackboneFinetuning callback
            {"params": self.upsampling_layers_rnn.parameters()},
            {"params": self.upsampling_layers_sf.parameters()},
        ]
        return params


class UpsamplingCRNN(torch.nn.Module):
    """Bidirectional Convolutional RNN network that handles heatmaps of context frames.

    The input to the CRNN is a set of heatmaps at times t-k, ..., t, ...t+k, one heatmap for each
    timepoint/keypoint

    The output of the CRNN is a single heatmap for each keypoint

    """

    def __init__(
        self,
        num_filters_for_upsampling: int,
        num_keypoints: int,
        upsampling_factor: Literal[1, 2] = 2,
        hkernel: int = 2,
        hstride: int = 2,
        hpad: int = 0,
        nfilters_channel: int = 16,
    ) -> None:
        """Upsampling Convolutional RNN - initialize input and hidden weights."""

        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.pixel_shuffle = nn.PixelShuffle(2)
        if self.upsampling_factor == 2:
            self.W_pre = HeatmapTracker.create_double_upsampling_layer(
                in_channels=num_filters_for_upsampling // 4,
                out_channels=num_keypoints,
            )
            in_channels_rnn = num_keypoints
        else:
            in_channels_rnn = num_filters_for_upsampling // 4

        self.W_f = HeatmapTracker.create_double_upsampling_layer(
            in_channels=in_channels_rnn,
            out_channels=num_keypoints,
        )
        H_f_layers = []
        H_f_layers.append(
            nn.Conv2d(
                in_channels=num_keypoints,
                out_channels=num_keypoints * nfilters_channel,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        H_f_layers.append(
            nn.ConvTranspose2d(
                in_channels=num_keypoints * nfilters_channel,
                out_channels=num_keypoints,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                output_padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        self.H_f = nn.Sequential(*H_f_layers)

        self.W_b = HeatmapTracker.create_double_upsampling_layer(
            in_channels=in_channels_rnn,
            out_channels=num_keypoints,
        )
        H_b_layers = []
        H_b_layers.append(
            nn.Conv2d(
                in_channels=num_keypoints,
                out_channels=num_keypoints * nfilters_channel,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        H_b_layers.append(
            nn.ConvTranspose2d(
                in_channels=num_keypoints * nfilters_channel,
                out_channels=num_keypoints,
                kernel_size=(hkernel, hkernel),
                stride=(hstride, hstride),
                padding=(hpad, hpad),
                output_padding=(hpad, hpad),
                groups=num_keypoints,
            )
        )
        self.H_b = nn.Sequential(*H_b_layers)
        self.initialize_layers()
        if self.upsampling_factor == 2:
            self.layers = torch.nn.ModuleList([self.W_pre, self.W_f, self.H_f, self.W_b, self.H_b])
        else:
            self.layers = torch.nn.ModuleList([self.W_f, self.H_f, self.W_b, self.H_b])

    def initialize_layers(self):
        if self.upsampling_factor == 2:
            torch.nn.init.xavier_uniform_(self.W_pre.weight, gain=1.0)
            torch.nn.init.zeros_(self.W_pre.bias)

        torch.nn.init.xavier_uniform_(self.W_f.weight, gain=1.0)
        torch.nn.init.zeros_(self.W_f.bias)
        for index, layer in enumerate(self.H_f):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.xavier_uniform_(self.W_b.weight, gain=1.0)
        torch.nn.init.zeros_(self.W_b.bias)
        for index, layer in enumerate(self.H_b):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
            torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        representations: TensorType["frames", "batch", "features", "rep_height", "rep_width"]
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:

        # expand representations in spatial domain using pixel shuffle to create heatmaps
        if self.upsampling_factor == 2:
            # upsample once more before passing through RNN
            # need to reshape to push through convolution layers
            frames, batch, features, rep_height, rep_width = representations.shape
            frames_batch_shape = batch * frames
            representations_batch_frames: TensorType[
                "batch*frames", "features", "rep_height", "rep_width"
            ] = representations.reshape(frames_batch_shape, features, rep_height, rep_width)
            x_tensor = self.W_pre(self.pixel_shuffle(representations_batch_frames))
            x_tensor = x_tensor.reshape(
                frames,
                batch,
                x_tensor.shape[1],
                x_tensor.shape[2],
                x_tensor.shape[3],
            )
        else:
            x_tensor = self.pixel_shuffle(representations)

        # push heatmaps through CRNN
        x_f = self.W_f(x_tensor[0])
        for frame_batch in x_tensor[1:]:  # forward pass
            x_f = self.W_f(frame_batch) + self.H_f(x_f)
        x_tensor_b = torch.flip(x_tensor, dims=[0])
        x_b = self.W_b(x_tensor_b[0])
        for frame_batch in x_tensor_b[1:]:  # backwards pass
            x_b = self.W_b(frame_batch) + self.H_b(x_b)

        # average forward/backward heatmaps
        heatmaps = (x_f + x_b) / 2

        return heatmaps
