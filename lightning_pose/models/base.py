"""Base class for backbone that acts as a feature extractor."""

from typing import Dict, Literal, Optional, Union

import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.utils import (
    BaseLabeledBatchDict,
    HeatmapLabeledBatchDict,
    SemiSupervisedBatchDict,
    SemiSupervisedHeatmapBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.models import ALLOWED_BACKBONES

MULTISTEPLR_MILESTONES_DEFAULT = [100, 200, 300]
MULTISTEPLR_GAMMA_DEFAULT = 0.5


def get_context_from_sequence(
    img_seq: Union[
        TensorType["seq_len", "RGB":3, "image_height", "image_width"],
        TensorType["seq_len", "n_features", "rep_height", "rep_width"],
    ],
    context_length: int,
) -> Union[
        TensorType["seq_len", "context_length", "RGB": 3, "image_height", "image_width"],
        TensorType["seq_len", "context_length", "n_features", "rep_height", "rep_width"],
]:
    # our goal is to extract 5-frame sequences from this sequence
    img_shape = img_seq.shape[1:]  # e.g., (3, H, W)
    seq_len = img_seq.shape[0]  # how many images in batch
    train_seq = torch.zeros((seq_len, context_length, *img_shape), device=img_seq.device)
    # define pads: start pad repeats the zeroth image twice. end pad repeats the last image twice.
    # this is to give padding for the first and last frames of the sequence
    pad_start = torch.tile(img_seq[0].unsqueeze(0), (2, 1, 1, 1))
    pad_end = torch.tile(img_seq[-1].unsqueeze(0), (2, 1, 1, 1))
    # pad the sequence
    padded_seq = torch.cat((pad_start, img_seq, pad_end), dim=0)
    # padded_seq = torch.cat((two_pad, img_seq, two_pad), dim=0)
    for i in range(seq_len):
        # extract 5-frame sequences from the padded sequence
        train_seq[i] = padded_seq[i : i + context_length]
    return train_seq


class BaseFeatureExtractor(LightningModule):
    """Object that contains the base resnet feature extractor."""

    def __init__(
        self,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = False,
        image_size: int = 256,
        model_type: Literal["heatmap", "regression"] = "heatmap",
        **kwargs,
    ) -> None:
        """A CNN model that takes in images and generates features.

        ResNets will be loaded from torchvision and can be either pre-trained
        on ImageNet or randomly initialized. These were originally used for
        classification tasks, so we truncate their final fully connected layer.

        Args:
            backbone: which backbone version to use; defaults to resnet50
            pretrained: True to load weights pretrained on imagenet (torchvision models only)
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
            image_size: height/width of frames
            model_type: type of model

        """
        super().__init__()
        print("\n Initializing a {} instance.".format(self._get_name()))

        self.backbone_arch = backbone

        if "sam" in self.backbone_arch:
            raise NotImplementedError(
                "segment-anything backbones are not supported in this branch."
                "If you are running the code from a github installation, switch to the branch"
                "features/vit."
                "If you have pip installed lightning pose, there is no access to segment-anything"
                "models due to dependency/installation issues. "
                "For more information please contatct the package maintainers."
            )
        else:
            from lightning_pose.models.backbones.torchvision import build_backbone

        self.backbone, self.num_fc_input_features = build_backbone(
            backbone_arch=self.backbone_arch,
            pretrained=pretrained,
            model_type=model_type,  # for torchvision only
            image_size=image_size,  # for ViTs only
        )

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.do_context = do_context

    def get_representations(
        self,
        images: Union[
            TensorType["batch", "RGB":3, "image_height", "image_width"],
            TensorType["batch", "frames", "RGB":3, "image_height", "image_width"],
            TensorType["sequence_length", "RGB":3, "image_height", "image_width"],
        ],
    ) -> Union[
        TensorType["new_batch", "features", "rep_height", "rep_width"],
        TensorType["new_batch", "features", "rep_height", "rep_width", "frames"],
    ]:
        """Forward pass from images to feature maps.

        Wrapper around the backbone's feature_extractor() method for typechecking purposes.
        See tests/models/test_base.py for example shapes.

        Args:
            images: a batch of images
            do_context: whether or not to use extra frames of context

        Returns:
            a representation of the images; features differ as a function of resnet
            version. Representation height and width differ as a function of image
            dimensions, and are not necessarily equal.

        """
        if self.do_context:
            if len(images.shape) == 5:
                # non-consecutive sequences, used for batches of labeled data during training
                batch, frames, channels, image_height, image_width = images.shape
                frames_batch_shape = batch * frames
                images_batch_frames: TensorType[
                    "batch*frames", "channels":3, "image_height", "image_width"
                ] = images.reshape(frames_batch_shape, channels, image_height, image_width)
                outputs: TensorType[
                    "batch*frames", "features", "rep_height", "rep_width"
                ] = self.backbone(images_batch_frames)
                outputs: TensorType[
                    "batch", "frames", "features", "rep_height", "rep_width"
                ] = outputs.reshape(
                    images.shape[0],
                    images.shape[1],
                    outputs.shape[1],
                    outputs.shape[2],
                    outputs.shape[3],
                )
            elif len(images.shape) == 4:
                # we have a single sequence of frames from DALI (not a batch of sequences)
                # valid frame := a frame that has two frames before it and two frames after it
                # we push it as is through the backbone, and then use tiling to make it into
                # (sequence_length, features, rep_height, rep_width, num_context_frames)
                # for now we discard the padded frames (first and last two)
                # the output will be one representation per valid frame
                sequence_length, channels, image_height, image_width = images.shape
                representations: TensorType[
                    "sequence_length", "channels":3, "rep_height", "rep_width"
                ] = self.backbone(images)
                # we need to tile the representations to make it into
                # (num_valid_frames, features, rep_height, rep_width, num_context_frames)
                # TODO: context frames should be configurable
                tiled_representations = get_context_from_sequence(
                    img_seq=representations, context_length=5
                )
                # get rid of first and last two frames
                if tiled_representations.shape[0] < 5:
                    raise RuntimeError(
                        "Not enough valid frames to make a context representation."
                    )
                outputs = tiled_representations[2:-2, :, :, :, :]

            # for both types of batches, we reshape in the same way
            # context is in the last dimension for the linear layer.
            representations: TensorType[
                "batch", "features", "rep_height", "rep_width", "frames"
            ] = torch.permute(outputs, (0, 2, 3, 4, 1))
        else:
            representations = self.backbone(images)

        return representations

    def forward(
        self,
        images: Union[
            TensorType["batch", "RGB":3, "image_height", "image_width"],
            TensorType["batch", "seq_length", "RGB":3, "image_height", "image_width"],
            TensorType["seq_length", "RGB":3, "image_height", "image_width"],
        ],
    ) -> Union[
        TensorType["new_batch", "features", "rep_height", "rep_width"],
        TensorType["new_batch", "features", "rep_height", "rep_width", "frames"],
    ]:
        """Forward pass from images to representations.

        Wrapper around self.get_representations().
        Fancier childern models will use get_representations() in their forward methods.

        Args:
            images: a batch of images.

        Returns:
            a representation of the images.

        """
        return self.get_representations(images)

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_scheduler(self, optimizer):

        # define a scheduler that reduces the base learning rate
        if self.lr_scheduler == "multisteplr" or self.lr_scheduler == "multistep_lr":

            if self.lr_scheduler_params is None:
                milestones = MULTISTEPLR_MILESTONES_DEFAULT
                gamma = MULTISTEPLR_GAMMA_DEFAULT
            else:
                milestones = self.lr_scheduler_params.get(
                    "milestones", MULTISTEPLR_MILESTONES_DEFAULT)
                gamma = self.lr_scheduler_params.get("gamma", MULTISTEPLR_GAMMA_DEFAULT)

            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        else:
            raise NotImplementedError("'%s' is an invalid LR scheduler" % self.lr_scheduler)

        return scheduler

    def configure_optimizers(self) -> dict:
        """Select optimizer, lr scheduler, and metric for monitoring."""

        # get trainable params
        params = self.get_parameters()

        # init standard adam optimizer
        optimizer = Adam(params, lr=1e-3)

        # get learning rate scheduler
        scheduler = self.get_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_supervised_loss",
        }


class BaseSupervisedTracker(BaseFeatureExtractor):
    """Base class for supervised trackers."""

    def get_loss_inputs_labeled(
        self,
        batch_dict: Union[BaseLabeledBatchDict, HeatmapLabeledBatchDict],
    ) -> dict:
        """Return predicted coordinates for a batch of data."""
        raise NotImplementedError

    def evaluate_labeled(
        self,
        batch_dict: Union[BaseLabeledBatchDict, HeatmapLabeledBatchDict],
        stage: Optional[Literal["train", "val", "test"]] = None,
    ) -> TensorType[(), float]:
        """Compute and log the losses on a batch of labeled data."""

        # forward pass; collected true and predicted heatmaps, keypoints
        data_dict = self.get_loss_inputs_labeled(batch_dict=batch_dict)

        # compute and log loss on labeled data
        loss, log_list = self.loss_factory(stage=stage, **data_dict)

        # compute and log pixel_error loss on labeled data
        loss_rmse, _ = self.rmse_loss(stage=stage, **data_dict)

        if stage:
            # log overall supervised loss
            self.log(f"{stage}_supervised_loss", loss, prog_bar=True)
            # log supervised pixel_error
            self.log(f"{stage}_supervised_rmse", loss_rmse)
            # log individual supervised losses
            for log_dict in log_list:
                self.log(**log_dict)

        return loss

    def training_step(
        self,
        train_batch: Union[BaseLabeledBatchDict, HeatmapLabeledBatchDict],
        batch_idx: int,
    ) -> Dict[str, TensorType[(), float]]:
        """Base training step, a wrapper around the `evaluate_labeled` method."""
        loss = self.evaluate_labeled(train_batch, "train")
        return {"loss": loss}

    def validation_step(
        self,
        val_batch: Union[BaseLabeledBatchDict, HeatmapLabeledBatchDict],
        batch_idx: int,
    ) -> None:
        """Base validation step, a wrapper around the `evaluate_labeled` method."""
        self.evaluate_labeled(val_batch, "val")

    def test_step(
        self,
        test_batch: Union[BaseLabeledBatchDict, HeatmapLabeledBatchDict],
        batch_idx: int,
    ) -> None:
        """Base test step, a wrapper around the `evaluate_labeled` method."""
        self.evaluate_labeled(test_batch, "test")

    def get_parameters(self):

        if getattr(self, "upsampling_layers", None) is not None:
            # single optimizer with single learning rate
            params = [
                # don't uncomment line below;
                # the BackboneFinetuning callback should add backbone to the params
                # {"params": self.backbone.parameters()},
                # important this is the 0th element, for BackboneFinetuning callback
                {"params": self.upsampling_layers.parameters()},
                # {"params": self.unnormalized_weights},
            ]
        else:
            # standard adam optimizer
            params = filter(lambda p: p.requires_grad, self.parameters())

        return params


@typechecked
class SemiSupervisedTrackerMixin(object):
    """Mixin class providing training step function for semi-supervised models."""

    def get_loss_inputs_unlabeled(self, batch: UnlabeledBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        raise NotImplementedError

    def evaluate_unlabeled(
        self,
        batch: UnlabeledBatchDict,
        stage: Optional[Literal["train", "val", "test"]] = None,
        anneal_weight: Union[float, torch.Tensor] = 1.0,
    ) -> TensorType[(), float]:
        """Compute and log the losses on a batch of unlabeled data (frames only)."""

        # forward pass: collect predicted heatmaps and keypoints
        data_dict = self.get_loss_inputs_unlabeled(batch=batch)

        # compute loss on unlabeled data
        loss, log_list = self.loss_factory_unsup(
            stage=stage,
            anneal_weight=anneal_weight,
            **data_dict,
        )

        if stage:
            # log individual unsupervised losses
            for log_dict in log_list:
                self.log(**log_dict)

        return loss

    def training_step(
        self,
        train_batch: Union[SemiSupervisedBatchDict, SemiSupervisedHeatmapBatchDict],
        batch_idx: int,
    ) -> Dict[str, TensorType[(), float]]:
        """Training step computes and logs both supervised and unsupervised losses."""

        # on each epoch, self.total_unsupervised_importance is modified by the
        # AnnealWeight callback
        self.log(
            "total_unsupervised_importance",
            self.total_unsupervised_importance,
            prog_bar=True,
        )

        # computes and logs supervised losses
        # train_batch["labeled"] contains:
        # - images
        # - keypoints
        # - heatmaps
        loss_super = self.evaluate_labeled(
            batch_dict=train_batch["labeled"],
            stage="train",
        )

        # computes and logs unsupervised losses
        # train_batch["unlabeled"] contains:
        # - images
        loss_unsuper = self.evaluate_unlabeled(
            batch=train_batch["unlabeled"],
            stage="train",
            anneal_weight=self.total_unsupervised_importance,
        )

        # log total loss
        total_loss = loss_super + loss_unsuper
        self.log("total_loss", total_loss, prog_bar=True)

        return {"loss": total_loss}

    def get_parameters(self):

        if getattr(self, "upsampling_layers", None) is not None:
            # if we're here this is a heatmap model
            params = [
                # don't uncomment line below;
                # the BackboneFinetuning callback should add backbone to the params
                # {"params": self.backbone.parameters()},
                # important this is the 0th element, for BackboneFinetuning callback
                {"params": self.upsampling_layers.parameters()},
                {"params": self.unnormalized_weights},
            ]

        else:
            # standard adam optimizer for regression model
            params = filter(lambda p: p.requires_grad, self.parameters())

        return params

    # # single optimizer with different learning rates
    # def configure_optimizers(self):
    #     params_net = [
    #         # {"params": self.backbone.parameters()},
    #         #  don't uncomment above line; the BackboneFinetuning callback should add
    #         # backbone to the params.
    #         {
    #             "params": self.upsampling_layers.parameters()
    #         },  # important that this is the 0th element, for BackboneFineTuning
    #     ]
    #     optimizer = Adam(params_net, lr=1e-3)
    #     scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.5)
    #
    #     optimizers = [optimizer]
    #     lr_schedulers = [scheduler]
    #
    #     return optimizers, lr_schedulers
