"""Base class for backbone that acts as a feature extractor."""

from typing import Any, Literal, Union

import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.datatypes import (
    BaseLabeledBatchDict,
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    SemiSupervisedBatchDict,
    SemiSupervisedHeatmapBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.models.backbones import ALLOWED_BACKBONES

# to ignore imports for sphix-autoapidoc
__all__ = [
    "get_context_from_sequence",
    "BaseFeatureExtractor",
    "BaseSupervisedTracker",
    "SemiSupervisedTrackerMixin",
]

DEFAULT_LR_SCHEDULER_PARAMS = OmegaConf.create(
    {
        "milestones": [150, 200, 250],
        "gamma": 0.5,
    }
)

DEFAULT_OPTIMIZER_PARAMS = OmegaConf.create(
    {
        "learning_rate": 1e-3,
    }
)


class LrNotImplementedError(NotImplementedError):
    def __init__(self, lr_scheduler: str):
        super(LrNotImplementedError, self).__init__(
            "'%s' is an invalid LR scheduler. Must be multisteplr." % lr_scheduler
        )
        self.lr_scheduler = lr_scheduler


class OptimizerNotImplementedError(NotImplementedError):
    def __init__(self, optimizer: str):
        super(LrNotImplementedError, self).__init__(
            "'%s' is an invalid optimizer. Must be Adam or AdamW." % optimizer
        )
        self.optimizer = optimizer


def _apply_defaults_for_lr_scheduler_params(
    lr_scheduler: str, lr_scheduler_params: DictConfig | dict | None
) -> DictConfig:
    if lr_scheduler not in ("multistep_lr", "multisteplr"):
        raise LrNotImplementedError(lr_scheduler)

    if lr_scheduler_params is None:
        lr_scheduler_params = DEFAULT_LR_SCHEDULER_PARAMS
    else:
        lr_scheduler_params = OmegaConf.merge(
            DEFAULT_LR_SCHEDULER_PARAMS, lr_scheduler_params
        )

    return lr_scheduler_params


def _apply_defaults_for_optimizer_params(
    optimizer: str, optimizer_params: DictConfig | dict | None
) -> DictConfig:
    if optimizer not in ("Adam", "AdamW"):
        raise OptimizerNotImplementedError(optimizer)

    if optimizer_params is None:
        optimizer_params = DEFAULT_OPTIMIZER_PARAMS
    else:
        optimizer_params = OmegaConf.merge(DEFAULT_OPTIMIZER_PARAMS, optimizer_params)

    return optimizer_params


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
        lr_scheduler_params: DictConfig | dict | None = None,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | dict | None = None,
        do_context: bool = False,
        image_size: int = 256,
        model_type: Literal["heatmap", "regression"] = "heatmap",
        **kwargs: Any,
    ) -> None:
        """A CNN model that takes in images and generates features.

        ResNets will be loaded from torchvision and can be either pre-trained
        on ImageNet or randomly initialized. These were originally used for
        classification tasks, so we truncate their final fully connected layer.

        Args:
            backbone: which backbone version to use; defaults to resnet50
            pretrained: True to load weights pretrained on imagenet (torchvision models only)
            optimizer: optimizer class to instantiate (Adam, AdamW, more to be added in future)
            optimizer_params: arguments to pass to optimizer
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
            do_context: include temporal context when processing each frame
            image_size: height/width of frames, for ViT models only
            model_type: type of model

        """
        super().__init__()
        if self.local_rank == 0:
            print(f"\nInitializing a {self._get_name()} instance with {backbone} backbone.")

        self.backbone_arch = backbone

        if self.backbone_arch.startswith("vit"):
            from lightning_pose.models.backbones.vits import build_backbone
        else:
            from lightning_pose.models.backbones.torchvision import build_backbone

        self.backbone, self.num_fc_input_features = build_backbone(
            backbone_arch=self.backbone_arch,
            pretrained=pretrained,
            model_type=model_type,  # for torchvision only
            image_size=image_size,  # for ViTs only
            backbone_checkpoint=kwargs.get('backbone_checkpoint'),  # for ViTMAE's only
        )

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = _apply_defaults_for_lr_scheduler_params(
            lr_scheduler, lr_scheduler_params
        )
        self.optimizer = optimizer
        self.optimizer_params = _apply_defaults_for_optimizer_params(
            optimizer, optimizer_params
        )
        self.do_context = do_context

    def get_representations(
        self,
        images: (
            TensorType["batch", "channels":3, "image_height", "image_width"]
            | TensorType["batch", "frames", "channels":3, "image_height", "image_width"]
            | TensorType["seq_len", "channels":3, "image_height", "image_width"]
            | TensorType[
                "batch", "views", "frames", "channels":3, "image_height", "image_width"
            ]
            | TensorType[
                "seq_len", "view", "frames", "channels":3, "image_height", "image_width"
            ]
        ),
        is_multiview: bool = False,
    ) -> (
        TensorType["new_batch", "features", "rep_height", "rep_width"]
        | TensorType["new_batch", "features", "rep_height", "rep_width", "frames"]
    ):
        """Forward pass from images to feature maps.

        Wrapper around the backbone's feature_extractor() method for typechecking purposes.
        See tests/models/test_base.py for example shapes.

        Batch options
        -------------
        - TensorType["batch", "channels":3, "image_height", "image_width"]
          single view, labeled batch

        - TensorType["batch", "frames", "channels":3, "image_height", "image_width"]
          single view, labeled context batch

        - TensorType["seq_len", "channels":3, "image_height", "image_width"]
          single view, unlabeled batch from DALI

        - TensorType["batch", "views", "frames", "channels":3, "image_height", "image_width"]
          multivew, labeled context batch

        - TensorType["seq_len", "views", "channels":3, "image_height", "image_width"]
          multiview, unlabeled batch from DALI

        Args:
            images: a batch of images
            is_multiview: flag to distinguish batches of the same size

        Returns:
            a representation of the images; features differ as a function of resnet version.
            Representation height and width differ as a function of image dimensions, and are not
            necessarily equal.

        """
        if self.do_context:
            if (len(images.shape) == 5 and not is_multiview) or len(images.shape) == 6:
                # len = 5
                # incoming batch: singleview labeled batch
                # incoming shape: (batch, frames, channels, height, width)
                #
                # len = 6
                # incoming batch: multiview labeled batch
                # incoming shape: (batch, num_views, frames, channels, height, width)

                if len(images.shape) == 6:
                    # stacking all the views in batch dimension
                    shape = images.shape
                    images = images.reshape(-1, shape[-4], shape[-3], shape[-2], shape[-1])

                batch, frames, channels, image_height, image_width = images.shape
                frames_batch_shape = batch * frames
                images_batch_frames = images.reshape(
                    frames_batch_shape, channels, image_height, image_width,
                )
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
            elif len(images.shape) == 5 and is_multiview:
                # incoming batch: multiview unlabeled batch
                # incoming shape: (seq, num_views, channels, height, width)

                batch, num_views, channels, image_height, image_width = images.shape
                batch_views_shape = batch * num_views
                images_batch_views = images.reshape(
                    batch_views_shape, channels, image_height, image_width,
                )
                outputs: TensorType[
                    "batch*views", "features", "rep_height", "rep_width"
                ] = self.backbone(images_batch_views)
                outputs: TensorType[
                    "batch", "views", "features", "rep_height", "rep_width"
                ] = outputs.reshape(
                    batch,
                    num_views,
                    outputs.shape[1],
                    outputs.shape[2],
                    outputs.shape[3],
                )
                # stack views across feature dimension
                outputs: TensorType[
                    "batch", "views * features", "rep_height", "rep_width"
                ] = outputs.reshape(batch, -1, outputs.shape[-2], outputs.shape[-1])

                # we need to tile the representations to make it into
                # (num_valid_frames, features, rep_height, rep_width, num_context_frames)
                tiled_representations = get_context_from_sequence(
                    img_seq=outputs, context_length=5,
                )
                # get rid of first and last two frames
                if tiled_representations.shape[0] < 5:
                    raise RuntimeError("Not enough valid frames to make a context representation.")
                outputs = tiled_representations[2:-2, :, :, :, :]

            elif len(images.shape) == 4:
                # we have a single sequence of frames from DALI (not a batch of sequences)
                # valid frame := a frame that has two frames before it and two frames after it
                # we push it as is through the backbone, and then use tiling to make it into
                # (sequence_length, features, rep_height, rep_width, num_context_frames)
                # for now we discard the padded frames (first and last two)
                # the output will be one representation per valid frame
                sequence_length, channels, image_height, image_width = images.shape
                representations: TensorType[
                    "sequence_length", "features", "rep_height", "rep_width"
                ] = self.backbone(images)
                # we need to tile the representations to make it into
                # (num_valid_frames, features, rep_height, rep_width, num_context_frames)
                # TODO: context frames should be configurable
                tiled_representations = get_context_from_sequence(
                    img_seq=representations, context_length=5,
                )
                # get rid of first and last two frames
                if tiled_representations.shape[0] < 5:
                    raise RuntimeError("Not enough valid frames to make a context representation.")
                outputs = tiled_representations[2:-2, :, :, :, :]

            # for both types of batches, we reshape in the same way
            # context is in the last dimension for the linear layer.
            representations: TensorType[
                "batch", "features", "rep_height", "rep_width", "frames"
            ] = torch.permute(outputs, (0, 2, 3, 4, 1))
        else:
            # incoming batch: singleview labeled/unlabeled, multiview labeled/unlabeled reshaped
            # incoming shape: (batch, channels, height, width)
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

    def get_scheduler(self, optimizer):
        if self.lr_scheduler not in ("multistep_lr", "multisteplr"):
            raise LrNotImplementedError(self.lr_scheduler)
        # define a scheduler that reduces the base learning rate
        milestones = self.lr_scheduler_params.milestones
        gamma = self.lr_scheduler_params.gamma

        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        return scheduler

    def get_parameters(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params

    def configure_optimizers(self) -> dict:
        """Select optimizer, lr scheduler, and metric for monitoring."""

        # get trainable params
        params = self.get_parameters()

        # init standard adam optimizer
        if self.optimizer == "Adam":
            optimizer = optim.Adam(params, lr=self.optimizer_params.learning_rate)
        elif self.optimizer == "AdamW":
            optimizer = optim.AdamW(params, lr=self.optimizer_params.learning_rate)
        else:
            raise OptimizerNotImplementedError(self.optimizer)

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
        batch_dict: Union[
            BaseLabeledBatchDict,
            HeatmapLabeledBatchDict,
            MultiviewLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
    ) -> dict:
        """Return predicted coordinates for a batch of data."""
        raise NotImplementedError

    def evaluate_labeled(
        self,
        batch_dict: Union[
            BaseLabeledBatchDict,
            HeatmapLabeledBatchDict,
            MultiviewLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
        stage: Literal["train", "val", "test"] | None = None,
        anneal_weight: torch.Tensor | None = None,
    ) -> TensorType[(), float]:
        """Compute and log the losses on a batch of labeled data."""

        # forward pass; collected true and predicted heatmaps, keypoints
        data_dict = self.get_loss_inputs_labeled(batch_dict=batch_dict)

        # compute and log loss on labeled data
        loss, log_list = self.loss_factory(stage=stage, anneal_weight=anneal_weight, **data_dict)

        # compute and log pixel_error loss on labeled data
        loss_rmse, _ = self.rmse_loss(stage=stage, **data_dict)

        if stage:
            # logging with sync_dist=True will average the metric across GPUs in
            # multi-GPU training. Performance overhead was found negligible.

            # log overall supervised loss
            self.log(f"{stage}_supervised_loss", loss, prog_bar=True, sync_dist=True)
            # log supervised pixel_error
            self.log(f"{stage}_supervised_rmse", loss_rmse, sync_dist=True)
            # log individual supervised losses
            for log_dict in log_list:
                self.log(
                    log_dict['name'],
                    log_dict['value'].to(self.device),
                    prog_bar=log_dict.get('prog_bar', False),
                    sync_dist=True)

        return loss

    def training_step(
        self,
        batch_dict: Union[
            BaseLabeledBatchDict,
            HeatmapLabeledBatchDict,
            MultiviewLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
        batch_idx: int,
    ) -> dict[str, TensorType[(), float]]:
        """Base training step, a wrapper around the `evaluate_labeled` method."""
        # on each epoch, self.total_unsupervised_importance is modified by the
        # AnnealWeight callback
        if hasattr(self, "total_unsupervised_importance"):
            self.log(
                "total_unsupervised_importance",
                self.total_unsupervised_importance,
                prog_bar=True,
                # don't need to sync_dist because this is always the same across processes.
            )
            anneal_weight = self.total_unsupervised_importance
        else:
            anneal_weight = None
        loss = self.evaluate_labeled(batch_dict, "train", anneal_weight=anneal_weight)
        return {"loss": loss}

    def validation_step(
        self,
        batch_dict: Union[
            BaseLabeledBatchDict,
            HeatmapLabeledBatchDict,
            MultiviewLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
        batch_idx: int,
    ) -> None:
        """Base validation step, a wrapper around the `evaluate_labeled` method."""
        self.evaluate_labeled(batch_dict, "val")

    def test_step(
        self,
        batch_dict: Union[
            BaseLabeledBatchDict,
            HeatmapLabeledBatchDict,
            MultiviewLabeledBatchDict,
            MultiviewHeatmapLabeledBatchDict,
        ],
        batch_idx: int,
    ) -> None:
        """Base test step, a wrapper around the `evaluate_labeled` method."""
        self.evaluate_labeled(batch_dict, "test")


@typechecked
class SemiSupervisedTrackerMixin(object):
    """Mixin class providing training step function for semi-supervised models."""

    def get_loss_inputs_unlabeled(
        self,
        batch_dict: UnlabeledBatchDict | MultiviewUnlabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        raise NotImplementedError

    def evaluate_unlabeled(
        self,
        batch_dict: UnlabeledBatchDict | MultiviewUnlabeledBatchDict,
        stage: Literal["train", "val", "test"] | None = None,
        anneal_weight: float | torch.Tensor = 1.0,
    ) -> TensorType[(), float]:
        """Compute and log the losses on a batch of unlabeled data (frames only)."""

        # forward pass: collect predicted heatmaps and keypoints
        data_dict = self.get_loss_inputs_unlabeled(batch_dict=batch_dict)

        # compute loss on unlabeled data
        loss, log_list = self.loss_factory_unsup(
            stage=stage,
            anneal_weight=anneal_weight,
            **data_dict,
        )

        if stage:
            # log individual unsupervised losses
            for log_dict in log_list:
                self.log(
                    log_dict['name'],
                    log_dict['value'].to(self.device),
                    prog_bar=log_dict.get('prog_bar', False),
                    sync_dist=True)

        return loss

    def training_step(
        self,
        batch_dict: SemiSupervisedBatchDict | SemiSupervisedHeatmapBatchDict,
        batch_idx: int,
    ) -> dict[str, TensorType[(), float]]:
        """Training step computes and logs both supervised and unsupervised losses."""

        # on each epoch, self.total_unsupervised_importance is modified by the
        # AnnealWeight callback
        self.log(
            "total_unsupervised_importance",
            self.total_unsupervised_importance,
            prog_bar=True,
            # don't need to sync_dist because this is always the same across processes.
        )

        # computes and logs supervised losses
        # train_batch["labeled"] contains:
        # - images
        # - keypoints
        # - heatmaps
        loss_super = self.evaluate_labeled(
            batch_dict=batch_dict["labeled"],
            stage="train",
            anneal_weight=self.total_unsupervised_importance,
        )

        # computes and logs unsupervised losses
        # train_batch["unlabeled"] contains:
        # - images
        loss_unsuper = self.evaluate_unlabeled(
            batch_dict=batch_dict["unlabeled"],
            stage="train",
            anneal_weight=self.total_unsupervised_importance,
        )

        # log total loss
        total_loss = loss_super + loss_unsuper
        self.log("total_loss", total_loss, prog_bar=True, sync_dist=True)

        return {"loss": total_loss}
