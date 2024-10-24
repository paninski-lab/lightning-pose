"""Base class for backbone that acts as a feature extractor."""

from typing import Any, Dict, Literal, Optional, Union

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
    MultiviewHeatmapLabeledBatchDict,
    MultiviewLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    SemiSupervisedBatchDict,
    SemiSupervisedHeatmapBatchDict,
    UnlabeledBatchDict,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "normalized_to_bbox",
    "convert_bbox_coords",
    "get_context_from_sequence",
    "BaseFeatureExtractor",
    "BaseSupervisedTracker",
    "SemiSupervisedTrackerMixin",
]

MULTISTEPLR_MILESTONES_DEFAULT = [100, 200, 300]
MULTISTEPLR_GAMMA_DEFAULT = 0.5

# list of all allowed backbone options
ALLOWED_BACKBONES = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet50_contrastive",  # needs extra install: pip install -e .[extra_models]
    "resnet50_animal_apose",
    "resnet50_animal_ap10k",
    "resnet50_human_jhmdb",
    "resnet50_human_res_rle",
    "resnet50_human_top_res",
    "resnet50_human_hand",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    # "vit_h_sam",
    "vit_b_sam",
]


def normalized_to_bbox(
    keypoints: TensorType["batch", "num_keypoints", "xy":2],
    bbox: TensorType["batch", "xyhw":4]
) -> TensorType["batch", "num_keypoints", "xy":2]:
    if keypoints.shape[0] == bbox.shape[0]:
        # normal batch
        keypoints[:, :, 0] *= bbox[:, 3].unsqueeze(1)  # scale x by box width
        keypoints[:, :, 0] += bbox[:, 0].unsqueeze(1)  # add bbox x offset
        keypoints[:, :, 1] *= bbox[:, 2].unsqueeze(1)  # scale y by box height
        keypoints[:, :, 1] += bbox[:, 1].unsqueeze(1)  # add bbox y offset
    else:
        # context batch; we don't have predictions for first/last two frames
        keypoints[:, :, 0] *= bbox[2:-2, 3].unsqueeze(1)  # scale x by box width
        keypoints[:, :, 0] += bbox[2:-2, 0].unsqueeze(1)  # add bbox x offset
        keypoints[:, :, 1] *= bbox[2:-2, 2].unsqueeze(1)  # scale y by box height
        keypoints[:, :, 1] += bbox[2:-2, 1].unsqueeze(1)  # add bbox y offset
    return keypoints


def convert_bbox_coords(
    batch_dict: Union[
        HeatmapLabeledBatchDict,
        MultiviewHeatmapLabeledBatchDict,
        MultiviewUnlabeledBatchDict,
        UnlabeledBatchDict,
    ],
    predicted_keypoints: TensorType["batch", "num_targets"],
) -> TensorType["batch", "num_targets"]:
    """Transform keypoints from bbox coordinates to absolute frame coordinates."""
    num_targets = predicted_keypoints.shape[1]
    num_keypoints = num_targets // 2
    # reshape from (batch, n_targets) back to (batch, n_key, 2), in x,y order
    predicted_keypoints = predicted_keypoints.reshape((-1, num_keypoints, 2))
    # divide by image dims to get 0-1 normalized coordinates
    if "images" in batch_dict.keys():
        predicted_keypoints[:, :, 0] /= batch_dict["images"].shape[-1]  # -1 dim is width "x"
        predicted_keypoints[:, :, 1] /= batch_dict["images"].shape[-2]  # -2 dim is height "y"
    else:  # we have unlabeled dict, 'frames' instead of 'images'
        predicted_keypoints[:, :, 0] /= batch_dict["frames"].shape[-1]  # -1 dim is width "x"
        predicted_keypoints[:, :, 1] /= batch_dict["frames"].shape[-2]  # -2 dim is height "y"
    # multiply and add by bbox dims (x,y,h,w)
    if "num_views" in batch_dict.keys() and int(batch_dict["num_views"].max()) > 1:
        unique = batch_dict["num_views"].unique()
        if len(unique) != 1:
            raise ValueError(
                f"each batch element must contain the same number of views; "
                f"found elements with {unique} views"
            )
        num_views = int(unique)
        num_keypoints_per_view = num_keypoints // num_views
        for v in range(num_views):
            idx_beg = num_keypoints_per_view * v
            idx_end = num_keypoints_per_view * (v + 1)
            predicted_keypoints[:, idx_beg:idx_end, :] = normalized_to_bbox(
                predicted_keypoints[:, idx_beg:idx_end, :],
                batch_dict["bbox"][:, 4 * v:4 * (v + 1)],
            )
    else:
        predicted_keypoints = normalized_to_bbox(predicted_keypoints, batch_dict["bbox"])
    # return new keypoints, reshaped to (batch, num_targets)
    return predicted_keypoints.reshape((-1, num_targets))


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
        **kwargs: Any,
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
            do_context: include temporal context when processing each frame
            image_size: height/width of frames, for ViT models only
            model_type: type of model

        """
        super().__init__()
        if self.local_rank == 0:
            print(f"\n Initializing a {self._get_name()} instance.")

        self.backbone_arch = backbone

        if "sam" in self.backbone_arch:
            from lightning_pose.models.backbones.vits import build_backbone
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
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"],
            TensorType["seq_len", "channels":3, "image_height", "image_width"],
            TensorType["batch", "views", "frames", "channels":3, "image_height", "image_width"],
            TensorType["seq_len", "view", "frames", "channels":3, "image_height", "image_width"],
        ],
        is_multiview: bool = False,
    ) -> Union[
        TensorType["new_batch", "features", "rep_height", "rep_width"],
        TensorType["new_batch", "features", "rep_height", "rep_width", "frames"],
    ]:
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

    def get_parameters(self):
        if getattr(self, "upsampling_layers", None) is not None:
            params = [
                {"params": self.backbone.parameters(), "lr": 0, "name": "backbone"},
                {"params": self.upsampling_layers.parameters(), "name": "upsampling"},
            ]
        else:
            params = filter(lambda p: p.requires_grad, self.parameters())

        return params

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
    ) -> Dict[str, TensorType[(), float]]:
        """Base training step, a wrapper around the `evaluate_labeled` method."""
        loss = self.evaluate_labeled(batch_dict, "train")
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

    def get_loss_inputs_unlabeled(self, batch_dict: UnlabeledBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        raise NotImplementedError

    def evaluate_unlabeled(
        self,
        batch_dict: Union[UnlabeledBatchDict, MultiviewUnlabeledBatchDict],
        stage: Optional[Literal["train", "val", "test"]] = None,
        anneal_weight: Union[float, torch.Tensor] = 1.0,
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
        batch_dict: Union[SemiSupervisedBatchDict, SemiSupervisedHeatmapBatchDict],
        batch_idx: int,
    ) -> Dict[str, TensorType[(), float]]:
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
