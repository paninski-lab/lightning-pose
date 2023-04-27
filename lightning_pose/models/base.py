"""Base class for backbone that acts as a feature extractor."""

from omegaconf import DictConfig
from lightning.pytorch import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchtyping import TensorType
import torchvision.models as tvmodels
from typeguard import typechecked
from typing import Dict, Literal, Optional, Tuple, Union

from collections import OrderedDict
from segment_anything import sam_model_registry

from lightning_pose.data.utils import (
    BaseLabeledBatchDict,
    HeatmapLabeledBatchDict,
    UnlabeledBatchDict,
    SemiSupervisedBatchDict,
    SemiSupervisedHeatmapBatchDict,
)


MULTISTEPLR_MILESTONES_DEFAULT = [100, 200, 300]
MULTISTEPLR_GAMMA_DEFAULT = 0.5


@typechecked
def grab_layers_sequential(model, last_layer_ind: int) -> torch.nn.Sequential:
    """Package selected number of layers into a nn.Sequential object.

    Args:
        model: original resnet or efficientnet model
        last_layer_ind: final layer to pass data through

    Returns:
        potentially reduced backbone model

    """
    layers = list(model.children())[: last_layer_ind + 1]
    return nn.Sequential(*layers)


@typechecked
def grab_layers_sequential_3d(model, last_layer_ind: int) -> torch.nn.Sequential:
    """This is to use a 3d model to extract features"""
    # the AvgPool3d halves the feature maps dims
    layers = list(model.children())[0][:last_layer_ind + 1] + \
             [nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))]
    return nn.Sequential(*layers)


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
        backbone: Literal[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnet50_3d",
            "resnet50_contrastive",
            "resnet50_animal_apose",
            "resnet50_animal_ap10k",
            "resnet50_human_jhmdb",
            "resnet50_human_res_rle",
            "resnet50_human_top_res",
            "vit_h_sam",
        ] = "resnet50",
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = False,
    ) -> None:
        """A CNN model that takes in images and generates features.

        ResNets will be loaded from torchvision and can be either pre-trained
        on ImageNet or randomly initialized. These were originally used for
        classification tasks, so we truncate their final fully connected layer.

        Args:
            backbone: which backbone version to use; defaults to resnet50
            pretrained: True to load weights pretrained on imagenet
            last_resnet_layer_to_get: Defaults to -2.
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers

        """
        super().__init__()
        print("\n Initializing a {} instance.".format(self._get_name()))

        self.backbone_arch = backbone
        self.mode = "2d"

        # load backbone weights
        if "3d" in backbone:
            base = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
            self.mode = "3d"

        elif backbone == "resnet50_contrastive":
            # load resnet50 pretrained using SimCLR on imagenet
            from pl_bolts.models.self_supervised import SimCLR

            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            base = simclr.encoder

        elif "resnet50_animal" in backbone:
            base = getattr(tvmodels, "resnet50")(pretrained=False)
            backbone_type = "_".join(backbone.split("_")[2:])
            if backbone_type == "apose":
                anim_weights = "https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth"
            else:
                anim_weights = "https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth"

            state_dict = torch.hub.load_state_dict_from_url(anim_weights)["state_dict"]
            new_state_dict = OrderedDict()
            for key in state_dict:
                if "backbone" in key:
                    new_key = ".".join(key.split(".")[1:])
                    new_state_dict[new_key] = state_dict[key]
            base.load_state_dict(new_state_dict, strict=False)

        elif "resnet50_human" in backbone:
            base = getattr(tvmodels, "resnet50")(pretrained=False)
            backbone_type = "_".join(backbone.split("_")[2:])
            if backbone_type == "jhmdb":
                hum_weights = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth"
            elif backbone_type == "res_rle":
                hum_weights = "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_rle-5f92a619_20220504.pth"
            elif backbone_type == "top_res":
                hum_weights = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth"

            state_dict = torch.hub.load_state_dict_from_url(hum_weights)["state_dict"]
            new_state_dict = OrderedDict()
            for key in state_dict:
                if "backbone" in key:
                    new_key = ".".join(key.split(".")[1:])
                    new_state_dict[new_key] = state_dict[key]
            base.load_state_dict(new_state_dict, strict=False)
        elif "vit_h_sam" in backbone:
            checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
            base = sam_model_registry["vit_h"]()
            base.load_state_dict(state_dict)
            self.mode = "transformer"
            
        else:
            # load resnet or efficientnet models from torchvision.models
            base = getattr(tvmodels, backbone)(pretrained=pretrained)

        # get truncated version of backbone
        if "3d" in backbone:
            self.backbone = grab_layers_sequential_3d(
                model=base, last_layer_ind=last_resnet_layer_to_get
            )
        elif 'sam' in backbone:
            self.backbone = base.image_encoder
        else:
            self.backbone = grab_layers_sequential(
                model=base, last_layer_ind=last_resnet_layer_to_get,
            )

        # compute number of input features
        if "resnet" in backbone and "3d" not in backbone:
            self.num_fc_input_features = base.fc.in_features
        elif "eff" in backbone:
            self.num_fc_input_features = base.classifier[-1].in_features
        elif "3d" in backbone:
            self.num_fc_input_features = base.blocks[-1].proj.in_features // 2
        elif 'sam' in backbone:
            self.num_fc_input_features = self.backbone.neck[-2].in_channels

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.do_context = do_context

        self.backbone.cuda()

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
        if self.mode == "2d":
            if self.do_context:
                if len(images.shape) == 5:
                    # non-consecutive sequences. can be used for supervised and unsupervised models
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
                image_batch = images
                representations = self.backbone(image_batch)

        elif self.mode == "3d":
            # reshape to (batch, channels, frames, img_height, img_width)
            images = torch.permute(images, (0, 2, 1, 3, 4))
            # turn (0, 1, 2, 3, 4) into (0, 1, 1, 2, 2, 3, 3, 4)
            images = torch.repeat_interleave(images, 2, dim=2)[:, :, 1:-1, ...]
            output = self.backbone(images)
            # representations = torch.mean(output, dim=2)
            representations: TensorType[
                "batch", "features", "rep_height", "rep_width", "frames"
            ] = torch.permute(output, (0, 1, 3, 4, 2))
        elif self.mode == "transformer":
            with torch.no_grad(): # TODO: temporary no_grad
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

        # define different learning rate for weights in front of unsupervised losses
        if len(self.loss_factory_unsup.loss_weights_parameter_dict) > 0:
            params.append({
                "params": self.loss_factory_unsup.loss_weights_parameter_dict.parameters(),
                "lr": 1e-2,
            })

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
    #     if self.learn_weights:
    #         params_weights = [{"params": self.loss_weights_dict.parameters()}]
    #         optimizer_weights = Adam(params_weights, lr=1e-3)
    #         optimizers.append(optimizer_weights)
    #         scheduler_weights = MultiStepLR(
    #             optimizer, milestones=[100, 200, 300], gamma=0.5
    #         )
    #         lr_schedulers.append(scheduler_weights)
    #
    #     return optimizers, lr_schedulers
