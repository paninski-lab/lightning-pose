"""Base class for resnet backbone that acts as a feature extractor."""

from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torchtyping import TensorType, patch_typeguard
import torchvision.models as models
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

patch_typeguard()  # use before @typechecked

MULTISTEPLR_MILESTONES_DEFAULT = [100, 200, 300]
MULTISTEPLR_GAMMA_DEFAULT = 0.5


@typechecked
def grab_resnet_backbone(
    resnet_version: Literal[18, 34, 50, 101, 152] = 18,
    pretrained: bool = True,
) -> models.resnet.ResNet:
    """Load resnet architecture from torchvision.

    Args:
        resnet_version: choose network depth
        pretrained: True to load weights pretrained on imagenet

    Returns:
        selected resnet architecture as a model object

    """
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    return resnets[resnet_version](pretrained)


@typechecked
def grab_layers_sequential(
    model: models.resnet.ResNet, last_layer_ind: Optional[int] = None
) -> torch.nn.modules.container.Sequential:
    """Package selected number of layers into a nn.Sequential object.

    Args:
        model: original resnet model
        last_layer_ind: final layer to pass data through

    Returns:
        potentially reduced backbone model

    """
    layers = list(model.children())[: last_layer_ind + 1]
    return nn.Sequential(*layers)


class BaseBatchDict(TypedDict):
    """Class for finer control over typechecking."""

    images: Union[TensorType["batch", "RGB":3, "image_height", "image_width", float], 
                  TensorType["batch",  "frames", "RGB":3, "image_height", "image_width", float]]
    keypoints: TensorType["batch", "num_targets", float]
    idxs: TensorType["batch", int]


class HeatmapBatchDict(BaseBatchDict):
    """Class for finer control over typechecking."""

    heatmaps: TensorType[
        "batch", "num_keypoints", "heatmap_height", "heatmap_width", float
    ]


class SemiSupervisedBatchDict(TypedDict):
    """Class for finer control over typechecking."""

    labeled: BaseBatchDict
    unlabeled: TensorType[
        "sequence_length", "RGB":3, "image_height", "image_width", float
    ]


class SemiSupervisedHeatmapBatchDict(TypedDict):
    """Class for finer control over typechecking."""

    labeled: HeatmapBatchDict
    unlabeled: TensorType[
        "sequence_length", "RGB":3, "image_height", "image_width", float
    ]


class BaseFeatureExtractor(LightningModule):
    """Object that contains the base resnet feature extractor."""

    def __init__(
        self,
        resnet_version: Literal[18, 34, 50, 101, 152] = 18,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[dict] = None,
    ) -> None:
        """A ResNet model that takes in images and generates features.

        ResNets will be loaded from torchvision and can be either pre-trained
        on ImageNet or randomly initialized. These were originally used for
        classification tasks, so we truncate their final fully connected layer.

        Args:
            resnet_version: which ResNet version to use; defaults to 18
            pretrained: True to load weights pretrained on imagenet
            last_resnet_layer_to_get: Defaults to -2.
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers

        """
        super().__init__()
        print("\n Initializing a {} instance.".format(self._get_name()))

        self.resnet_version = resnet_version
        base = grab_resnet_backbone(
            resnet_version=self.resnet_version, pretrained=pretrained
        )
        self.num_fc_input_features = base.fc.in_features
        self.backbone = grab_layers_sequential(
            model=base,
            last_layer_ind=last_resnet_layer_to_get,
        )
#         self.representation_fc = torch.nn.Linear(5, 1, bias=False)

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

    def get_representations(
        self,
        images: Union[TensorType["batch", "RGB":3, "image_height", "image_width", float], 
                      TensorType["batch",  "frames", "RGB":3, "image_height", "image_width", float]],
        do_context: bool,
    ) -> TensorType["batch", "features", "rep_height", "rep_width", float]:
        """Forward pass from images to feature maps.

        Wrapper around the backbone's feature_extractor() method for typechecking
        purposes.
        See tests/models/test_base.py for example shapes.

        Args:
            images: a batch of images
            do_context: whether or not to use extra frames of context

        Returns:
            a representation of the images; features differ as a function of resnet
            version. Representation height and width differ as a function of image
            dimensions, and are not necessarily equal.

        """
        if do_context:
#             images_by_frame_group: TensorType["frames", "batch", "channels":3, "image_height", "image_width"] = torch.permute(images, (1, 0, 2, 3, 4))
#             outputs = []
            frames_batch_shape = images.shape[0] * images.shape[1]
            channels = images.shape[2]
            image_height = images.shape[3]
            image_width = images.shape[4]
            images_batch_frames: TensorType["batch*frames", "channels":3, "image_height", "image_width"] = images.reshape(frames_batch_shape, channels,
                                                                                                                          image_height, image_width)
            outputs: TensorType["batch*frames", "features", "rep_height", "rep_width"] = self.backbone(images_batch_frames)
            outputs: TensorType["batch", "frames", "features", "rep_height", "rep_width"] = outputs.reshape(images.shape[0], images.shape[1], outputs.shape[1],
                                                                                                            outputs.shape[2], outputs.shape[3])
            representations: TensorType["batch", "features", "rep_height", "rep_width", "frames"] = torch.permute(outputs, (0, 2, 3, 4, 1))
                
                #             for image_batch in images_by_frame_group:
#                 output = self.backbone(image_batch)
#                 output = torch.unsqueeze(output, dim=1)
#                 outputs.append(output)
#             outputs: TensorType["batch", "frames", "features", "rep_height", "rep_width"] = torch.cat(outputs, dim=1)
        else:
            image_batch = images
            representations = self.backbone(image_batch)
            
        return representations

    def forward(self, images):
        """Forward pass from images to representations.

        Wrapper around self.get_representations().
        Fancier childern models will use get_representations() in their forward
        methods.

        Args:
            images (torch.tensor(float)): a batch of images.

        Returns:
            torch.tensor(float): a representation of the images.
        """
        return self.get_representations(images)

    def configure_optimizers(self):
        """Select optimizer, lr scheduler, and metric for monitoring."""

        # standard adam optimizer
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)

        # define a scheduler that reduces the base learning rate
        if self.lr_scheduler == "multisteplr" or self.lr_scheduler == "multistep_lr":

            if self.lr_scheduler_params is None:
                milestones = MULTISTEPLR_MILESTONES_DEFAULT
                gamma = MULTISTEPLR_GAMMA_DEFAULT
            else:
                milestones = self.lr_scheduler_params.get(
                    "milestones", MULTISTEPLR_MILESTONES_DEFAULT)
                gamma = self.lr_scheduler_params.get(
                    "gamma", MULTISTEPLR_GAMMA_DEFAULT)

            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        else:
            raise NotImplementedError(
                "'%s' is an invalid LR scheduler" % self.lr_scheduler
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_supervised_loss",
        }


class BaseSupervisedTracker(BaseFeatureExtractor):
    """Base class for supervised trackers."""

    @typechecked
    def get_loss_inputs_labeled(
        self,
        batch_dict: Union[BaseBatchDict, HeatmapBatchDict],
    ) -> dict:
        """Return predicted coordinates for a batch of data."""
        raise NotImplementedError

    @typechecked
    def evaluate_labeled(
        self,
        batch_dict: Union[BaseBatchDict, HeatmapBatchDict],
        stage: Optional[Literal["train", "val", "test"]] = None,
    ) -> TensorType[(), float]:
        """Compute and log the losses on a batch of labeled data."""

        # forward pass; collected true and predicted heatmaps, keypoints
        data_dict = self.get_loss_inputs_labeled(batch_dict=batch_dict)

        # compute and log loss on labeled data
        loss, log_list = self.loss_factory(stage=stage, **data_dict)

        # compute and log rmse loss on labeled data
        loss_rmse, _ = self.rmse_loss(stage=stage, **data_dict)

        if stage:
            # log overall supervised loss
            self.log(f"{stage}_supervised_loss", loss, prog_bar=True)
            # log supervised rmse
            self.log(f"{stage}_supervised_rmse", loss_rmse)
            # log individual supervised losses
            for log_dict in log_list:
                self.log(**log_dict)

        return loss

    @typechecked
    def training_step(
        self,
        train_batch: Union[BaseBatchDict, HeatmapBatchDict],
        batch_idx: int,
    ) -> Dict[str, TensorType[(), float]]:
        """Base training step, a wrapper around the `evaluate_labeled` method."""
        print(self.representation_fc.weight)
        loss = self.evaluate_labeled(train_batch, "train")
        return {"loss": loss}

    @typechecked
    def validation_step(
        self,
        val_batch: Union[BaseBatchDict, HeatmapBatchDict],
        batch_idx: int,
    ) -> None:
        """Base validation step, a wrapper around the `evaluate_labeled` method."""
        self.evaluate_labeled(val_batch, "val")

    @typechecked
    def test_step(
        self,
        test_batch: Union[BaseBatchDict, HeatmapBatchDict],
        batch_idx: int,
    ) -> None:
        """Base test step, a wrapper around the `evaluate_labeled` method."""
        self.evaluate_labeled(test_batch, "test")

    @typechecked
    def configure_optimizers(self) -> dict:
        """Select optimizer, lr scheduler, and metric for monitoring."""

        if getattr(self, "upsampling_layers", None) is not None:

            # single optimizer with single learning rate
            params = [
                # {"params": self.backbone.parameters()},
                #  don't uncomment above line; the BackboneFinetuning callback should
                # add backbone to the params.
                {
                    "params": self.upsampling_layers.parameters()
                },  # important this is the 0th element, for BackboneFinetuning callback
                {
                    "params": self.representation_fc.parameters()
                }, 
            ]

        else:
            # standard adam optimizer
            params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = Adam(params, lr=1e-3)

        # define a scheduler that reduces the base learning rate
        if self.lr_scheduler == "multisteplr" or self.lr_scheduler == "multistep_lr":

            if self.lr_scheduler_params is None:
                milestones = MULTISTEPLR_MILESTONES_DEFAULT
                gamma = MULTISTEPLR_GAMMA_DEFAULT
            else:
                milestones = self.lr_scheduler_params.get(
                    "milestones", MULTISTEPLR_MILESTONES_DEFAULT)
                gamma = self.lr_scheduler_params.get(
                    "gamma", MULTISTEPLR_GAMMA_DEFAULT)

            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        else:
            raise NotImplementedError(
                "'%s' is an invalid LR scheduler" % self.lr_scheduler
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_supervised_loss",
        }


class SemiSupervisedTrackerMixin(object):
    """Mixin class providing training step function for semi-supervised models."""

    @typechecked
    def get_loss_inputs_unlabeled(self, batch: torch.Tensor) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        raise NotImplementedError

    @typechecked
    def evaluate_unlabeled(
        self,
        batch: TensorType["batch", "channels":3, "image_height", "image_width", float],
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

    @typechecked
    def training_step(
        self,
        train_batch: Union[SemiSupervisedBatchDict, SemiSupervisedHeatmapBatchDict],
        batch_idx: int,
    ) -> Dict[str, TensorType[(), float]]:
        """Training step computes and logs both supervised and unsupervised losses."""

        # on each epoch, self.total_unsupervised_importance is modified by the
        # AnnealWeight callback
        print(self.representation_fc.weight)
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

    def configure_optimizers(self):
        """Single optimizer with different learning rates."""

        if getattr(self, "upsampling_layers", None) is not None:
            # check if heatmap
            params = [
                # {"params": self.backbone.parameters()},
                #  don't uncomment above line; the BackboneFinetuning callback should
                # add backbone to the params.
                {
                    "params": self.upsampling_layers.parameters()
                },  # important this is the 0th element, for BackboneFinetuning callback
                {
                    "params": self.representation_fc.parameters()
                }, 
            ]

        else:
            # standard adam optimizer
            params = filter(lambda p: p.requires_grad, self.parameters())

        # define different learning rate for weights in front of unsupervised losses
        if len(self.loss_factory_unsup.loss_weights_parameter_dict) > 0:
            params.append(
                {
                    "params": self.loss_factory_unsup.loss_weights_parameter_dict.parameters(),
                    "lr": 1e-2,
                }
            )

        optimizer = Adam(params, lr=1e-3)

        # define a scheduler that reduces the base learning rate
        if self.lr_scheduler == "multisteplr" or self.lr_scheduler == "multistep_lr":

            if self.lr_scheduler_params is None:
                milestones = MULTISTEPLR_MILESTONES_DEFAULT
                gamma = MULTISTEPLR_GAMMA_DEFAULT
            else:
                milestones = self.lr_scheduler_params.get(
                    "milestones", MULTISTEPLR_MILESTONES_DEFAULT)
                gamma = self.lr_scheduler_params.get(
                    "gamma", MULTISTEPLR_GAMMA_DEFAULT)

            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        else:
            raise NotImplementedError(
                "'%s' is an invalid LR scheduler" % self.lr_scheduler
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_supervised_loss",
        }

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
