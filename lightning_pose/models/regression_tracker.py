"""Models that produce (x, y) coordinates of keypoints from images."""

from omegaconf import DictConfig
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal

from lightning_pose.data.utils import (
    evaluate_heatmaps_at_location,
    undo_affine_transform,
    BaseLabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import BaseSupervisedTracker, SemiSupervisedTrackerMixin

patch_typeguard()  # use before @typechecked


@typechecked
class RegressionTracker(BaseSupervisedTracker):
    """Base model that produces (x, y) predictions of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
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
        ] = "resnet50",
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = False,
    ) -> None:
        """Base model that produces (x, y) coordinates of keypoints from images.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of backbone model
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=do_context,
        )
        self.num_keypoints = num_keypoints
        self.num_targets = self.num_keypoints * 2
        self.loss_factory = loss_factory
        self.final_layer = nn.Linear(self.num_fc_input_features, self.num_targets)
        self.torch_seed = torch_seed
        self.do_context = do_context
        if self.mode == "2d":
            self.unnormalized_weights = nn.parameter.Parameter(
                torch.Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]), requires_grad=False)
            self.representation_fc = lambda x: x @ torch.transpose(
                nn.functional.softmax(self.unnormalized_weights, dim=1), 0, 1)
        elif self.mode == "3d":
            self.unnormalized_weights = nn.parameter.Parameter(
                torch.Tensor([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]),
                requires_grad=False
            )
            self.representation_fc = lambda x: x @ torch.transpose(
                nn.functional.softmax(self.unnormalized_weights, dim=1), 0, 1)

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters(ignore="loss_factory")  # cannot be pickled

    def forward(
        self,
        images: Union[
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"],
        ],
    ) -> TensorType["num_valid_outputs", "two_x_num_keypoints"]:
        """Forward pass through the network."""
        # see input lines for shape of "images"
        representations = self.get_representations(images)
        # handle context frames first
        if (self.mode == "2d" and self.do_context) or self.mode == "3d":
            # push through a linear layer to get the final representation
            # input shape (batch, features, rep_height, rep_width, frames)
            representations: TensorType[
                "batch", "features", "rep_height", "rep_width", "frames"
            ] = self.representation_fc(representations)
            # final squeeze
            representations: TensorType[
                "batch", "features", "rep_height", "rep_widht"
            ] = torch.squeeze(representations, 4)
        # "representations" is shape (batch, features, rep_height, rep_width)
        reps_reshaped = representations.reshape(representations.shape[0], representations.shape[1])
        # after reshaping, is shape (batch, features)
        out = self.final_layer(reps_reshaped)
        # "out" is shape (num_valid_outputs, 2 * num_keypoints) where `num_valid_outputs` is not
        # necessarily the same as `batch` for context models (using unlabeled video data)
        return out

    def get_loss_inputs_labeled(self, batch_dict: BaseLabeledBatchDict) -> dict:
        """Return predicted coordinates for a batch of data."""
        predicted_keypoints = self.forward(batch_dict["images"])
        return {
            "keypoints_targ": batch_dict["keypoints"],
            "keypoints_pred": predicted_keypoints,
        }

    def predict_step(
        self,
        batch: Union[BaseLabeledBatchDict, UnlabeledBatchDict],
        batch_idx: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict keypoints for a batch of video frames.

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
        # images -> keypoints
        predicted_keypoints = self.forward(images)
        # regression model does not include a notion of confidence, set to all zeros
        confidence = torch.zeros((predicted_keypoints.shape[0], predicted_keypoints.shape[1] // 2))
        return predicted_keypoints, confidence


@typechecked
class SemiSupervisedRegressionTracker(SemiSupervisedTrackerMixin, RegressionTracker):
    """Model produces vectors of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        loss_factory_unsupervised: LossFactory,
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
        ] = "resnet50",
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -2,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = False,
    ) -> None:
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of original model
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=do_context,
        )
        self.loss_factory_unsup = loss_factory_unsupervised
        loss_names = loss_factory_unsupervised.loss_instance_dict.keys()
        if "unimodal_mse" in loss_names or "unimodal_wasserstein" in loss_names:
            raise ValueError("cannot use unimodal loss in regression tracker")

        # this attribute will be modified by AnnealWeight callback during training
        self.total_unsupervised_importance = torch.tensor(1.0)
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))

    def get_loss_inputs_unlabeled(self, batch: UnlabeledBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        predicted_keypoints = self.forward(batch["frames"])
        # undo augmentation if needed
        if batch["transforms"].shape[-1] == 3:
            # reshape to (seq_len, n_keypoints, 2)
            pred_kps = torch.reshape(predicted_keypoints, (predicted_keypoints.shape[0], -1, 2))
            # undo
            pred_kps = undo_affine_transform(pred_kps, batch["transforms"])
            # reshape to (seq_len, n_keypoints * 2)
            predicted_keypoints = torch.reshape(pred_kps, (pred_kps.shape[0], -1))
        return {"keypoints_pred": predicted_keypoints}
