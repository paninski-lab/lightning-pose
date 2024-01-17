"""Models that produce (x, y) coordinates of keypoints from images."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.utils import (
    BaseLabeledBatchDict,
    UnlabeledBatchDict,
    undo_affine_transform,
)
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    ALLOWED_BACKBONES,
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)

# to ignore imports for sphix-autoapidoc
__all__ = [
    "RegressionTracker",
    "SemiSupervisedRegressionTracker",
]


class RegressionTracker(BaseSupervisedTracker):
    """Base model that produces (x, y) predictions of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: Optional[LossFactory] = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Base model that produces (x, y) coordinates of keypoints from images.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """

        # for reproducible weight initialization
        torch.manual_seed(torch_seed)

        if "vit" in backbone:
            raise ValueError("Regression trackers are not compatible with ViT backbones")

        # for backwards compatibility
        if "do_context" in kwargs.keys():
            del kwargs["do_context"]

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            model_type="regression",
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        self.num_targets = self.num_keypoints * 2
        self.loss_factory = loss_factory
        self.final_layer = nn.Linear(self.num_fc_input_features, self.num_targets)
        self.torch_seed = torch_seed

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedRegressionTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: TensorType["batch", "channels":3, "image_height", "image_width"]
    ) -> TensorType["batch", "two_x_num_keypoints"]:
        """Forward pass through the network."""
        # see input lines for shape of "images"
        representations = self.get_representations(images)
        # "representations" is shape (batch, features, rep_height, rep_width)
        reps_reshaped = representations.reshape(representations.shape[0], representations.shape[1])
        # after reshaping, is shape (batch, features)
        out = self.final_layer(reps_reshaped)
        # "out" is shape (batch, 2 * num_keypoints)
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
        batch_dict: Union[BaseLabeledBatchDict, UnlabeledBatchDict],
        batch_idx: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]
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
        loss_factory: Optional[LossFactory] = None,
        loss_factory_unsupervised: Optional[LossFactory] = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        pretrained: bool = True,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            backbone: ResNet or EfficientNet variant to be used
            pretrained: True to load pretrained imagenet weights
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
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )
        self.loss_factory_unsup = loss_factory_unsupervised
        loss_names = loss_factory_unsupervised.loss_instance_dict.keys()
        if "unimodal_mse" in loss_names or "unimodal_wasserstein" in loss_names:
            raise ValueError("cannot use unimodal loss in regression tracker")

        # this attribute will be modified by AnnealWeight callback during training
        self.total_unsupervised_importance = torch.tensor(1.0)
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))

    def get_loss_inputs_unlabeled(self, batch_dict: UnlabeledBatchDict) -> Dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        predicted_keypoints = self.forward(batch_dict["frames"])
        # undo augmentation if needed
        if batch_dict["transforms"].shape[-1] == 3:
            # reshape to (seq_len, n_keypoints, 2)
            pred_kps = torch.reshape(predicted_keypoints, (predicted_keypoints.shape[0], -1, 2))
            # undo
            pred_kps = undo_affine_transform(pred_kps, batch_dict["transforms"])
            # reshape to (seq_len, n_keypoints * 2)
            predicted_keypoints = torch.reshape(pred_kps, (pred_kps.shape[0], -1))
        return {"keypoints_pred": predicted_keypoints}
