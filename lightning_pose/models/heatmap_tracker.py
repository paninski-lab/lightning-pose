"""Models that produce heatmaps of keypoints from images."""

from kornia.geometry.subpix import spatial_softmax2d, spatial_expectation2d, conv_soft_argmax2d
from kornia.geometry.transform import pyrup
from omegaconf import DictConfig
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union
from typing_extensions import Literal
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from lightning_pose.data.utils import evaluate_heatmaps_at_location
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.base import (
    BaseBatchDict,
    BaseSupervisedTracker,
    HeatmapBatchDict,
    SemiSupervisedTrackerMixin,
)

patch_typeguard()  # use before @typechecked

def create_double_upsampling_layer(
        in_channels: int,
        out_channels: int,
) -> torch.nn.ConvTranspose2d:
    """Perform ConvTranspose2d to double the output shape."""
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
    )

class Upsampling_CRNN(torch.nn.Module):
    def __init__(self, num_filters_for_upsampling, num_keypoints, upsampling_factor=2, hkernel=2, hstride=2, hpad=0):
        """
        Upsampling Convolutional RNN - initialize input and hidden weights.
        """
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.pixel_shuffle = nn.PixelShuffle(2)
        if self.upsampling_factor == 2:
            self.W_pre = create_double_upsampling_layer(
                    in_channels=num_filters_for_upsampling // 4,
                    out_channels=num_keypoints,
            )
            in_channels_rnn = num_keypoints
        else:
            in_channels_rnn = num_filters_for_upsampling // 4,
    
        self.W_f = create_double_upsampling_layer(
                in_channels=in_channels_rnn,
                out_channels=num_keypoints,
        )
        H_f_layers = []
        H_f_layers.append(nn.Conv2d(
                    in_channels=num_keypoints,
                    out_channels=num_keypoints,
                    kernel_size=(hkernel, hkernel),
                    stride=(hstride, hstride),
                    padding=(hpad, hpad),
                    groups=num_keypoints,
                )
        )
        H_f_layers.append(nn.ConvTranspose2d(
                    in_channels=num_keypoints,
                    out_channels=num_keypoints,
                    kernel_size=(hkernel, hkernel),
                    stride=(hstride, hstride),
                    padding=(hpad, hpad),
                    output_padding=(hpad, hpad),
                    groups=num_keypoints,
                )
        ) 
        self.H_f = nn.Sequential(*H_f_layers)

        
        self.W_b = create_double_upsampling_layer(
                in_channels=in_channels_rnn,
                out_channels=num_keypoints,
        )
        H_b_layers = []
        H_b_layers.append(nn.Conv2d(
                    in_channels=num_keypoints,
                    out_channels=num_keypoints,
                    kernel_size=(hkernel, hkernel),
                    stride=(hstride, hstride),
                    padding=(hpad, hpad),
                    groups=num_keypoints,
                )
        )
        H_b_layers.append(nn.ConvTranspose2d(
                    in_channels=num_keypoints,
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
        self.layers = torch.nn.ModuleList([self.W_pre, self.W_f, self.H_f, self.W_b, self.H_b])
        
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

    def forward(self, representations):
        representations = torch.permute(representations, (4, 0, 1, 2, 3)) #frames, batch, features, rep_height, rep_width
        if self.upsampling_factor == 2:
            #upsample once before passing through RNN
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
            x_f = self.W_f(x_tensor[0])
            for frame_batch in x_tensor[1:]: #forward pass
                x_f = self.W_f(frame_batch) + self.H_f(x_f)
            x_tensor_b = torch.flip(x_tensor, dims=[0])
            x_b = self.W_b(x_tensor_b[0])
            for frame_batch in x_tensor_b[1:]: #backwards pass
                x_b = self.W_b(frame_batch) + self.H_b(x_b)
        else:
            x_tensor = representations
            x_f = self.W_f(x_tensor[0])
            for frame_batch in x_tensor[1:]: #forward pass
                x_f = self.W_f(self.pixel_shuffle(frame_batch)) + self.H_f(x_f)
            x_tensor_b = torch.flip(x_tensor, dims=[0])
            x_b = self.W_b(x_tensor_b[0])
            for frame_batch in x_tensor_b[1:]: #backwards pass
                x_b = self.W_b(self.pixel_shuffle(frame_batch)) + self.H_b(x_b)          
        heatmaps = (x_f + x_b) / 2
        return heatmaps

@typechecked
class HeatmapTracker(BaseSupervisedTracker):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnet50_3d", "resnet50_contrastive",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"] = "resnet50",
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,  # change
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = True,
        do_crnn: bool = False,
    ) -> None:
        """Initialize a DLC-like model with resnet backbone.
        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            last_resnet_layer_to_get: skip final layers of backbone model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions
            do_crnn: use CRNN to improve temporal predictions
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
            do_crnn=do_crnn,
        )
        self.num_keypoints = num_keypoints
        self.num_targets = num_keypoints * 2
        self.loss_factory = loss_factory.to(self.device)
        # TODO: downsample_factor may be in mismatch between datamodule and model.
        self.downsample_factor = downsample_factor
        if self.mode != "crnn":
            self.upsampling_layers = self.make_upsampling_layers()
            self.initialize_upsampling_layers()
        self.output_shape = output_shape
        # TODO: temp=1000 works for 64x64 heatmaps, need to generalize to other shapes
        self.temperature = torch.tensor(1000.0, device=self.device)  # soft argmax temp
        self.torch_seed = torch_seed
        self.do_context = do_context
        if self.mode == "2d":
#             self.representation_fc = torch.nn.Linear(5, 1, bias=False)
            self.unnormalized_weights = nn.parameter.Parameter(torch.Tensor([[.2, .2, .2, .2, .2]]), requires_grad=False)
            self.representation_fc = lambda x: x @ torch.transpose(nn.functional.softmax(self.unnormalized_weights), 0, 1)
        elif self.mode == "3d":
            self.representation_fc = torch.nn.Linear(8, 1, bias=False)
        elif self.mode == "crnn":
            self.unnormalized_weights = nn.parameter.Parameter(torch.Tensor([[.2, .2, .2, .2, .2]]), requires_grad=False)
            self.crnn = Upsampling_CRNN(self.num_filters_for_upsampling, self.num_keypoints)
            #overwrite upsampling layers
            self.upsampling_layers = self.crnn.layers

        # use this to log auxiliary information: rmse on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        self.save_hyperparameters(ignore="loss_factory")  # cannot be pickled

    @property
    def num_filters_for_upsampling(self) -> int:
        return self.num_fc_input_features

    @property
    def coordinate_scale(self) -> TensorType[(), int]:
        return torch.tensor(2**self.downsample_factor, device=self.device)

    def run_subpixelmaxima(
        self,
        heatmaps: TensorType[
            "batch", "num_keypoints", "heatmap_height", "heatmap_width"
        ],
    ) -> Tuple[
        TensorType["batch", "num_targets"],
        TensorType["batch", "num_keypoints"],
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

        # compute confidences as softmax value at prediction
        confidences = evaluate_heatmaps_at_location(heatmaps=softmaxes, locs=preds)

        # OLD BAD WAY
        # confidences = torch.amax(softmaxes, dim=(2, 3))

        return preds.reshape(-1, self.num_targets), confidences

    def initialize_upsampling_layers(self) -> None:
        """Intialize the Conv2DTranspose upsampling layers."""
        # TODO: test that running this method changes the weights and biases
        for index, layer in enumerate(self.upsampling_layers):
            if index > 0:  # we ignore the PixelShuffle
                if isinstance(layer, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    torch.nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    torch.nn.init.constant_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)

    def make_upsampling_layers(self) -> torch.nn.Sequential:
        # Note:
        # https://github.com/jgraving/DeepPoseKit/blob/
        # cecdb0c8c364ea049a3b705275ae71a2f366d4da/deepposekit/models/DeepLabCut.py#L131
        # in their model, the pixel shuffle happens only for downsample_factor=2
        upsampling_layers = []
        upsampling_layers.append(nn.PixelShuffle(2))
#         upsampling_layers.append(nn.BatchNorm2d(self.num_filters_for_upsampling // 4))
#         upsampling_layers.append(nn.ReLU(inplace=True))
        upsampling_layers.append(
            create_double_upsampling_layer(
                in_channels=self.num_filters_for_upsampling // 4,
                out_channels=self.num_keypoints,
            )
        )  # up to here results in downsample_factor=3
        if self.downsample_factor == 2:
#             upsampling_layers.append(nn.BatchNorm2d(self.num_keypoints))
#             upsampling_layers.append(nn.ReLU(inplace=True))
            upsampling_layers.append(
                create_double_upsampling_layer(
                    in_channels=self.num_keypoints,
                    out_channels=self.num_keypoints,
                )
            )
        return nn.Sequential(*upsampling_layers)

    def heatmaps_from_representations(
        self,
        representations: Union[TensorType["batch", "features", "rep_height", "rep_width"], 
                               TensorType["batch", "features", "rep_height", "rep_width", "frames"]],
    ) -> TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Wrapper around self.upsampling_layers for type and shape assertion."""
        if self.do_context:
            if self.mode == "crnn":
                heatmaps = self.crnn(representations)
            else:
                # push through a linear layer to get the final representation
                representations: TensorType[
                    "batch", "features", "rep_height", "rep_width", 1
                ] = self.representation_fc(representations)
                # final squeeze
                representations: TensorType[
                    "batch", "features", "rep_height", "rep_width"
                ] = torch.squeeze(representations, 4)
                heatmaps = self.upsampling_layers(representations)
        else:
            heatmaps = self.upsampling_layers(representations)
        return heatmaps

    def forward(
        self,
        images: Union[
            TensorType["batch", "channels":3, "image_height", "image_width"],
            TensorType["batch", "frames", "channels":3, "image_height", "image_width"]]
    ) -> TensorType["num_valid_outputs", "num_keypoints", "heatmap_height", "heatmap_width"]:
        """Forward pass through the network."""
        # we get one representation for each desired output. 
        # in the case of unsupervised sequences + context, we have outputs for all images but the first two and last two.
        # this is all handled internally by get_representations()
        representations = self.get_representations(images, self.do_context)
        heatmaps = self.heatmaps_from_representations(representations)
        # softmax temp stays 1 here; to modify for model predictions, see constructor
        return spatial_softmax2d(heatmaps, temperature=torch.tensor([1.0]))

    def get_loss_inputs_labeled(self, batch_dict: HeatmapBatchDict) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(
            batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_targ": batch_dict["keypoints"],
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }
    
    def predict_step(self, batch: Union[dict, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict heatmaps and keypoints for a batch of video frames.
        assuming a DALI video loader is passed in 
        trainer = Trainer(devices=8, accelerator="gpu")
        predictions = trainer.predict(model, data_loader) """
        if isinstance(batch, dict): # labeled image dataloaders
            images = batch["images"]
        else: # unlabeled dali video dataloaders
            images = batch
        # images -> heatmaps
        predicted_heatmaps = self.forward(images)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        return (predicted_keypoints, confidence)


@typechecked
class SemiSupervisedHeatmapTracker(SemiSupervisedTrackerMixin, HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory,
        loss_factory_unsupervised: LossFactory,
        backbone: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnet50_3d", "resnet50_contrastive",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"] = "resnet50",
        downsample_factor: Literal[2, 3] = 2,
        pretrained: bool = True,
        last_resnet_layer_to_get: int = -3,
        output_shape: Optional[tuple] = None,
        torch_seed: int = 123,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: Optional[Union[DictConfig, dict]] = None,
        do_context: bool = True,
        do_crnn: bool = False,
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
            last_resnet_layer_to_get: skip final layers of original model
            output_shape: hard-coded image size to avoid dynamic shape
                computations
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma
            do_context: use temporal context frames to improve predictions
            do_crnn: use CRNN to improve temporal predictions
        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            last_resnet_layer_to_get=last_resnet_layer_to_get,
            output_shape=output_shape,
            torch_seed=torch_seed,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            do_context=do_context,
            do_crnn=do_crnn,
        )
        self.loss_factory_unsup = loss_factory_unsupervised.to(self.device)

        # this attribute will be modified by AnnealWeight callback during training
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))
        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(
        self,
        batch: Union[TensorType[
            "sequence_length", "RGB":3, "image_height", "image_width", float
        ], TensorType[
            "sequence_length", "context":5, "RGB":3, "image_height", "image_width", float
        ]],
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.run_subpixelmaxima(predicted_heatmaps)
        return {
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }