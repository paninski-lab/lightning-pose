import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Callable, Optional, Tuple, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import numpy as np
from pose_est_nets.utils.heatmap_tracker_utils import find_subpixel_maxima, largest_factor, format_mouse_data

patch_typeguard()

class DLC(LightningModule):
    def __init__(
        self,
        num_targets: int,
        resnet_version: int = 18,
        downsample_factor: Optional[int] = 3,
        transfer: Optional[bool] = False,
    ) -> None:
        """
        Initializes DLC model with resnet backbone
        :param num_targets: number of body parts
        :param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
            large the resnet will be.
        :param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
            meaning the entire model will be trained unless this flag is provided
        """
        super(DLC, self).__init__()
        self.__dict__.update(locals())  # todo: what is this?
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        # Using a pretrained ResNet backbone
        backbone = resnets[resnet_version](pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-2] #also excluding the penultimate pooling layer
        self.feature_extractor = nn.Sequential(*layers)
        self.upsampling_layers = []
        # TODO: Add normalization
        # TODO: Should depend on input size
        self.num_keypoints = num_targets//2
        self.downsample_factor = downsample_factor
        self.coordinate_scale = torch.tensor(2 ** downsample_factor, device = 'cuda')
        if (downsample_factor == 3):
            self.upsampling_layers += [ #shape = [batch, 2048, 12, 12]
                #nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                nn.PixelShuffle(2), 
                nn.ConvTranspose2d(in_channels = int(num_filters/4), out_channels = self.num_keypoints, kernel_size = (3, 3), stride = (2,2), padding = (1,1), output_padding = (1,1)) # [batch, 17, 48, 48]
            ]
            self.upsampling_layers = nn.Sequential(*self.upsampling_layers)
            torch.nn.init.xavier_uniform_(self.upsampling_layers[-1].weight)
            torch.nn.init.zeros_(self.upsampling_layers[-1].bias)
        elif (downsample_factor == 2):
            self.upsampling_layers += [ #shape = [batch, 2048, 12, 12]
                nn.PixelShuffle(2),
                nn.ConvTranspose2d(in_channels = int(num_filters/4), out_channels = self.num_keypoints, kernel_size = (3, 3), stride = (2,2), padding = (1,1), output_padding = (1,1)),#[batch, 17, 48, 48]
                nn.ConvTranspose2d(in_channels = self.num_keypoints, out_channels = self.num_keypoints, kernel_size = (3, 3), stride = (2,2), padding = (1,1), output_padding = (1,1)) #[batch, 17, 96, 96]
            ]
            self.upsampling_layers = nn.Sequential(*self.upsampling_layers)
            torch.nn.init.xavier_uniform_(self.upsampling_layers[-1].weight)
            torch.nn.init.zeros_(self.upsampling_layers[-1].bias)
            torch.nn.init.xavier_uniform_(self.upsampling_layers[-2].weight)
            torch.nn.init.zeros_(self.upsampling_layers[-2].bias)
        else:
            print("downsample factor not supported!")
            exit()
 
        self.batch_size = 16
        self.num_workers = 0
    @typechecked
    def forward(self, x: TensorType["batch", 3, "Height", "Width"]) -> TensorType["batch", 17, "Out_Height", "Out_Width"]: #how do I use a variable to indicate number of keypoints
        """
        Forward pass through the network
        :param x: input
        :return: output of network
        """
        #self.feature_extractor.eval()
        #with torch.no_grad():
        representations = self.feature_extractor(x)
        out = self.upsampling_layers(representations)
        return out

    @staticmethod
    @typechecked
    def heatmap_loss(y: TensorType["batch", 17, "Out_Height", "Out_Width"], y_hat: TensorType["batch", 17, "Out_Height", "Out_Width"]) -> TensorType[()]:
        """
        Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
        :param y: ground truth. shape=(num_targets, 2)
        :param y_hat: prediction. shape=(num_targets, 2)
        :return: mse loss
        """
        # apply mask, only computes loss on heatmaps where the ground truth heatmap is not all zeros (aka not an occluded keypoint)
        max_vals = torch.amax(y, dim = (2,3))
        zeros = torch.zeros(size = (y.shape[0], y.shape[1]), device = y_hat.device)
        mask = torch.eq(max_vals, zeros)
        mask = ~mask
        mask = torch.unsqueeze(mask, 2)
        mask = torch.unsqueeze(mask, 3)
        # compute loss
        loss = F.mse_loss(torch.masked_select(y_hat, mask), torch.masked_select(y, mask))
        return loss
    
    @typechecked
    #what are we doing about NANS?
    def pca_2view_loss(self, y_hat: TensorType["batch", 17, "Out_Height", "Out_Width"]) -> TensorType[()]:
        kernel_size = np.min(self.output_shape) #change from numpy to torch
        kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
        keypoints = find_subpixel_maxima(y_hat.detach(), torch.tensor(kernel_size, device = 'cuda'), torch.tensor(self.output_sigma, device = 'cuda'), self.upsample_factor, self.coordinate_scale, self.confidence_scale)
        keypoints = keypoints[:,:,:2]
        data_arr = format_mouse_data(keypoints)
        garbage_component = self.pca_param_dict["bot_1_eigenvector"]
        garbage_variance = torch.matmul(data_arr.T, garbage_component.T)
        return torch.linalg.norm(garbage_variance)
               
    def training_step(self, data, batch_idx):
        #x, y_heatmap, y_keypoints = data
        x, y = data
        # forward pass
        y_hat = self.forward(x)
        # compute loss
        loss = self.heatmap_loss(y, y_hat)
        #heatmap_loss = self.heatmap_loss(y_heatmap, y_hat)
        pca_view_loss = self.pca_2view_loss(y_hat)
        loss += (pca_view_loss/10000) #can improve scaling
        #ppca_loss = 
        # log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "pca_loss", pca_view_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, data, batch_idx):
        x, y = data
        y_hat = self.forward(x)
        # compute loss
        loss = self.heatmap_loss(y, y_hat)
        # log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        self.validation_step(data, batch_idx)
    
    def computeSubPixMax(self, heatmaps_pred, heatmaps_y, threshold):
        kernel_size = np.min(self.output_shape)
        kernel_size = (kernel_size // largest_factor(kernel_size)) + 1
        pred_keypoints = find_subpixel_maxima(heatmaps_pred.detach(), torch.tensor(kernel_size, device = heatmaps_pred.device), torch.tensor(self.output_sigma, device = heatmaps_pred.device), self.upsample_factor, self.coordinate_scale, self.confidence_scale)
        y_keypoints = find_subpixel_maxima(heatmaps_y.detach(), torch.tensor(kernel_size, device = heatmaps_pred.device), torch.tensor(self.output_sigma, device = heatmaps_pred.device), self.upsample_factor, self.coordinate_scale, self.confidence_scale)
        if threshold: # TODO: convert to vectorized selection based on bool ops
            pred_kpts_list = []
            y_kpts_list = []
            for i in range(pred_keypoints.shape[1]): # pred_keypoints is shape(1, num_keypoints, 3) the last entry being (x,y, confidence)
                if pred_keypoints[0, i, 2] > 0.001: #threshold for low confidence predictions
                    pred_kpts_list.append(pred_keypoints[0, i, :2].cpu().numpy())
                if y_keypoints[0, i, 2] > 0.001:
                    y_kpts_list.append(y_keypoints[0, i, :2].cpu().numpy())
            return torch.tensor(pred_kpts_list), torch.tensor(y_kpts_list)

        pred_keypoints = pred_keypoints[0,:,:2] #getting rid of the actual max value
        y_keypoints = y_keypoints[0,:,:2]
        return pred_keypoints, y_keypoints

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.2, patience = 20, verbose = True)
        return {'optimizer' : optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

#Might be good to write compute same padding function

