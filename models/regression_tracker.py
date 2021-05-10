import torch
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from typing import Any, Callable, Optional, Tuple, List


class RegressionTracker(LightningModule):
	def __init__(self,
				num_targets: int,
				resnet_version: int = 18,
				transfer: Optional[bool] = False
				 ) -> None:
		"""
		Initializes regression tracker model with resnet backbone
		:param num_targets: number of body parts
		:param resnet_version: The ResNet variant to be used (e.g. 18, 34, 50, 101, or 152). Essentially specifies how
			large the resnet will be.
		:param transfer:  Flag to indicate whether this is a transfer learning task or not; defaults to false,
			meaning the entire model will be trained unless this flag is provided
		"""
		super(RegressionTracker, self).__init__()
		self.__dict__.update(locals())  # todo: what is this?
		resnets = {
			18: models.resnet18, 34: models.resnet34,
			50: models.resnet50, 101: models.resnet101,
			152: models.resnet152
		}
		# Using a pretrained ResNet backbone
		self.resnet_model = resnets[resnet_version](pretrained=True)
		# replace the final fc layer by a new trainable one
		linear_size = list(self.resnet_model.children())[-1].in_features
		self.resnet_model.fc = nn.Linear(linear_size, num_targets)

		# freeze all layers but the last
		if transfer:
			for child in list(self.resnet_model.children())[:-1]:
				for param in child.parameters():
					param.requires_grad = False
			print('Froze all layers but the last.')

	def forward(self,
				x: torch.tensor
				) -> torch.tensor:
		"""
		Forward pass through the network
		:param x: input
		:return: output of network
		"""
		with torch.no_grad():
			out = self.resnet_model(x)
		return out

	@staticmethod
	def regression_loss(y: torch.tensor,
						y_hat: torch.tensor
						) -> torch.tensor:
		"""
		Computes mse loss between ground truth (x,y) coordinates and predicted (x^,y^) coordinates
		:param y: ground truth. shape=(num_targets, 2)
		:param y_hat: prediction. shape=(num_targets, 2)
		:return: mse loss
		"""
		# apply mask
		y_mask = torch.where(torch.isnan(y), 0.0, y)
		y_hat_mask = torch.where(torch.isnan(y), 0.0, y_hat)

		# compute loss
		loss = F.mse_loss(y_hat_mask, y_mask)

		return loss

	def training_step(self, data, batch_idx):
		x, y = data
		# forward pass
		y_hat = self.resnet_model(x)
		# compute loss
		loss = self.regression_loss(y, y_hat)
		# log training loss
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {'loss': loss}

	def validation_step(self, data, batch_idx):
		x, y = data
		y_hat = self.forward(x)
		# compute loss
		loss = self.regression_loss(y, y_hat)
		# log validation loss
		self.log('val_loss', loss, prog_bar=True, logger=True)

	def test_step(self, data, batch_idx):
		self.validation_step(data, batch_idx)

	def configure_optimizers(self):
		return Adam(self.parameters(), lr=1e-3)