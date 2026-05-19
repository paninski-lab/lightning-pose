"""Public API for loading trained models and running predictions."""

from lightning_pose.api.model import Model
from lightning_pose.api.model_config import ModelConfig

__all__ = ["Model", "ModelConfig"]
