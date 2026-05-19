"""Data loading, preprocessing, and augmentation for pose estimation."""

# statistics of imagenet dataset on which the resnet was trained
# see https://pytorch.org/vision/stable/models.html
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

from lightning_pose.data.factory import (  # noqa: E402
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)
