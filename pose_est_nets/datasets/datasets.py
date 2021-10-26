"""Dataset objects store images, labels, and functions for manipulation."""

import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from typing import Callable, Literal, List, Optional, Tuple, Union

from pose_est_nets.utils.dataset_utils import draw_keypoints

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# statistics of imagenet dataset on which the resnet was trained
# see https://pytorch.org/vision/stable/models.html
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# TODO: review transforms -- resize is done by imgaug.augmenters coming from
# TODO: the main script. it is fed as input. internally, we always normalize to
# TODO: imagenet params.


class BaseTrackingDataset(torch.utils.data.Dataset):
    """Base dataset that contains images and keypoints as (x, y) pairs."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        imgaug_transform: Optional[Callable] = None,
        pytorch_transform_list: Optional[List] = None,
    ) -> None:
        """Initialize the Regression Dataset.

        The csv file of labels will be searched for in the following order:
        1. assume csv is located at `root_directory/csv_path` (i.e. `csv_path`
            argument is a path relative to `root_directory`)
        2. if not found, assume `csv_path` is absolute. Note the image paths
            within the csv must still be relative to `root_directory`
        3. if not found, assume dlc directory structure:
           `root_directory/training-datasets/iteration-0/csv_path` (`csv_path`
           argument will look like "CollectedData_<scorer>.csv")

        Args:
            root_directory: path to data directory
            csv_path: path to CSV file (within root_directory). CSV file should
                be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows: which rows in the csv are header rows
            transform: torchvision transform to apply to images

        """
        self.root_directory = root_directory
        self.imgaug_transform = imgaug_transform

        # load csv data
        # step 1
        csv_file = os.path.join(root_directory, csv_path)
        if not os.path.exists(csv_file):
            # step 2: assume csv_path is absolute
            csv_file = csv_path
            if not os.path.exists(csv_file):
                # step 3: assume dlc directory structure
                import glob
                glob_path = os.path.join(
                        root_directory,
                        "training-datasets",
                        "iteration-0",
                        "*",  # wildcard handles proj-specific dlc naming conventions
                        csv_path,
                    )
                options = glob.glob(glob_path)
                if not options or not os.path.exists(options[0]):
                    raise FileNotFoundError("Could not find csv file!")
                csv_file = options[0]

        csv_data = pd.read_csv(csv_file, header=header_rows)
        self.image_names = list(csv_data.iloc[:, 0])
        self.keypoints = torch.tensor(
            csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32
        )
        # convert to x,y coordinates
        self.keypoints = self.keypoints.reshape(self.keypoints.shape[0], -1, 2)
        if pytorch_transform_list is None:
            pytorch_transform_list = []  # make the None an empty list
        pytorch_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]

        self.pytorch_transform = transforms.Compose(pytorch_transform_list)
        # keypoints has been already transformed above
        self.num_targets = self.keypoints.shape[1] * 2
        self.num_keypoints = self.keypoints.shape[1]

    @property
    def height(self):
        return self.imgaug_transform[0].get_parameters()[0][0].value
        # Assuming resizing transformation is the first imgaug one

    @property
    def width(self):
        return self.imgaug_transform[0].get_parameters()[0][1].value
        # Assuming resizing transformation is the first imgaug one

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        # get img_name from self.image_names
        img_name = self.image_names[idx]
        # read image from file and apply transformations (if any)
        file_name = os.path.join(self.root_directory, img_name)
        # if 1 color channel, change to 3.
        image = Image.open(file_name).convert("RGB")

        # get current image keypoints from self.keypoints
        keypoints_on_image = self.keypoints[idx]
        if self.imgaug_transform is not None:
            transformed_images, transformed_keypoints = self.imgaug_transform(
                images=np.expand_dims(image, axis=0),
                keypoints=np.expand_dims(keypoints_on_image, axis=0),
            )  # expands add batch dim for imgaug
            # get rid of the batch dim
            transformed_images = transformed_images.squeeze(0)
            transformed_keypoints = transformed_keypoints.squeeze(0)
            # TODO: the problem below is that it messes up
            # TODO: datasets.compute_heatmaps()
            transformed_keypoints = transformed_keypoints.reshape(
                transformed_keypoints.shape[0] * transformed_keypoints.shape[1]
            )
        transformed_images = self.pytorch_transform(transformed_images)
        assert transformed_keypoints.shape == (self.num_targets,)

        # ret = (transformed_images, torch.from_numpy(transformed_keypoints))
        return transformed_images, torch.from_numpy(transformed_keypoints)


# the only addition here, should be the heatmap creation method.
class HeatmapDataset(BaseTrackingDataset):
    """Heatmap dataset that contains the images and keypoints in 2D arrays."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        imgaug_transform: Optional[Callable] = None,
        pytorch_transform_list: Optional[List] = None,
        no_nans: bool = False,
        downsample_factor: int = 2,
    ) -> None:
        """Initialize the Heatmap Dataset.

        Args:
            root_directory: path to data directory
            cav_path: path to CSV or h5 file  (within root_directory). CSV file
                should be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows: which rows in the csv are header rows
            transform: torchvision transform to resize the images, image
                dimensions must be repeatably divisible by 2
            no_nans: whether or not to throw out all frames that have occluded
                keypoints

        """
        super().__init__(
            root_directory,
            csv_path,
            header_rows,
            imgaug_transform,
            pytorch_transform_list,
        )

        if self.height % 128 != 0 or self.height % 128 != 0:
            print(
                "image dimensions (after transformation) must be repeatably "
                + "divisible by 2!"
            )
            print("current image dimensions after transformation are:")
            exit()

        if no_nans:
            # Checks for images with set of keypoints that include any nan, so
            # that they can be excluded from the data entirely, like DPK does
            self.fully_labeled_idxs = self.get_fully_labeled_idxs()
            self.image_names = [
                self.image_names[idx] for idx in self.fully_labeled_idxs
            ]
            self.keypoints = torch.index_select(
                self.keypoints, 0, self.fully_labeled_idxs
            )
            self.keypoints = torch.tensor(self.keypoints)

        self.downsample_factor = downsample_factor
        # self.sigma = 5
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor

        # Compute heatmaps as preprocessing step
        # check that max of heatmaps look good
        self.num_targets = torch.numel(self.keypoints[0])
        self.num_keypoints = self.num_targets // 2
        self.label_heatmaps = None  # populated by `compute_heatmaps()`
        self.compute_heatmaps()

    @property
    def output_shape(self):
        return (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )

    def compute_heatmaps(self):
        """Compute 2D heatmaps from (x, y) coordinates

        original image dims e.g., (406, 396) ->
        resized image dims e.g., (384, 384) ->
        potentially downsampled heatmaps e.g., (96, 96)

        """
        label_heatmaps = []
        for idx in range(len(self.image_names)):
            x, y = super().__getitem__(idx)
            # super().__getitem__ returns flat keypoints, reshape to
            # (num_keypoints, 2)
            y_heatmap = draw_keypoints(
                y.numpy().reshape(self.num_keypoints, 2),
                x.shape[-2],
                x.shape[-1],
                self.output_shape,
                sigma=self.output_sigma,
            )
            assert y_heatmap.shape == (*self.output_shape, self.num_keypoints)
            label_heatmaps.append(y_heatmap)

        self.label_heatmaps = torch.from_numpy(np.asarray(label_heatmaps)).float()
        self.label_heatmaps = self.label_heatmaps.permute(0, 3, 1, 2)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, torch.Tensor]:
        """Get batch of data.

        Calls the base dataset to get an image and a label, then additionaly
        return the corresponding heatmap.

        """
        # could modify this if speed bottleneck
        image, keypoints = super().__getitem__(idx)
        heatmaps = self.label_heatmaps[idx]
        return image, heatmaps, keypoints

    def get_fully_labeled_idxs(self):  # TODO: make shorter
        nan_check = torch.isnan(self.keypoints)
        nan_check = nan_check[:, :, 0]
        nan_check = ~nan_check
        annotated = torch.all(nan_check, dim=1)
        annotated_index = torch.where(annotated)
        return annotated_index[0]
