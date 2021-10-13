import torch
import pandas as pd
from torch import cuda
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from typing import Callable, Optional, Tuple, List
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from pose_est_nets.utils.heatmap_tracker_utils import format_mouse_data
from pose_est_nets.utils.dataset_utils import draw_keypoints
import h5py
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typeguard import typechecked
import sklearn

# set the random seed as input here.
# TODO: when moving to runs, we would like different random seeds, so consider eliminating.
# TODO: review the transforms -- resize is done by imgaug.augmenters coming from the main script. it is fed as input. internally, we always normalize to imagenet params.
TORCH_MANUAL_SEED = 42
_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# statistics of imagenet dataset on which the resnet was trained
# see https://pytorch.org/vision/stable/models.html
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
num_processes = os.cpu_count()


class BaseTrackingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        imgaug_transform: Optional[Callable] = None,
        pytorch_transform_list: Optional[List] = None,
    ) -> None:
        """
        Initializes the Regression Dataset
        Parameters:
            root_directory (str): path to data directory
            csv_path (str): path to CSV file (within root_directory). CSV file should be
                in the form (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows (List[int]): (optional) which rows in the csv are header rows
            transform (torchvision.transforms): (optional) transform to apply to images
        Returns:
            None
        """
        self.root_directory = root_directory
        self.imgaug_transform = imgaug_transform
        csv_data = pd.read_csv(
            os.path.join(root_directory, csv_path), header=header_rows
        )
        self.image_names = list(csv_data.iloc[:, 0])
        self.labels = torch.tensor(csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
        self.labels = self.labels.reshape(
            self.labels.shape[0], -1, 2
        )  # converted to x,y coordinates
        if pytorch_transform_list is None:
            pytorch_transform_list = []  # make the None an empty list
        pytorch_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]

        self.pytorch_transform = transforms.Compose(pytorch_transform_list)
        self.num_targets = (
            self.labels.shape[1] * 2
        )  # labels has been already transformed above
        self.num_keypoints = self.labels.shape[1]

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
        image = Image.open(os.path.join(self.root_directory, img_name)).convert(
            "RGB"
        )  # Rick's images have 1 color channel; change to 3.

        keypoints_on_image = self.labels[
            idx
        ]  # get current image labels from self.labels
        if self.imgaug_transform is not None:
            transformed_images, transformed_keypoints = self.imgaug_transform(
                images=np.expand_dims(image, axis=0),  # add batch dim
                keypoints=np.expand_dims(keypoints_on_image, axis=0),  # add batch dim
            )
            # get rid of the batch dim
            transformed_images = transformed_images.squeeze(0)
            transformed_keypoints = transformed_keypoints.squeeze(0)
            # TODO: the problem below is that it messes up datasets.compute_heatmaps()
            transformed_keypoints = transformed_keypoints.reshape(
                transformed_keypoints.shape[0] * transformed_keypoints.shape[1]
            )
        transformed_images = self.pytorch_transform(transformed_images)
        assert transformed_keypoints.shape == (self.num_targets,)

        # ret = (transformed_images, torch.from_numpy(transformed_keypoints))
        return transformed_images, torch.from_numpy(transformed_keypoints)


# the only addition here, should be the heatmap creation method.
class HeatmapDataset(BaseTrackingDataset):
    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        imgaug_transform: Optional[Callable] = None,
        pytorch_transform_list: Optional[List] = None,
        noNans: Optional[bool] = False,
        downsample_factor: Optional[int] = 2,
    ) -> None:
        """
        Initializes the DLC Heatmap Dataset
        Parameters:
            root_directory (str): path to data directory
            data_path (str): path to CSV or h5 file  (within root_directory). CSV file should be
                in the form (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows (List[int]): (optional) which rows in the csv are header rows
            transform (torchvision.transforms): (optional) transform to resize the images, image dimensions must be repeatably divisible by 2
            noNans (bool): whether or not to throw out all frames that have occluded keypoints
        Returns:
            None
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
                "image dimensions (after transformation) must be repeatably divisible by 2!"
            )
            print("current image dimensions after transformation are:")
            exit()

        if noNans:
            # Checks for images with set of keypoints that include any nan, so that they can be excluded from the data entirely, like DeepPoseKit does
            self.fully_labeled_idxs = self.get_fully_labeled_idxs()
            self.image_names = [
                self.image_names[idx] for idx in self.fully_labeled_idxs
            ]
            self.labels = torch.index_select(self.labels, 0, self.fully_labeled_idxs)
            self.labels = torch.tensor(self.labels)

        self.downsample_factor = downsample_factor
        # self.sigma = 5
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor

        # Compute heatmaps as preprocessing step
        # check that max of heatmaps look good
        self.num_targets = torch.numel(self.labels[0])
        self.num_keypoints = self.num_targets // 2
        self.compute_heatmaps()

    @property
    def output_shape(self):
        return (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )

    def compute_heatmaps(self):
        """note: original image dims e.g., (406, 396) -> resized image dims e.g., (384, 384) -> potentially downsampled heatmaps e.g., (96, 96)"""
        label_heatmaps = []
        for idx in range(len(self.image_names)):
            x, y = super().__getitem__(idx)
            y_heatmap = draw_keypoints(
                y.numpy().reshape(
                    self.num_keypoints, 2
                ),  # Note: super().__getitem__ returns flat labels, we reshape to (num_keypoints,2)
                x.shape[-2],
                x.shape[-1],
                self.output_shape,
                sigma=self.output_sigma,
            )
            assert y_heatmap.shape == (*self.output_shape, self.num_keypoints)
            label_heatmaps.append(y_heatmap)

        self.label_heatmaps = torch.from_numpy(np.asarray(label_heatmaps)).float()
        self.label_heatmaps = self.label_heatmaps.permute(0, 3, 1, 2)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """we call the base dataset to get an image and a label.
        we additionaly return the corresponding heatmap."""
        image, labels = super().__getitem__(
            idx
        )  # could modify this if speed bottleneck
        heatmaps = self.label_heatmaps[idx]
        return image, heatmaps, labels

    def get_fully_labeled_idxs(self):  # TODO: make shorter
        nan_check = torch.isnan(self.labels)
        nan_check = nan_check[:, :, 0]
        nan_check = ~nan_check
        annotated = torch.all(nan_check, dim=1)
        annotated_index = torch.where(annotated)
        return annotated_index[0]
