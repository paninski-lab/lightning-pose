"""Dataset objects store images, labels, and functions for manipulation."""

import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from typing import Callable, List, Literal, Optional, Tuple, TypedDict, Union
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.utils import generate_heatmaps
from lightning_pose.utils.io import get_keypoint_names

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

patch_typeguard()  # use before @typechecked


class BaseExampleDict(TypedDict):
    """Class for finer control over typechecking."""

    images: Union[
        TensorType["RGB":3, "image_height", "image_width"],
        TensorType["frames", "RGB":3, "image_height", "image_width"],
    ]
    keypoints: TensorType["num_targets"]
    idxs: int


class HeatmapExampleDict(BaseExampleDict):
    """Class for finer control over typechecking."""

    heatmaps: TensorType["num_keypoints", "heatmap_height", "heatmap_width"]


class BaseTrackingDataset(torch.utils.data.Dataset):
    """Base dataset that contains images and keypoints as (x, y) pairs."""

    @typechecked
    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        imgaug_transform: Optional[Callable] = None,
        pytorch_transform_list: Optional[List] = None,
        do_context: bool = True,
    ) -> None:
        """Initialize a dataset for regression (rather than heatmap) models.

        The csv file of labels will be searched for in the following order:
        1. assume csv is located at `root_directory/csv_path` (i.e. `csv_path`
            argument is a path relative to `root_directory`)
        2. if not found, assume `csv_path` is absolute. Note the image paths
            within the csv must still be relative to `root_directory`
        3. if not found, assume dlc directory structure:
           `root_directory/training-data/iteration-0/csv_path` (`csv_path`
           argument will look like "CollectedData_<scorer>.csv")

        Args:
            root_directory: path to data directory
            csv_path: path to CSV file (within root_directory). CSV file should
                be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows: which rows in the csv are header rows
            transform: torchvision transform to apply to images
            do_context: include additional frames of context if possible.

        """
        self.root_directory = root_directory
        self.imgaug_transform = imgaug_transform
        self.do_context = do_context

        # load csv data
        # step 1
        if os.path.isfile(csv_path):
            csv_file = csv_path
        else:
            csv_file = os.path.join(root_directory, csv_path)
        if not os.path.exists(csv_file):
            # step 2: assume csv_path is absolute
            csv_file = csv_path
            if not os.path.exists(csv_file):
                # step 3: assume dlc directory structure
                import glob

                glob_path = os.path.join(
                    root_directory,
                    "training-data",
                    "iteration-0",
                    "*",  # wildcard handles proj-specific dlc naming conventions
                    csv_path,
                )
                options = glob.glob(glob_path)
                if not options or not os.path.exists(options[0]):
                    raise FileNotFoundError("Could not find csv file!")
                csv_file = options[0]

        csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
        self.keypoint_names = get_keypoint_names(
            csv_file=csv_file, header_rows=header_rows)
        if header_rows == [1, 2] or header_rows == [0, 1]:
            # self.keypoint_names = csv_data.columns.levels[0]
            # ^this returns a sorted list for some reason, don't want that
            self.keypoint_names = [b[0] for b in csv_data.columns if b[1] == 'x']
        elif header_rows == [0, 1, 2]:
            # self.keypoint_names = csv_data.columns.levels[1]
            self.keypoint_names = [b[1] for b in csv_data.columns if b[2] == 'x']

        self.image_names = list(csv_data.index)
        self.keypoints = torch.tensor(csv_data.to_numpy(), dtype=torch.float32)
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
        # assume resizing transformation is the last imgaug one
        return self.imgaug_transform[-1].get_parameters()[0][0].value

    @property
    def width(self):
        # assume resizing transformation is the last imgaug one
        return self.imgaug_transform[-1].get_parameters()[0][1].value

    def __len__(self) -> int:
        return len(self.image_names)

    @typechecked
    def __getitem__(self, idx: int):

        img_name = self.image_names[idx]

        if not self.do_context:
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
                transformed_images = transformed_images[0]
                transformed_keypoints = transformed_keypoints[0]
                # TODO: the problem below is that it messes up
                # TODO: data.compute_heatmaps()
                transformed_keypoints = transformed_keypoints.reshape(
                    transformed_keypoints.shape[0] * transformed_keypoints.shape[1]
                )
            else:
                transformed_images = np.expand_dims(image, axis=0)
                transformed_keypoints = np.expand_dims(keypoints_on_image, axis=0)

            transformed_images = self.pytorch_transform(
               transformed_images
            )
            assert transformed_keypoints.shape == (self.num_targets,)

        else:
            # get index of the image
            idx_img = img_name.split("/")[-1].replace("img", "")
            idx_img = int(idx_img.replace(".png", ""))

            # get the frames -> t-2, t-1, t, t+1, t + 2
            list_idx = [idx_img - 2, idx_img - 1, idx_img, idx_img + 1, idx_img + 2]
            list_img_names = []

            for fr_num in list_idx:
                # replace frame number with 0 if we're at the beginning of the video
                fr_num = max(0, fr_num)
                # split name into pieces
                img_pieces = img_name.split("/")
                # figure out length of integer
                int_len = len(img_pieces[-1].replace(".png", "").replace("img", ""))
                # replace original frame number with context frame number
                img_pieces[-1] = "img%s.png" % str(fr_num).zfill(int_len)
                list_img_names.append("/".join(img_pieces))

            # read the images from image list to create dataset
            keypoints_on_image = self.keypoints[idx]
            images = []
            for name in list_img_names:
                # read image from file and apply transformations (if any)
                file_name = os.path.join(self.root_directory, name)
                # current renaming scheme loses a leading zero when going down an order
                # of magnitude, i.e. 1001 - 2 -> 999 instead of 0999
                # if not os.path.isfile(file_name):
                #     file_name = file_name.replace("img", "img0")
                #     # handle case where we go up an order of magnitude, i.e.
                #     # 009 -> 0010 instead of 010
                #     if not os.path.isfile(file_name):
                #         # take away leading zero added above, as well as leading zero no
                #         # longer needed since we're moving up an order of magnitude
                #         file_name = file_name.replace("img00", "img")
                # if 1 color channel, change to 3.
                image = Image.open(file_name).convert("RGB")
                images.append(np.asarray(image))

            keypoints_on_image = torch.unsqueeze(keypoints_on_image, 0)
            keypoints_on_image = list(keypoints_on_image.tile((5, 1, 1)).numpy())

            if self.imgaug_transform is not None:
                transformed_images, transformed_keypoints = self.imgaug_transform(
                    images=images,
                    keypoints=keypoints_on_image,
                )  # expands add batch dim for imgaug
                # get rid of the batch dim
                transformed_images = np.asarray(transformed_images)
                transformed_keypoints = transformed_keypoints[0]
                # TODO: the problem below is that it messes up
                # TODO: data.compute_heatmaps()
                transformed_keypoints = transformed_keypoints.reshape(
                    transformed_keypoints.shape[0] * transformed_keypoints.shape[1]
                )
            else:
                transformed_images = np.asarray(images)
                transformed_keypoints = keypoints_on_image[0]

            for i, transformed_image in enumerate(transformed_images):
                transformed_image = self.pytorch_transform(transformed_image)
                if i == 0:
                    image_frames_tensor = torch.unsqueeze(transformed_image, dim=0)
                else:
                    image_expand = torch.unsqueeze(transformed_image, dim=0)
                    image_frames_tensor = torch.cat(
                        (image_frames_tensor, image_expand), dim=0
                    )

            transformed_images = image_frames_tensor
            assert transformed_keypoints.shape == (self.num_targets,)
        return {
            "images": transformed_images,
            "keypoints": torch.from_numpy(transformed_keypoints),
            "idxs": idx,
        }


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
        do_context: bool = True,
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
            do_context,
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
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    def compute_heatmaps(self):
        """Compute 2D heatmaps from (x, y) coordinates.

        original image dims e.g., (406, 396) ->
        resized image dims e.g., (384, 384) ->
        potentially downsampled heatmaps e.g., (96, 96)

        """

        label_heatmaps = torch.empty(
            size=(len(self.image_names), self.num_keypoints, *self.output_shape)
        )
        for idx in range(len(self.image_names)):
            example_dict: BaseExampleDict = super().__getitem__(idx)
            # super().__getitem__ returns flat keypoints, reshape to
            if self.do_context:
                image_height = example_dict["images"][0].shape[-2]
                image_width = example_dict["images"][0].shape[-1]
            else:
                image_height = example_dict["images"].shape[-2]
                image_width = example_dict["images"].shape[-1]

            y_heatmap = generate_heatmaps(
                example_dict["keypoints"].reshape(
                    1, self.num_keypoints, 2
                ),  # add batch dim
                image_height,
                image_width,
                output_shape=self.output_shape,
                sigma=self.output_sigma,
            )
            assert y_heatmap.shape == (1, self.num_keypoints, *self.output_shape)
            label_heatmaps[idx] = y_heatmap[0]

        self.label_heatmaps = label_heatmaps

    @typechecked
    def __getitem__(self, idx: int) -> HeatmapExampleDict:
        """Get an example from the dataset.

        Calls the base dataset to get an image and a label, then additionally
        returns the corresponding heatmap.

        """
        example_dict: BaseExampleDict = super().__getitem__(idx)
        example_dict["heatmaps"] = self.label_heatmaps[idx]
        return example_dict

    def get_fully_labeled_idxs(self):  # TODO: make shorter
        nan_check = torch.isnan(self.keypoints)
        nan_check = nan_check[:, :, 0]
        nan_check = ~nan_check
        annotated = torch.all(nan_check, dim=1)
        annotated_index = torch.where(annotated)
        return annotated_index[0]
