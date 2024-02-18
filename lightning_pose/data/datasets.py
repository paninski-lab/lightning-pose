"""Dataset objects store images, labels, and functions for manipulation."""

import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchtyping import TensorType
from torchvision import transforms

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.utils import (
    BaseLabeledExampleDict,
    HeatmapLabeledExampleDict,
    MultiviewHeatmapLabeledExampleDict,
    generate_heatmaps,
)
from lightning_pose.utils.io import get_keypoint_names

# to ignore imports for sphix-autoapidoc
__all__ = [
    "BaseTrackingDataset",
    "HeatmapDataset",
]

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseTrackingDataset(torch.utils.data.Dataset):
    """Base dataset that contains images and keypoints as (x, y) pairs."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = [0, 1, 2],
        imgaug_transform: Optional[Callable] = None,
        do_context: bool = False,
        delimiter: str = "img"
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
            csv_path: path to CSV file (within root_directory). CSV file should be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            do_context: include additional frames of context if possible.

        """
        self.root_directory = root_directory
        self.csv_path = csv_path
        self.header_rows = header_rows
        self.imgaug_transform = imgaug_transform
        self.do_context = do_context
        self.delimiter = delimiter

        # load csv data
        if os.path.isfile(csv_path):
            csv_file = csv_path
        else:
            csv_file = os.path.join(root_directory, csv_path)
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Could not find csv file at {csv_file}!")

        csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
        self.keypoint_names = get_keypoint_names(csv_file=csv_file, header_rows=header_rows)
        self.image_names = list(csv_data.index)
        self.keypoints = torch.tensor(csv_data.to_numpy(), dtype=torch.float32)
        # convert to x,y coordinates
        self.keypoints = self.keypoints.reshape(self.keypoints.shape[0], -1, 2)

        # send image to tensor and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

        # keypoints has been already transformed above
        self.num_targets = self.keypoints.shape[1] * 2
        self.num_keypoints = self.keypoints.shape[1]

        self.data_length = len(self.image_names)

    @property
    def height(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.imgaug_transform[-1].get_parameters()[0][0].value

    @property
    def width(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.imgaug_transform[-1].get_parameters()[0][1].value

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> BaseLabeledExampleDict:
        img_name = self.image_names[idx]
        keypoints_on_image = self.keypoints[idx]

        if not self.do_context:
            # read image from file and apply transformations (if any)
            file_name = os.path.join(self.root_directory, img_name)
            # if 1 color channel, change to 3.
            image = Image.open(file_name).convert("RGB")
            if self.imgaug_transform is not None:
                transformed_images, transformed_keypoints = self.imgaug_transform(
                    images=np.expand_dims(image, axis=0),
                    keypoints=np.expand_dims(keypoints_on_image, axis=0),
                )  # expands add batch dim for imgaug
                # get rid of the batch dim
                transformed_images = transformed_images[0]
                transformed_keypoints = transformed_keypoints[0].reshape(-1)
            else:
                transformed_images = np.expand_dims(image, axis=0)
                transformed_keypoints = np.expand_dims(keypoints_on_image, axis=0)

            transformed_images = self.pytorch_transform(transformed_images)

        else:
            # get index of the image
            # idx_img = img_name.split("/")[-1].replace("img", "")
            idx_img_basename = os.path.basename(img_name)
            idx_img_basename_delimated = idx_img_basename.split(self.delimiter)[-1]
            # image_format = idx_img.split('.')[-1]
            idx_img_str = idx_img_basename_delimated.split('.')[0]
            # figure out length of integer
            idx_img = int(idx_img_str)
            int_len = len(idx_img_str)
            # get the frames -> t-2, t-1, t, t+1, t + 2
            list_idx = [idx_img - 2, idx_img - 1, idx_img, idx_img + 1, idx_img + 2]
            list_img_names = []
            for fr_num in list_idx:
                # replace frame number with 0 if we're at the beginning of the video
                fr_num = max(0, fr_num)
                # split name into pieces
                img_pieces = img_name.split("/")
                # replace original frame number with context frame number
                fr_num = str(fr_num)
                if len(fr_num) > int_len:
                    fr_num = fr_num.zfill(int_len + 1)
                else:
                    fr_num = fr_num.zfill(int_len)
                img_pieces[-1] = img_pieces[-1].replace(idx_img_str, fr_num)
                list_img_names.append("/".join(img_pieces))
            # read the images from image list to create dataset
            images = []
            for name in list_img_names:
                # read image from file and apply transformations (if any)
                file_name = os.path.join(self.root_directory, name)
                # if 1 color channel, change to 3.
                image = Image.open(file_name).convert("RGB")
                images.append(np.asarray(image))

            # apply data aug pipeline
            if self.imgaug_transform is not None:
                # need to apply the same transform to all context frames
                seed = np.random.randint(low=0, high=123456)
                transformed_images = []
                for img in images:
                    self.imgaug_transform.seed_(seed)
                    transformed_image, transformed_keypoints = self.imgaug_transform(
                        images=[img], keypoints=[keypoints_on_image.numpy()]
                    )
                    transformed_images.append(transformed_image[0])
                transformed_images = np.asarray(transformed_images)
                transformed_keypoints = transformed_keypoints[0].reshape(-1)
            else:
                transformed_images = np.asarray(images)
                transformed_keypoints = keypoints_on_image.numpy().reshape(-1)

            # send frames to tensors and normalize
            # need to loop through because ToTensor transform only operates on single images
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

        return BaseLabeledExampleDict(
            images=transformed_images,  # shape (3, img_height, img_width) or (5, 3, H, W)
            keypoints=torch.from_numpy(transformed_keypoints),  # shape (n_targets,)
            idxs=idx,
            bbox=torch.tensor([0, 0, image.height, image.width])  # x,y,h,w of bounding box
        )


# the only addition here, should be the heatmap creation method.
class HeatmapDataset(BaseTrackingDataset):
    """Heatmap dataset that contains the images and keypoints in 2D arrays."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = [0, 1, 2],
        imgaug_transform: Optional[Callable] = None,
        downsample_factor: Literal[1, 2, 3] = 2,
        do_context: bool = False,
        uniform_heatmaps: bool = False,
        delimiter: str = "img"
    ) -> None:
        """Initialize the Heatmap Dataset.

        Args:
            root_directory: path to data directory
            csv_path: path to CSV or h5 file  (within root_directory). CSV file
                should be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            downsample_factor: factor by which to downsample original image dims to have a smaller
                heatmap
            do_context: include additional frames of context if possible

        """
        super().__init__(
            root_directory=root_directory,
            csv_path=csv_path,
            header_rows=header_rows,
            imgaug_transform=imgaug_transform,
            do_context=do_context,
            delimiter=delimiter
        )

        if self.height % 128 != 0 or self.height % 128 != 0:
            print(
                "image dimensions (after transformation) must be repeatably "
                + "divisible by 2!"
            )
            print("current image dimensions after transformation are:")
            exit()

        self.downsample_factor = downsample_factor
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor
        self.uniform_heatmaps = uniform_heatmaps

        # Compute heatmaps as preprocessing step
        self.num_targets = torch.numel(self.keypoints[0])
        self.num_keypoints = self.num_targets // 2

    @property
    def output_shape(self) -> tuple:
        return (
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    def compute_heatmap(
        self, example_dict: BaseLabeledExampleDict
    ) -> TensorType["num_keypoints", "heatmap_height", "heatmap_width"]:
        """Compute 2D heatmaps from arbitrary (x, y) coordinates."""

        # reshape
        keypoints = example_dict["keypoints"].reshape(self.num_keypoints, 2)

        # introduce new nans where data augmentation has moved the keypoint out of the original
        # frame
        new_nans = torch.logical_or(
            torch.lt(keypoints[:, 0], torch.tensor(0)),
            torch.lt(keypoints[:, 1], torch.tensor(0)),
        )
        new_nans = torch.logical_or(
            new_nans, torch.ge(keypoints[:, 0], torch.tensor(self.width))
        )
        new_nans = torch.logical_or(
            new_nans, torch.ge(keypoints[:, 1], torch.tensor(self.height))
        )
        keypoints[new_nans, :] = torch.nan

        y_heatmap = generate_heatmaps(
            keypoints=keypoints.unsqueeze(0),  # add batch dim
            height=self.height,
            width=self.width,
            output_shape=self.output_shape,
            sigma=self.output_sigma,
            uniform_heatmaps=self.uniform_heatmaps,
        )

        return y_heatmap[0]

    def compute_heatmaps(self):
        """Compute initial 2D heatmaps for all labeled data.

        original image dims e.g., (406, 396) ->
        resized image dims e.g., (384, 384) ->
        potentially downsampled heatmaps e.g., (96, 96)

        """
        label_heatmaps = torch.empty(
            size=(len(self.image_names), self.num_keypoints, *self.output_shape)
        )
        for idx in range(len(self.image_names)):
            example_dict: BaseLabeledExampleDict = super().__getitem__(idx)
            label_heatmaps[idx] = self.compute_heatmap(example_dict)

        self.label_heatmaps = label_heatmaps

    def __getitem__(self, idx: int) -> HeatmapLabeledExampleDict:
        """Get an example from the dataset.

        Calls the base dataset to get an image and a label, then additionally
        returns the corresponding heatmap.

        """
        example_dict: BaseLabeledExampleDict = super().__getitem__(idx)
        example_dict["heatmaps"] = self.compute_heatmap(example_dict)
        return example_dict


class DetectorDataset(BaseTrackingDataset):
    """Heatmap dataset that contains the images and keypoints in 2D arrays."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        resized_dims: Tuple,
        keypoints_for_crop: List,
        header_rows: Optional[List[int]] = [0, 1, 2],
        imgaug_transform: Optional[Callable] = None,
        downsample_factor: Literal[1, 2, 3] = 2,
        do_context: bool = False,
        uniform_heatmaps: bool = False,
    ) -> None:
        """Initialize the Heatmap Dataset.

        Args:
            root_directory: path to data directory
            csv_path: path to CSV or h5 file  (within root_directory). CSV file
                should be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            downsample_factor: factor by which to downsample original image dims to have a smaller
                heatmap
            do_context: include additional frames of context if possible

        """
        super().__init__(
            root_directory=root_directory,
            csv_path=csv_path,
            header_rows=header_rows,
            imgaug_transform=imgaug_transform,
            do_context=do_context,
        )
        self.resized_dims = resized_dims
        self.downsample_factor = downsample_factor
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor
        self.uniform_heatmaps = uniform_heatmaps
        self.keypoints_for_crop = keypoints_for_crop

        # add a resize operation to end of image augmentation pipeline
        self.imgaug_transform.append(iaa.Resize({
            "height": self.height, "width": self.width
        }))

        # average finescale keypoints together to get coarsescale keypoints
        coarse_kpts = torch.zeros((self.keypoints.shape[0], len(keypoints_for_crop), 2))
        for i, k in enumerate(keypoints_for_crop):
            coarse_kpts[:, i, :] = self.keypoints[:, k, :].mean(dim=1)
        self.keypoints = coarse_kpts

        if self.height % 128 != 0 or self.height % 128 != 0:
            print(
                "image dimensions (after transformation) must be repeatably "
                + "divisible by 2!"
            )
            print("current image dimensions after transformation are:")
            exit()

        # Compute heatmaps as preprocessing step
        self.num_targets = torch.numel(self.keypoints[0])
        self.num_keypoints = self.num_targets // 2
        # self.label_heatmaps = None  # populated by `self.compute_heatmaps()`
        # print('Computing Heatmaps')
        # self.compute_heatmaps()

    @property
    def height(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.resized_dims[1]

    @property
    def width(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.resized_dims[0]

    @property
    def output_shape(self) -> tuple:
        return (
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    def compute_heatmap(
            self, example_dict: BaseLabeledExampleDict
    ) -> TensorType["num_keypoints", "heatmap_height", "heatmap_width"]:
        """Compute 2D heatmaps from arbitrary (x, y) coordinates."""

        # reshape
        keypoints = example_dict["keypoints"].reshape(self.num_keypoints, 2)

        # introduce new nans where data augmentation has moved the keypoint out of the original
        # frame
        new_nans = torch.logical_or(
            torch.lt(keypoints[:, 0], torch.tensor(0)),
            torch.lt(keypoints[:, 1], torch.tensor(0)),
        )
        new_nans = torch.logical_or(
            new_nans, torch.ge(keypoints[:, 0], torch.tensor(self.width))
        )
        new_nans = torch.logical_or(
            new_nans, torch.ge(keypoints[:, 1], torch.tensor(self.height))
        )
        keypoints[new_nans, :] = torch.nan

        y_heatmap = generate_heatmaps(
            keypoints=keypoints.unsqueeze(0),  # add batch dim
            height=self.height,
            width=self.width,
            output_shape=self.output_shape,
            sigma=self.output_sigma,
            uniform_heatmaps=self.uniform_heatmaps,
        )

        return y_heatmap[0]

    def __getitem__(self, idx: int) -> HeatmapLabeledExampleDict:
        img_name = self.image_names[idx]
        keypoints_on_image = self.keypoints[idx]

        # read image from file and apply transformations (if any)
        file_name = os.path.join(self.root_directory, img_name)
        # if 1 color channel, change to 3.
        image = Image.open(file_name).convert("RGB")
        if self.imgaug_transform is not None:
            transformed_images, transformed_keypoints = self.imgaug_transform(
                images=np.expand_dims(image, axis=0),
                keypoints=np.expand_dims(keypoints_on_image, axis=0),
            )  # expands add batch dim for imgaug
            # get rid of the batch dim
            transformed_images = transformed_images[0]
            transformed_keypoints = transformed_keypoints[0].reshape(-1)
        else:
            transformed_images = np.expand_dims(image, axis=0)
            transformed_keypoints = np.expand_dims(keypoints_on_image, axis=0)

        transformed_images = self.pytorch_transform(transformed_images)
        assert transformed_keypoints.shape == (self.num_targets,)
        example_dict = BaseLabeledExampleDict(
            images=transformed_images,  # shape (3, img_height, img_width)
            keypoints=torch.from_numpy(transformed_keypoints),  # shape (n_targets,)
            idxs=idx,
            bbox=torch.tensor([0, 0, image.height, image.width])  # x,y,h,w of bounding box
        )
        example_dict["heatmaps"] = self.compute_heatmap(example_dict)
        return example_dict


class MultiviewHeatmapDataset(torch.utils.data.Dataset):
    """Heatmap dataset that contains the images and keypoints in 2D arrays from all the cameras."""

    def __init__(
        self,
        root_directory: str,
        csv_paths: List[str],
        view_names: List[str],
        header_rows: Optional[List[int]] = [0, 1, 2],
        downsample_factor: Literal[1, 2, 3] = 2,
        uniform_heatmaps: bool = False,
        do_context: bool = False,
        imgaug_transform: Optional[Callable] = None,
        delimiter: str = "img"
    ) -> None:
        """Initialize the MultiViewHeatmap Dataset.

        Args:
            root_directory: path to data directory
            csv_paths: paths to CSV files (within root_directory). CSV files
                should be in this form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                these should match in all CSV files
                Note: image_path is relative to the given root_directory
                we suggest that these CSV files start with the view numbers
            view_names: a list of integers with the view numbers
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            downsample_factor: factor by which to downsample original image dims to have a smaller
                heatmap
            do_context: include additional frames of context if possible
        """

        if len(view_names) != len(csv_paths):
            raise ValueError("number of names does not match with the number of files!")

        self.delimiter = delimiter
        self.do_context = do_context

        self.imgaug_transform = imgaug_transform
        self.downsample_factor = downsample_factor
        self.dataset = {}
        self.keypoint_names = {}
        self.data_length = {}
        self.num_keypoints = {}
        for view, csv_path in zip(view_names, csv_paths):
            self.dataset[view] = HeatmapDataset(
                root_directory=root_directory,
                csv_path=csv_path,
                header_rows=header_rows,
                imgaug_transform=imgaug_transform,
                downsample_factor=downsample_factor,
                do_context=do_context,
                uniform_heatmaps=uniform_heatmaps,
                delimiter=self.delimiter
            )
            self.keypoint_names[view] = self.dataset[view].keypoint_names
            self.data_length[view] = len(self.dataset[view])
            self.num_keypoints[view] = self.dataset[view].num_keypoints

        self.view_names = view_names

        # check if all CSV files have the same number of columns
        self.num_keypoints = sum(self.num_keypoints.values())

        # check if all the data is in correct order, self.data_length changes here
        self.check_data_images_names()

        self.num_targets = self.num_keypoints * 2

    def check_data_images_names(self):
        """Data checking
        Each object in self.datasets will have the attribute image_names
        (i.e. self.datasets['top'].image_names) since each values is a
        HeatmapDataset. Include a check to make sure that the image names
        are the same across all views, so that when it loads element n from
        each individual view we know these are properly matched.

        Args:

            delimiter: for spliting the file name string to get the frame number and format.

        """
        # check if all CSV files have the same number of rows
        if len(set(list(self.data_length.values()))) != 1:
            raise ImportError("the CSV files do not match in row numbers!")

        for key_num, keypoint in enumerate(self.keypoint_names[self.view_names[0]]):
            for view, keypointComp in self.keypoint_names.items():
                if keypoint != keypointComp[key_num]:
                    raise ImportError(f"the keypoints are not in correct order! \
                                      view: {self.view_names[0]} vs {view} | \
                                        {keypoint} != {keypointComp}")

        self.data_length = list(self.data_length.values())[0]
        for idx in range(self.data_length):
            img_name_buff = []
            for view, heatmaps in self.dataset.items():
                img_name_buff.append(heatmaps.image_names[idx].split(self.delimiter)[-1])
                if len(set(img_name_buff)) != 1:
                    raise ImportError(f"Discrepancy in images names across CSV \
                                      files! index:{idx}, image frame names:{img_name_buff}")

    @property
    def height(self) -> int:
        return self.imgaug_transform[-1].get_parameters()[0][0].value

    @property
    def width(self) -> int:
        # assume resizing transformation is the last imgaug one
        return self.imgaug_transform[-1].get_parameters()[0][1].value

    def __len__(self) -> int:
        return self.data_length

    @property
    def output_shape(self) -> tuple:
        return (
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    @property
    def num_views(self) -> int:
        return len(self.view_names)

    def fusion(self, datadict: dict) -> Tuple[
        Union[
            TensorType["num_views", "RGB":3, "image_height", "image_width", float],
            TensorType["num_views", "frames", "RGB":3, "image_height", "image_width", float]
        ],
        TensorType["keypoints"],
        TensorType["num_views", "heatmap_height", "heatmap_width", float],
        TensorType["num_views", "xyhw":4, float],
        List,
    ]:
        """Merge images, heatmaps, keypoints, and bboxes across views.

        Args:
            datadict: this comes from HeatmapDataset.__getItems__(idx) for each view.

        Returns:
            tuple
                - images
                - keypoints
                - heatmaps
                - bboxes
                - concat order

        """
        images = []
        keypoints = []
        heatmaps = []
        bboxes = []
        concat_order = []
        for view, data in datadict.items():
            images.append(data["images"].unsqueeze(0))
            data["keypoints"] = data["keypoints"].reshape(int(data["keypoints"].shape[0] / 2), 2)
            keypoints.append(data["keypoints"])
            heatmaps.append(data["heatmaps"])
            bboxes.append(data["bbox"])
            concat_order.append(view)

        images = torch.cat(images, dim=0)
        keypoints = torch.cat(keypoints, dim=0).reshape(-1)
        heatmaps = torch.cat(heatmaps, dim=0)
        bboxes = torch.cat(bboxes, dim=0)

        assert keypoints.shape == (self.num_targets,)

        return images, keypoints, heatmaps, bboxes, concat_order

    def __getitem__(self, idx: int) -> MultiviewHeatmapLabeledExampleDict:
        """Get an example from the dataset.
        Calls the heatmapdataset for each csv file to get
        Images and their heatmaps and then stacks them.
        """
        datadict = {}
        for view in self.view_names:
            datadict[view] = self.dataset[view][idx]

        images, keypoints, heatmaps, bboxes, concat_order = self.fusion(datadict)
        # images normal:[view, RGB, H, W] context:[view, context, RGB, H, W]

        return MultiviewHeatmapLabeledExampleDict(
            images=images,  # shape (3, H, W) or (5, 3, H, W)
            keypoints=keypoints,  # shape (n_targets,)
            heatmaps=heatmaps,
            bbox=bboxes,
            idxs=idx,
            num_views=self.num_views,  # int
            concat_order=concat_order,  # List[int]
            view_names=self.view_names,  # List[int]
        )
