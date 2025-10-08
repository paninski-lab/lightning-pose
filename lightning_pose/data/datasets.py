"""Dataset objects store images, labels, and functions for manipulation."""

import os
from pathlib import Path
from typing import Callable, Literal, Tuple, Union

import cv2
import imgaug.augmenters as iaa
import kornia.geometry.transform as ktransform
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchtyping import TensorType
from torchvision import transforms

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.cameras import CameraGroup
from lightning_pose.data.datatypes import (
    BaseLabeledExampleDict,
    HeatmapLabeledExampleDict,
    MultiviewHeatmapLabeledExampleDict,
)
from lightning_pose.data.utils import generate_heatmaps, normalized_to_bbox
from lightning_pose.utils import io as io_utils

# to ignore imports for sphix-autoapidoc
__all__ = [
    "BaseTrackingDataset",
    "HeatmapDataset",
    "MultiviewHeatmapDataset",
]


class BaseTrackingDataset(torch.utils.data.Dataset):
    """Base dataset that contains images and keypoints as (x, y) pairs."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        image_resize_height: int,
        image_resize_width: int,
        header_rows: list[int] | None = [0, 1, 2],
        imgaug_transform: Callable | None = None,
        do_context: bool = False,
        resize: bool = True,
        bbox_path: str | None = None,
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
            resize_height: height to resize images before sending to network
            resize_width: height to resize images before sending to network
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            do_context: include additional frames of context if possible.
            resize: True to add final resizing augmentation before sending data to network. This
                can be set to False if inheritors of this class need to implement more
                sophisticated augmentations before resizing (e.g. 3d augmentations). Note that when
                this is False, it is up to the child class to perform this resizing on both images
                and keypoints before returning a batch of data.
            bbox_path: path to csv file that contains bounding box information; rows must be in
                same order as csv file

        """
        self.root_directory = Path(root_directory)
        self.image_resize_height = image_resize_height
        self.image_resize_width = image_resize_width
        self.csv_path = csv_path
        self.bbox_path = bbox_path
        self.header_rows = header_rows
        self.do_context = do_context
        if resize:
            imgaug_transform.add(iaa.Resize({
                "height": image_resize_height,
                "width": image_resize_width,
            }))
        self.imgaug_transform = imgaug_transform

        # load csv data
        if os.path.isfile(csv_path):
            csv_file = csv_path
        else:
            csv_file = os.path.join(root_directory, csv_path)
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Could not find csv file at {csv_file}!")

        csv_data = pd.read_csv(csv_file, header=header_rows, index_col=0)
        csv_data = io_utils.fix_empty_first_row(csv_data)
        self.keypoint_names = io_utils.get_keypoint_names(
            csv_file=csv_file, header_rows=header_rows,
        )
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

        # load bounding box data
        if bbox_path:
            if os.path.isfile(bbox_path):
                bbox_file = bbox_path
            else:
                bbox_file = os.path.join(root_directory, bbox_path)
            if not os.path.exists(bbox_file):
                raise FileNotFoundError(f"Could not find bbox file at {bbox_file}!")
            bboxes_df = pd.read_csv(bbox_file, header=[0], index_col=0)
            assert bboxes_df.index.equals(csv_data.index)
            bboxes = bboxes_df.to_numpy()
        else:
            bboxes = [None] * len(csv_data)
        self.bboxes = bboxes

    @property
    def height(self) -> int:
        return self.image_resize_height

    @property
    def width(self) -> int:
        return self.image_resize_width

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> BaseLabeledExampleDict:
        img_name = self.image_names[idx]
        keypoints_on_image = self.keypoints[idx]
        img_path = self.root_directory / img_name
        if not self.do_context:
            # read image from file and apply transformations (if any)
            # if 1 color channel, change to 3.
            image = Image.open(img_path).convert("RGB")
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
            context_img_paths = io_utils.get_context_img_paths(img_path)
            # read the images from image list to create dataset
            images = []
            for path in context_img_paths:
                # read image from file and apply transformations (if any)
                if not path.exists():
                    # revert to center frame
                    path = context_img_paths[2]
                # if 1 color channel, change to 3.
                image = Image.open(path).convert("RGB")
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

        # x,y,h,w of bounding box
        if self.bboxes[idx] is not None:
            bbox = torch.tensor(self.bboxes[idx])
        else:
            bbox = torch.tensor([0, 0, image.height, image.width])

        return BaseLabeledExampleDict(
            images=transformed_images,  # shape (3, img_height, img_width) or (5, 3, H, W)
            keypoints=torch.from_numpy(transformed_keypoints),  # shape (n_targets,)
            idxs=idx,
            bbox=bbox,
        )


# the only addition here, should be the heatmap creation method.
class HeatmapDataset(BaseTrackingDataset):
    """Heatmap dataset that contains the images and keypoints in 2D arrays."""

    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        image_resize_height: int,
        image_resize_width: int,
        header_rows: list[int] | None = [0, 1, 2],
        imgaug_transform: Callable | None = None,
        downsample_factor: Literal[1, 2, 3] = 2,
        do_context: bool = False,
        resize: bool = True,
        uniform_heatmaps: bool = False,
        bbox_path: str | None = None,
    ) -> None:
        """Initialize the Heatmap Dataset.

        Args:
            root_directory: path to data directory
            csv_path: path to CSV or h5 file  (within root_directory). CSV file
                should be in the form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            image_resize_height: height to resize images before sending to network
            image_resize_width: height to resize images before sending to network
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            downsample_factor: factor by which to downsample original image dims to have a smaller
                heatmap
            do_context: include additional frames of context if possible
            resize: True to add final resizing augmentation before sending data to network. This
                can be set to False if inheritors of this class need to implement more
                sophisticated augmentations before resizing (e.g. 3d augmentations). Note that when
                this is False, it is up to the child class to perform this resizing on both images
                and keypoints before returning a batch of data.
            uniform_heatmaps: True to force the model to output uniform heatmaps for missing data;
                False will output all-zero heatmaps
            bbox_path: path to csv file that contains bounding box information; rows must be in
                same order as csv file

        """
        super().__init__(
            root_directory=root_directory,
            csv_path=csv_path,
            image_resize_height=image_resize_height,
            image_resize_width=image_resize_width,
            header_rows=header_rows,
            imgaug_transform=imgaug_transform,
            do_context=do_context,
            resize=resize,
            bbox_path=bbox_path,
        )

        if self.height % 128 != 0 or self.height % 128 != 0:
            print("image dimensions (after transformation) must be repeatably divisible by 2!")
            print("current image dimensions after transformation are:")
            exit()

        self.downsample_factor = downsample_factor
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor
        self.uniform_heatmaps = uniform_heatmaps
        self.num_targets = torch.numel(self.keypoints[0])
        self.num_keypoints = self.num_targets // 2

    @property
    def output_shape(self) -> tuple:
        return (
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    def compute_heatmap(
        self,
        example_dict: BaseLabeledExampleDict,
        ignore_nans: bool = False,
    ) -> TensorType["num_keypoints", "heatmap_height", "heatmap_width"]:
        """Compute 2D heatmaps from arbitrary (x, y) coordinates."""

        # reshape
        keypoints = example_dict["keypoints"].reshape(self.num_keypoints, 2)

        # introduce nans where data augmentation has moved the keypoint out of the original frame
        if not ignore_nans:
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
        """Compute initial 2D heatmaps for all labeled data. Note this will apply augmentations.

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

        return label_heatmaps

    def __getitem__(self, idx: int, ignore_nans: bool = False) -> HeatmapLabeledExampleDict:
        """Get an example from the dataset."""
        # call base dataset to get an image and labels
        example_dict: BaseLabeledExampleDict = super().__getitem__(idx)
        # compute the corresponding heatmaps
        example_dict["heatmaps"] = self.compute_heatmap(example_dict, ignore_nans)
        return example_dict


class MultiviewHeatmapDataset(torch.utils.data.Dataset):
    """Heatmap dataset that contains the images and keypoints in 2D arrays from all the cameras."""

    def __init__(
        self,
        root_directory: str,
        csv_paths: list[str],
        view_names: list[str],
        image_resize_height: int,
        image_resize_width: int,
        header_rows: list[int] | None = [0, 1, 2],
        imgaug_transform: Callable | None = None,
        downsample_factor: Literal[1, 2, 3] = 2,
        do_context: bool = False,
        resize: bool = False,
        uniform_heatmaps: bool = False,
        camera_params_path: str | None = None,
        bbox_paths: list[str] | None = None,
    ) -> None:
        """Initialize the MultiViewHeatmap Dataset.

        Args:
            root_directory: path to data directory
            csv_paths: paths to CSV files (within root_directory). CSV files should be in this form
                (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                these should match in all CSV files
                Note: image_path is relative to the given root_directory
            view_names: a list of strings with the view names
            image_resize_height: height to resize images before sending to network
            image_resize_width: height to resize images before sending to network
            header_rows: which rows in the csv are header rows
            imgaug_transform: imgaug transform pipeline to apply to images
            downsample_factor: factor by which to downsample original image dims to have a smaller
                heatmap
            do_context: include additional frames of context if possible
            resize: True to add final resizing augmentation before sending data to network. This
                can be set to False if inheritors of this class need to implement more
                sophisticated augmentations before resizing (e.g. 3d augmentations). Note that when
                this is False, it is up to the child class to perform this resizing on both images
                and keypoints before returning a batch of data.
            uniform_heatmaps: True to force the model to output uniform heatmaps for missing data;
                False will output all-zero heatmaps
            camera_params_path: path to toml file with camera calibration parameters in format
                output by anipose
            bbox_paths: paths to csv files of the form
                (image_path, x, h, height, width)
                where (x, y) correspond to upper left corner of bbox.
                These files should be in the same view order as the csv paths

        """

        if len(view_names) != len(csv_paths):
            raise ValueError("number of names does not match with the number of files!")
        print("Using MultiviewHeatmapDataset")
        self.root_directory = root_directory
        self.csv_paths = csv_paths
        self.bbox_paths = bbox_paths or [None] * len(view_names)
        self.view_names = view_names
        self.image_resize_height = image_resize_height
        self.image_resize_width = image_resize_width
        self.do_context = do_context

        # do this here so resizing doesn't get added multiple times when iterating over views
        if resize:
            imgaug_transform.add(iaa.Resize({
                "height": image_resize_height,
                "width": image_resize_width,
            }))
        self.imgaug_transform = imgaug_transform

        self.downsample_factor = downsample_factor
        self.dataset = {}
        self.keypoint_names = {}
        self.data_length = {}
        self.num_keypoints = {}
        for view, csv_path, bbox_path in zip(view_names, csv_paths, self.bbox_paths):
            self.dataset[view] = HeatmapDataset(
                root_directory=root_directory,
                csv_path=csv_path,
                image_resize_height=image_resize_height,
                image_resize_width=image_resize_width,
                header_rows=header_rows,
                imgaug_transform=imgaug_transform,
                downsample_factor=downsample_factor,
                do_context=do_context,
                resize=False,  # handled above in L396
                uniform_heatmaps=uniform_heatmaps,
                bbox_path=bbox_path,
            )
            self.keypoint_names[view] = self.dataset[view].keypoint_names
            self.data_length[view] = len(self.dataset[view])
            self.num_keypoints[view] = self.dataset[view].num_keypoints

        # check if all csv files have the same number of columns
        self.num_keypoints = sum(self.num_keypoints.values())

        # check if all the data is in correct order, self.data_length changes here
        self.check_data_images_names()

        self.num_targets = self.num_keypoints * 2

        if camera_params_path is not None:

            assert not do_context, "3D augmentations for context model not yet supported"

            cam_params_df = pd.read_csv(camera_params_path, index_col=0, header=[0])

            # make sure image numbers at least match
            img_idxs_labels = [
                i.split('/')[-1] for i in self.dataset[self.view_names[0]].image_names
            ]
            img_idxs_calib = [i.split('/')[-1] for i in cam_params_df.index]
            assert np.all(img_idxs_labels == img_idxs_calib)

            cam_params_file_to_camgroup = {}
            for cam_params_file in cam_params_df.file.unique():
                camgroup = CameraGroup.load(os.path.join(root_directory, cam_params_file))
                cam_names = camgroup.get_names()
                assert np.all(cam_names == view_names), (
                    "cfg.data.view_names must have same camera order as camera calibration file; "
                    f"instead found {view_names} and {cam_names}."
                )
                cam_params_file_to_camgroup[cam_params_file] = camgroup

        else:
            cam_params_df = None
            cam_params_file_to_camgroup = None

        self.cam_params_df = cam_params_df
        self.cam_params_file_to_camgroup = cam_params_file_to_camgroup

    def check_data_images_names(self):
        """Data checking
        Each object in self.datasets will have the attribute image_names
        (i.e. self.datasets['top'].image_names) since each values is a
        HeatmapDataset. Include a check to make sure that the image names
        are the same across all views, so that when it loads element n from
        each individual view we know these are properly matched.
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
            img_file_names = set()
            for view, heatmaps in self.dataset.items():
                img_file_names.add(Path(heatmaps.image_names[idx]).name)
                if len(img_file_names) > 1:
                    raise ImportError(
                        "Discrepancy in image file names across CSV files! "
                        "index:{idx}, image file names:{img_file_names}"
                    )

    @property
    def height(self) -> int:
        return self.image_resize_height

    @property
    def width(self) -> int:
        return self.image_resize_width

    def __len__(self) -> int:
        return self.data_length

    @property
    def output_shape(self) -> tuple:
        return (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )

    @property
    def num_views(self) -> int:
        return len(self.view_names)

    @staticmethod
    def _scale_translate_keypoints(
        keypoints_3d: np.ndarray,
        scale_params: tuple = (0.8, 1.2),
        shift_param: float = 0.25,
    ) -> np.ndarray:
        """Apply scale and translation transforms to 3D keypoints.

        Parameters
        ----------
        keypoints_3d: original keypoints, shape (num_keypoints, 3)
        scale_params: (a, b) are (min, max) ratio of scaling
        shift_param: max shift in each dimension, as a fraction of the range (kp_max - kp_min)

        Returns
        -------
        np.ndarray: augmented keypoints, shape (num_keypoints, 3)

        """
        # step 1: apply scaling
        scale_factor = np.random.uniform(*scale_params)  # scale scene up or down

        median = np.nanmedian(keypoints_3d, axis=0)
        keypoints_aug = (keypoints_3d - median) * scale_factor + median

        # step 2: apply translation
        extent = np.nanmax(keypoints_aug, axis=0) - np.nanmin(keypoints_aug, axis=0)
        rands = 2 * np.random.rand(3) - 1  # in [-1, 1]
        shift = shift_param * extent * rands
        keypoints_aug += shift

        return keypoints_aug

    @staticmethod
    def _transform_images(
        images: list,
        keypoints_orig: np.ndarray,
        keypoints_aug: np.ndarray,
        bboxes: list[np.ndarray],
    ) -> list:
        """Apply 2D transformations based on keypoint matching.

        Parameters
        ----------
        images: each element is a torch array of shape (3, height, width); h/w can be different for
            different views
        keypoints_orig: shape (num_views, num_keypoints, 2)
        keypoints_aug: shape (num_views, num_keypoints, 2)
        bboxes: shape (num_view, 4) -> x, y, h, w

        Returns
        -------
        list: each element corresponds to an augmented version of the input images; same device

        """

        device = images[0].device
        images_transformed = []

        for orig_img, kps_og, kps_aug, bbox in zip(images, keypoints_orig, keypoints_aug, bboxes):

            _, img_height, img_width = orig_img.shape

            # create a mask for valid keypoints (not NaN in either original or augmented)
            valid_mask = ~(np.isnan(kps_og).any(axis=1) | np.isnan(kps_aug).any(axis=1))

            # apply the same mask to both original and augmented keypoints
            orig_pts = kps_og[valid_mask]
            new_pts = kps_aug[valid_mask]

            # transform data points from original coordinate space to frame coordinate
            orig_pts[:, 0] = (orig_pts[:, 0] - bbox[0]) / bbox[3] * img_width
            new_pts[:, 0] = (new_pts[:, 0] - bbox[0]) / bbox[3] * img_width
            orig_pts[:, 1] = (orig_pts[:, 1] - bbox[1]) / bbox[2] * img_height
            new_pts[:, 1] = (new_pts[:, 1] - bbox[1]) / bbox[2] * img_height

            # ensure we have enough points for transformation
            if len(orig_pts) < 3:
                # If not enough points, return original image
                images_transformed.append(orig_img.clone().unsqueeze(0))
                continue

            # estimate the affine transformation matrix with non-uniform scaling
            M, _ = cv2.estimateAffinePartial2D(orig_pts, new_pts)

            # convert to tensor
            M_tensor = torch.tensor(M, dtype=torch.float32, device=device)

            # apply affine transformation
            # image has already been normalized, so pad with minimum value of image instead of 0s
            transformed_image = ktransform.warp_affine(
                orig_img.unsqueeze(0),  # (C, H, W)
                M_tensor.unsqueeze(0),  # (2, 3)
                dsize=(img_height, img_width),  # Keep original size
                padding_mode="fill",
                fill_value=torch.ones(3, device=orig_img.device) * orig_img.min(),
            )

            images_transformed.append(transformed_image)

        return images_transformed

    def _get_2d_keypoints_from_example_dict_absolute_coords(
        self,
        data_dict: dict,
        clone: bool = True,
    ) -> np.ndarray:
        keypoints_2d = np.zeros((self.num_views, self.num_keypoints // self.num_views, 2))
        for idx_view, (view, example_dict) in enumerate(data_dict.items()):
            if clone:
                keypoints_curr = example_dict["keypoints"].reshape(
                    self.num_keypoints // self.num_views, 2
                ).clone()
            else:
                keypoints_curr = example_dict["keypoints"].reshape(
                    self.num_keypoints // self.num_views, 2
                )
            # transform keypoints from bbox coordinates to absolute frame coordinates
            # 1. divide by image dims to get 0-1 normalized coords
            keypoints_curr[:, 0] = keypoints_curr[:, 0] / example_dict["images"].shape[-1]  # -1 x
            keypoints_curr[:, 1] = keypoints_curr[:, 1] / example_dict["images"].shape[-2]  # -2 y
            # 2. multiply and add by bbox dims
            keypoints_2d[idx_view] = normalized_to_bbox(
                keypoints=keypoints_curr.unsqueeze(0),
                bbox=example_dict["bbox"].unsqueeze(0),
            )[0].cpu().numpy()
        return keypoints_2d

    def _resize_keypoints(self, keypoints: np.ndarray, bboxes: list) -> list:
        """Resize keypoints to a uniform shape and return torch arrays."""
        keypoints_resized = []
        for idx_view in range(self.num_views):
            keypoints_ = keypoints[idx_view].copy()  # shape (num_keypoints, 2)
            bbox_ = bboxes[idx_view].cpu().numpy()
            keypoints_[:, 0] = ((keypoints_[:, 0] - bbox_[0]) / bbox_[3]) * self.width
            keypoints_[:, 1] = ((keypoints_[:, 1] - bbox_[1]) / bbox_[2]) * self.height
            keypoints_resized.append(keypoints_.reshape(-1))
        return keypoints_resized

    def _resize_images(self, images: list) -> list:
        """Resize images to a uniform shape."""
        images_resized = []
        for idx_view in range(self.num_views):
            images_resized.append(ktransform.resize(
                images[idx_view].clone(),
                size=(self.height, self.width),
            ))
        return images_resized

    def apply_3d_transforms(self, data_dict: dict, camgroup: CameraGroup) -> tuple:
        """Apply 3D transforms to keypoint and image data (scale, translate)."""

        # extract keypoints and images from each view
        keypoints_2d = self._get_2d_keypoints_from_example_dict_absolute_coords(data_dict)
        images = []
        bboxes = []
        for idx_view, (view, example_dict) in enumerate(data_dict.items()):
            images.append(example_dict["images"])
            bboxes.append(example_dict["bbox"])

        if np.all(np.isnan(keypoints_2d)):
            keypoints_3d_aug = np.nan * np.zeros((self.num_keypoints // self.num_views, 3))
            keypoints_2d_aug_resize = [
                torch.tensor(
                    np.nan * np.zeros((self.num_keypoints // self.num_views * 2)),
                    dtype=example_dict["keypoints"].dtype,
                    device=example_dict["keypoints"].device,
                )
                for _, example_dict in data_dict.items()
            ]
            images_aug = [im.unsqueeze(0) for im in images]
        else:

            # triangulate keypoints (2D -> 3D)
            keypoints_3d = camgroup.triangulate_fast(keypoints_2d.copy())

            # scale and translate keypoints in 3D
            keypoints_3d_aug = self._scale_translate_keypoints(keypoints_3d)

            # project 3D keypoints to 2D using the rotated cameras
            keypoints_2d_aug = camgroup.project(keypoints_3d_aug)

            # resize 2D keypoints to uniform dimensions for backbone network
            keypoints_2d_aug_resize_np = self._resize_keypoints(keypoints_2d_aug, bboxes)
            keypoints_2d_aug_resize = [
                torch.tensor(
                    a,
                    dtype=example_dict["keypoints"].dtype,
                    device=example_dict["keypoints"].device,
                )
                for a in keypoints_2d_aug_resize_np
            ]

            # transform images to match keypoint augmentations
            images_aug = self._transform_images(
                images=images,
                keypoints_orig=keypoints_2d.copy(),
                keypoints_aug=keypoints_2d_aug.copy(),
                bboxes=[b.cpu().numpy() for b in bboxes],
            )

        # resize to uniform dimensions for backbone network
        images_aug_resize = self._resize_images(images_aug)

        # create new data dict
        data_dict_aug = {}
        for idx_view, view in enumerate(self.view_names):
            example_dict = BaseLabeledExampleDict(
                images=images_aug_resize[idx_view][0],  # take image from view, ignore batch dim
                keypoints=keypoints_2d_aug_resize[idx_view],
                bbox=data_dict[view]["bbox"],
                idxs=data_dict[view]["idxs"],
            )
            example_dict["heatmaps"] = self.dataset[view].compute_heatmap(example_dict)
            data_dict_aug[view] = example_dict

        return data_dict_aug, torch.tensor(keypoints_3d_aug)

    def fusion(self, datadict: dict) -> Tuple[
        Union[
            TensorType["num_views", "RGB":3, "image_height", "image_width", float],
            TensorType["num_views", "frames", "RGB":3, "image_height", "image_width", float]
        ],
        TensorType["keypoints"],
        TensorType["num_views", "heatmap_height", "heatmap_width", float],
        TensorType["num_views * xyhw", float],
        list,
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

        Calls the HeatmapDataset for each view to get images, keypoints, and heatmaps then stacks
        them.

        """

        # load frames/keypoints and apply per-frame augmentations
        ignore_nans = True if self.cam_params_file_to_camgroup else False
        datadict = {}
        for view in self.view_names:
            datadict[view] = self.dataset[view].__getitem__(idx, ignore_nans=ignore_nans)

        # always provide 3D keypoints when camera params are available
        if self.cam_params_file_to_camgroup:
            # select proper camera calibration parameters for this data point
            camgroup = self.cam_params_file_to_camgroup[self.cam_params_df.iloc[idx].file]

            # load camera parameters
            intrinsic_matrix = torch.stack([
                torch.tensor(cam.get_camera_matrix()) for cam in camgroup.cameras
            ], dim=0)
            extrinsic_matrix = torch.stack([
                torch.tensor(cam.get_extrinsics_mat()[:3]) for cam in camgroup.cameras
            ], dim=0)
            distortions = torch.stack([
                torch.tensor(cam.get_distortions()) for cam in camgroup.cameras
            ], dim=0)

            # check if we should apply 3D augmentations (training) or just triangulate (validation)
            if self.imgaug_transform.__str__().find("Resize") == -1:
                # training: apply 3D transforms with augmentations
                datadict, keypoints_3d = self.apply_3d_transforms(datadict, camgroup)
            else:
                # validation: triangulate 3D keypoints without augmentations
                # extract keypoints from each view for triangulation (same as apply_3d_transforms)
                keypoints_2d = self._get_2d_keypoints_from_example_dict_absolute_coords(
                    datadict, clone=True,
                )
                # triangulate keypoints (2D -> 3D) without augmentations
                if np.all(np.isnan(keypoints_2d)):
                    keypoints_3d = torch.tensor(
                        np.nan * np.zeros((self.num_keypoints // self.num_views, 3)),
                    )
                else:
                    keypoints_3d = torch.tensor(
                        camgroup.triangulate_fast(keypoints_2d),
                    )

        else:
            # Use default values when no camera calibration
            keypoints_3d = torch.tensor([1])
            intrinsic_matrix = torch.eye(3).unsqueeze(0)
            extrinsic_matrix = torch.zeros(1, 3, 4)
            distortions = torch.zeros(1, 5)

        # fuse data from all views
        images, keypoints, heatmaps, bboxes, concat_order = self.fusion(datadict)
        assert np.all(concat_order == self.view_names)
        # images normal:[view, RGB, H, W] context:[view, context, RGB, H, W]

        return MultiviewHeatmapLabeledExampleDict(
            images=images.clone(),  # shape (3, H, W) or (5, 3, H, W)
            keypoints=keypoints.clone(),  # shape (n_targets,)
            heatmaps=heatmaps.clone(),
            bbox=bboxes.clone(),
            idxs=idx,
            num_views=self.num_views,  # int
            concat_order=concat_order,  # list[str]
            view_names=self.view_names,  # list[str]
            keypoints_3d=keypoints_3d.clone(),
            intrinsic_matrix=intrinsic_matrix.clone(),
            extrinsic_matrix=extrinsic_matrix.clone(),
            distortions=distortions.clone(),
        )
