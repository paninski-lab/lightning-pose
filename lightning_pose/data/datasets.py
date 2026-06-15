"""Dataset objects store images, labels, and functions for manipulation."""

import logging
import os
from pathlib import Path
from typing import Literal, cast

import cv2
import imgaug.augmenters as iaa
import imgaug.augmenters.size as _iaa_size
import kornia.geometry.transform as ktransform
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from PIL import Image
from torchvision import transforms

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.bboxes import norm_to_frame
from lightning_pose.data.cameras import CameraGroup
from lightning_pose.data.datatypes import (
    BaseLabeledExampleDict,
    HeatmapLabeledExampleDict,
    MultiviewHeatmapLabeledExampleDict,
)
from lightning_pose.data.heatmaps import generate_heatmaps
from lightning_pose.utils import io as io_utils

logger = logging.getLogger(__name__)

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


def _patched_prevent(axis_size: int, crop_start: int, crop_end: int) -> tuple[int, ...]:
    """Monkey patch to fix imaug 0.4.2 compatability issue with numpy 2.x"""
    result = _iaa_size._prevent_zero_sizes_after_crops_(
        np.array([axis_size], dtype=np.int32),
        np.array([crop_start], dtype=np.int32),
        np.array([crop_end], dtype=np.int32),
    )
    return tuple(int(np.asarray(v).flat[0]) for v in result)


#  monkey patch to fix imaug 0.4.2 compatability issue with numpy 2.x
_iaa_size._prevent_zero_size_after_crop_ = _patched_prevent


class BaseTrackingDataset(torch.utils.data.Dataset):
    """Base dataset that contains images and keypoints as (x, y) pairs."""

    def __init__(
        self,
        root_directory: str | Path,
        csv_path: str,
        image_resize_height: int,
        image_resize_width: int,
        header_rows: list[int] | None = [0, 1, 2],
        imgaug_transform: iaa.Sequential | None = None,
        do_context: bool = False,
        resize: bool = True,
        bbox_path: str | None = None,
        imgaug_hflip: bool = False,
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
            imgaug_hflip: if True, apply a random horizontal flip with probability 0.5 after the
                standard imgaug pipeline. All keypoint x-coordinates are mirrored; keypoints whose
                names end in ``_left`` or ``_right`` are additionally swapped with their partner so
                that label identity is preserved. Every ``_left`` keypoint must have a matching
                ``_right`` keypoint and vice versa, or a ValueError is raised. Disabled
                automatically for validation and test subsets by the data module.

        """
        self.root_directory = Path(root_directory)
        self.image_resize_height = image_resize_height
        self.image_resize_width = image_resize_width
        self.csv_path = csv_path
        self.bbox_path = bbox_path
        self.header_rows = header_rows
        self.do_context = do_context
        if resize:
            assert imgaug_transform is not None
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

        labeled_data = io_utils.parse_label_csv(csv_file, header_rows=header_rows)
        self.keypoint_names = labeled_data.keypoint_names
        self.image_names = labeled_data.image_names
        self.keypoints = labeled_data.keypoints
        self.visibility: torch.Tensor | None = labeled_data.visibility

        if self.visibility is not None:
            occluded_with_coords = (
                (self.visibility == 1) & ~torch.isnan(self.keypoints[:, :, 0])
            )
            if occluded_with_coords.any():
                logger.warning(
                    'found keypoints with visible=1 (occluded) that have non-NaN x,y '
                    'coordinates; the visibility flag takes precedence and a uniform heatmap '
                    'will be generated for these keypoints'
                )

        # send image to tensor and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

        # keypoints has been already transformed above
        self.num_targets = self.keypoints.shape[1] * 2
        self.num_keypoints = self.keypoints.shape[1]

        self.imgaug_hflip = imgaug_hflip
        if imgaug_hflip:
            logger.info("applying horizontal flip")
            self._hflip_swap_indices = self._build_hflip_swap_indices(self.keypoint_names)
        else:
            self._hflip_swap_indices = np.arange(self.num_keypoints, dtype=np.intp)

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
            assert bboxes_df.index.tolist() == self.image_names
            bboxes = bboxes_df.to_numpy()
        else:
            bboxes = [None] * len(self.image_names)
        self.bboxes = bboxes

    @staticmethod
    def _build_hflip_swap_indices(keypoint_names: list[str]) -> np.ndarray:
        """Build an index array that swaps lateralized (_left/_right) keypoint pairs.

        Args:
            keypoint_names: list of keypoint name strings from the label CSV.

        Returns:
            integer array of shape (num_keypoints,) where entry i gives the source keypoint
            index that should fill position i after a horizontal flip. Non-lateralized keypoints
            map to themselves; each _left/_right pair maps to its partner.

        Raises:
            ValueError: if any keypoint ending in _left has no matching _right partner,
                or vice versa.
        """
        indices = list(range(len(keypoint_names)))
        left_map = {
            name[:-5]: i for i, name in enumerate(keypoint_names) if name.endswith('_left')
        }
        right_map = {
            name[:-6]: i for i, name in enumerate(keypoint_names) if name.endswith('_right')
        }

        unmatched_left = sorted(f'{b}_left' for b in set(left_map) - set(right_map))
        unmatched_right = sorted(f'{b}_right' for b in set(right_map) - set(left_map))
        if unmatched_left:
            raise ValueError(
                f'imgaug_hflip requires matching _left/_right pairs, '
                f'but found _left keypoints with no _right partner: {unmatched_left}'
            )
        if unmatched_right:
            raise ValueError(
                f'imgaug_hflip requires matching _left/_right pairs, '
                f'but found _right keypoints with no _left partner: {unmatched_right}'
            )

        for base_name in left_map:
            idx_left = left_map[base_name]
            idx_right = right_map[base_name]
            indices[idx_left] = idx_right
            indices[idx_right] = idx_left

        return np.array(indices, dtype=np.intp)

    @property
    def height(self) -> int:
        """Image height in pixels after resizing."""
        return self.image_resize_height

    @property
    def width(self) -> int:
        """Image width in pixels after resizing."""
        return self.image_resize_width

    def __len__(self) -> int:
        """Return the number of labeled examples in the dataset."""
        return self.data_length

    def __getitem__(self, idx: int) -> BaseLabeledExampleDict:
        """Return one labeled example as a dictionary.

        Args:
            idx: index into the dataset.

        Returns:
            Dictionary with keys ``"images"``, ``"keypoints"``, ``"bbox"``, and ``"image_file"``.
        """
        img_name = self.image_names[idx]
        keypoints_on_image = self.keypoints[idx]
        img_path = self.root_directory / img_name
        if not self.do_context:
            do_hflip = self.imgaug_hflip and np.random.random() < 0.5
            # read image from file and apply transformations (if any)
            # if 1 color channel, change to 3.
            image = Image.open(img_path).convert("RGB")
            if self.imgaug_transform is not None:
                imgs_aug, kps_aug = self.imgaug_transform(  # type: ignore[misc]
                    images=np.expand_dims(np.array(image), axis=0),
                    keypoints=np.expand_dims(keypoints_on_image, axis=0),
                )  # expands add batch dim for imgaug
                # get rid of the batch dim
                transformed_images = imgs_aug[0]
                transformed_keypoints = kps_aug[0].reshape(-1)

                if do_hflip:
                    transformed_images = np.ascontiguousarray(np.fliplr(transformed_images))
                    kps_2d = transformed_keypoints.reshape(self.num_keypoints, 2).copy()
                    kps_2d[:, 0] = self.image_resize_width - kps_2d[:, 0]
                    kps_2d = kps_2d[self._hflip_swap_indices]
                    transformed_keypoints = kps_2d.reshape(-1)
            else:
                transformed_images = np.expand_dims(np.array(image), axis=0)
                transformed_keypoints = np.expand_dims(keypoints_on_image, axis=0)

            transformed_images = self.pytorch_transform(transformed_images)
            assert isinstance(transformed_images, torch.Tensor)

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

            # decide flip once so all context frames get the same transformation
            do_hflip = self.imgaug_hflip and np.random.random() < 0.5

            # apply data aug pipeline
            if self.imgaug_transform is not None:
                # need to apply the same transform to all context frames
                seed = np.random.randint(low=0, high=123456)
                transformed_images = []
                for img in images:
                    self.imgaug_transform.seed_(seed)
                    img_aug, kps_aug = self.imgaug_transform(  # type: ignore[misc]
                        images=[img], keypoints=[keypoints_on_image.numpy()]
                    )
                    transformed_images.append(img_aug[0])
                transformed_images = np.asarray(transformed_images)
                transformed_keypoints = kps_aug[0].reshape(-1)

                if do_hflip:
                    transformed_images = np.stack([np.fliplr(im) for im in transformed_images])
                    kps_2d = transformed_keypoints.reshape(self.num_keypoints, 2).copy()
                    kps_2d[:, 0] = self.image_resize_width - kps_2d[:, 0]
                    kps_2d = kps_2d[self._hflip_swap_indices]
                    transformed_keypoints = kps_2d.reshape(-1)
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

        if self.visibility is not None:
            vis = self.visibility[idx]
            if do_hflip:
                vis = vis[torch.from_numpy(self._hflip_swap_indices)]
        else:
            vis = torch.zeros(0, dtype=torch.long)

        return BaseLabeledExampleDict(
            images=transformed_images,  # shape (3, img_height, img_width) or (5, 3, H, W)
            keypoints=torch.from_numpy(transformed_keypoints),  # shape (n_targets,)
            idxs=idx,
            bbox=bbox,
            visibility=vis,
        )


# the only addition here, should be the heatmap creation method.
class HeatmapDataset(BaseTrackingDataset):
    """Heatmap dataset that extends BaseTrackingDataset with 2D Gaussian heatmap targets.

    Inherits all image loading, keypoint parsing, imgaug pipeline, and hflip logic from
    :class:`BaseTrackingDataset`. The key addition is :meth:`compute_heatmap`, which converts
    (x, y) keypoint coordinates into ``(K, H, W)`` heatmap tensors used as supervision targets.
    Visibility synthesis also happens here: when the CSV lacks a ``visible`` column,
    ``self.visibility`` is populated from NaN positions using the ``uniform_heatmaps`` flag (see
    the visibility section of CLAUDE.md for the full mapping).

    ``__getitem__`` calls ``super().__getitem__()`` to obtain the base dict, then appends
    ``heatmaps`` and ``labeled_heatmaps`` keys before returning.
    """

    def __init__(
        self,
        root_directory: str | Path,
        csv_path: str,
        image_resize_height: int,
        image_resize_width: int,
        header_rows: list[int] | None = [0, 1, 2],
        imgaug_transform: iaa.Sequential | None = None,
        downsample_factor: Literal[1, 2, 3] = 2,
        do_context: bool = False,
        resize: bool = True,
        uniform_heatmaps: bool = False,
        bbox_path: str | None = None,
        imgaug_hflip: bool = False,
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
            uniform_heatmaps: when the CSV has no ``visible`` column, controls the target
                heatmap for NaN (unlabeled) keypoints. True generates a uniform heatmap
                (visibility=1, encourages low-confidence predictions); False generates an
                all-zero heatmap (visibility=0, excluded from loss). Ignored when the CSV
                provides explicit visibility flags.
            bbox_path: path to csv file that contains bounding box information; rows must be in
                same order as csv file
            imgaug_hflip: see :class:`BaseTrackingDataset` for full documentation.

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
            imgaug_hflip=imgaug_hflip,
        )

        if self.height % 128 != 0 or self.height % 128 != 0:
            logger.error(
                'image dimensions (after transformation) must be repeatably divisible by 2; '
                f'current dimensions: height={self.height}, width={self.width}'
            )
            exit()

        self.downsample_factor: Literal[1, 2, 3] = downsample_factor
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor
        self.num_targets = torch.numel(self.keypoints[0])
        self.num_keypoints = self.num_targets // 2

        # synthesize visibility for CSVs that don't provide a visible column
        if self.visibility is None:
            nan_mask = torch.isnan(self.keypoints[:, :, 0])
            vis_for_nan = 1 if uniform_heatmaps else 0
            self.visibility = torch.where(
                nan_mask,
                torch.full_like(nan_mask, vis_for_nan, dtype=torch.long),
                torch.full_like(nan_mask, 2, dtype=torch.long),
            )

    @property
    def output_shape(self) -> tuple:
        """Spatial shape of the heatmap output (height, width) after downsampling.

        Returns:
            Tuple of ``(heatmap_height, heatmap_width)``.
        """
        return (
            self.height // 2**self.downsample_factor,
            self.width // 2**self.downsample_factor,
        )

    def compute_heatmap(
        self,
        example_dict: BaseLabeledExampleDict,
        ignore_nans: bool = False,
    ) -> Float[torch.Tensor, "num_keypoints heatmap_height heatmap_width"]:
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

        vis = (
            example_dict["visibility"].unsqueeze(0)
            if example_dict["visibility"].numel() > 0
            else None
        )

        y_heatmap = generate_heatmaps(
            keypoints=keypoints.unsqueeze(0),  # add batch dim
            height=self.height,
            width=self.width,
            output_shape=self.output_shape,
            sigma=self.output_sigma,
            visibility=vis,
        )

        return y_heatmap[0]

    def compute_heatmaps(self) -> torch.Tensor:
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
        example_dict["heatmaps"] = self.compute_heatmap(example_dict, ignore_nans)  # type: ignore[typeddict-unknown-key]
        return example_dict  # type: ignore[return-value]


class MultiviewHeatmapDataset(torch.utils.data.Dataset):
    """Heatmap dataset that aggregates one :class:`HeatmapDataset` per camera view.

    Internally stores a ``dict[str, HeatmapDataset]`` at ``self.dataset``, keyed by view name.
    ``__getitem__`` calls each child dataset and stacks the results into a single
    :class:`~lightning_pose.data.datatypes.MultiviewHeatmapLabeledExampleDict`.

    The shared ``imgaug_transform`` and ``imgaug_hflip`` attributes on this class are replicated
    to each child dataset so that the data module can update them in one place. ``imgaug_hflip``
    is always ``False`` here (multiview hflip is not supported; setting it in the config raises a
    ``ValueError`` in the factory). The data module checks ``hasattr(dataset, 'dataset')`` to
    detect the multiview case and iterate over child datasets when stripping val/test
    augmentations.

    Does **not** inherit from :class:`BaseTrackingDataset`; augmentation routing is delegated
    entirely to the child :class:`HeatmapDataset` instances.
    """

    def __init__(
        self,
        root_directory: str | Path,
        csv_paths: list[str],
        view_names: list[str],
        image_resize_height: int,
        image_resize_width: int,
        header_rows: list[int] | None = [0, 1, 2],
        imgaug_transform: iaa.Sequential | None = None,
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
        logger.info('using MultiviewHeatmapDataset')
        self.imgaug_hflip = False  # not supported for multiview; sentinel for data module
        self.root_directory = root_directory
        self.csv_paths = csv_paths
        self.bbox_paths = bbox_paths or [None] * len(view_names)
        self.view_names = view_names
        self.image_resize_height = image_resize_height
        self.image_resize_width = image_resize_width
        self.do_context = do_context

        # do this here so resizing doesn't get added multiple times when iterating over views
        if resize:
            assert imgaug_transform is not None
            imgaug_transform.add(iaa.Resize({
                "height": image_resize_height,
                "width": image_resize_width,
            }))
        self.imgaug_transform = imgaug_transform

        self.downsample_factor: Literal[1, 2, 3] = downsample_factor
        self.dataset: dict[str, HeatmapDataset] = {}
        self.keypoint_names: dict[str, list[str]] = {}
        data_length_by_view: dict[str, int] = {}
        num_keypoints_by_view: dict[str, int] = {}
        for view, csv_path, bbox_path in zip(view_names, csv_paths, self.bbox_paths, strict=True):
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
            data_length_by_view[view] = len(self.dataset[view])
            num_keypoints_by_view[view] = self.dataset[view].num_keypoints

        # check if all csv files have the same number of columns
        self.num_keypoints = sum(num_keypoints_by_view.values())

        # check if all the data is in correct order; sets self.data_length
        self.check_data_images_names(data_length_by_view)

        self.num_targets = self.num_keypoints * 2

        if camera_params_path is not None:
            cam_params_df, cam_params_file_to_camgroup = self._load_cam_params_from_csv(
                camera_params_path,
            )
        else:
            cam_params_df, cam_params_file_to_camgroup = (
                self._discover_cam_params_from_image_paths()
            )
        self.cam_params_df = cam_params_df
        self.cam_params_file_to_camgroup = cam_params_file_to_camgroup

    def _load_camgroup(self, calib_file: str) -> CameraGroup:
        """Load and validate a CameraGroup from a calibration file.

        Args:
            calib_file: path to calibration toml, relative to self.root_directory

        Returns:
            loaded CameraGroup

        Raises:
            AssertionError: if camera names don't match self.view_names
        """
        camgroup = CameraGroup.load(os.path.join(self.root_directory, calib_file))
        cam_names = camgroup.get_names()
        assert np.all(cam_names == self.view_names), (
            "cfg.data.view_names must have same camera order as camera calibration file; "
            f"instead found {self.view_names} and {cam_names}."
        )
        return camgroup

    def _load_cam_params_from_csv(
        self,
        camera_params_path: str,
    ) -> tuple[pd.DataFrame, dict[str, CameraGroup]]:
        """Load per-frame camera calibration parameters from a CSV file.

        Args:
            camera_params_path: path to CSV mapping each frame to a calibration toml file

        Returns:
            tuple of (cam_params_df, cam_params_file_to_camgroup)
        """
        assert not self.do_context, "3D augmentations for context model not yet supported"
        cam_params_df = pd.read_csv(camera_params_path, index_col=0, header=[0])
        img_idxs_labels = [
            i.split('/')[-1] for i in self.dataset[self.view_names[0]].image_names
        ]
        img_idxs_calib = [i.split('/')[-1] for i in cam_params_df.index]
        assert np.all(img_idxs_labels == img_idxs_calib)
        cam_params_file_to_camgroup = {
            f: self._load_camgroup(f) for f in cam_params_df.file.unique()
        }
        return cam_params_df, cam_params_file_to_camgroup

    def _discover_cam_params_from_image_paths(
        self,
    ) -> tuple[pd.DataFrame | None, dict[str, CameraGroup] | None]:
        """Derive per-frame calibration from image paths when no CSV is provided.

        Expects each frame's path to follow labeled-data/<session>_<view>/img<frameidx>.ext.
        Tries calibrations/<session>.toml first, then calibration.toml at root_directory.

        Returns:
            tuple of (cam_params_df, cam_params_file_to_camgroup); both None if no calibration
            files are found
        """
        image_names = self.dataset[self.view_names[0]].image_names
        cam_params_file_to_camgroup = {}
        calib_files = []
        all_found = True

        for img_name in image_names:
            parts = Path(img_name).parts
            try:
                ld_idx = next(i for i, p in enumerate(parts) if p == 'labeled-data')
            except StopIteration as err:
                raise ValueError(
                    f"Image path '{img_name}' does not match expected pattern "
                    "labeled-data/<session>_<view>/img<frameidx>.ext"
                ) from err
            folder_name = parts[ld_idx + 1]
            if '_' not in folder_name:
                raise ValueError(
                    f"Folder '{folder_name}' in image path '{img_name}' does not match "
                    "expected pattern <session>_<view>"
                )
            session_id = folder_name.rsplit('_', 1)[0]

            calib_by_session = Path(self.root_directory) / 'calibrations' / f'{session_id}.toml'
            calib_fallback = Path(self.root_directory) / 'calibration.toml'
            if calib_by_session.exists():
                calib_file = str(Path('calibrations') / f'{session_id}.toml')
            elif calib_fallback.exists():
                calib_file = 'calibration.toml'
            else:
                all_found = False
                calib_files.append(None)
                continue

            calib_files.append(calib_file)
            if calib_file not in cam_params_file_to_camgroup:
                cam_params_file_to_camgroup[calib_file] = self._load_camgroup(calib_file)

        if cam_params_file_to_camgroup and all_found:
            assert not self.do_context, "3D augmentations for context model not yet supported"
            cam_params_df = pd.DataFrame(
                {'file': calib_files}, index=image_names,  # type: ignore[arg-type]
            )
            return cam_params_df, cam_params_file_to_camgroup

        if cam_params_file_to_camgroup and not all_found:
            logger.warning(
                'calibration file not found for some frames; disabling 3D for entire dataset'
            )
        return None, None

    def check_data_images_names(self, data_length_by_view: dict[str, int]) -> None:
        """Data checking
        Each object in self.datasets will have the attribute image_names
        (i.e. self.datasets['top'].image_names) since each values is a
        HeatmapDataset. Include a check to make sure that the image names
        are the same across all views, so that when it loads element n from
        each individual view we know these are properly matched.

        Args:
            data_length_by_view: number of labeled frames per view

        """
        # check if all CSV files have the same number of rows
        if len(set(data_length_by_view.values())) != 1:
            raise ImportError("the CSV files do not match in row numbers!")

        for key_num, keypoint in enumerate(self.keypoint_names[self.view_names[0]]):
            for view, keypointComp in self.keypoint_names.items():
                if keypoint != keypointComp[key_num]:
                    raise ImportError(f"the keypoints are not in correct order! \
                                      view: {self.view_names[0]} vs {view} | \
                                        {keypoint} != {keypointComp}")

        self.data_length = list(data_length_by_view.values())[0]
        for idx in range(self.data_length):
            img_file_names = set()
            for _view, heatmaps in self.dataset.items():
                img_file_names.add(Path(heatmaps.image_names[idx]).name)
                if len(img_file_names) > 1:
                    raise ImportError(
                        "Discrepancy in image file names across CSV files! "
                        "index:{idx}, image file names:{img_file_names}"
                    )

    @property
    def height(self) -> int:
        """Image height in pixels after resizing."""
        return self.image_resize_height

    @property
    def width(self) -> int:
        """Image width in pixels after resizing."""
        return self.image_resize_width

    def __len__(self) -> int:
        """Return the number of labeled examples in the dataset."""
        return self.data_length

    @property
    def output_shape(self) -> tuple:
        """Spatial shape of the heatmap output (height, width) after downsampling.

        Returns:
            Tuple of ``(heatmap_height, heatmap_width)``.
        """
        return (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )

    @property
    def num_views(self) -> int:
        """Number of camera views in this multiview dataset."""
        return len(self.view_names)

    @staticmethod
    def _scale_translate_keypoints(
        keypoints_3d: np.ndarray,
        scale_params: tuple = (0.8, 1.2),
        shift_param: float = 0.25,
    ) -> np.ndarray:
        """Apply scale and translation transforms to 3D keypoints.

        Args:
            keypoints_3d: original keypoints, shape (num_keypoints, 3)
            scale_params: (a, b) are (min, max) ratio of scaling
            shift_param: max shift in each dimension, as a fraction of the range (kp_max - kp_min)

        Returns:
            augmented keypoints, shape (num_keypoints, 3)
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

        Args:
            images: each element is a torch array of shape (3, height, width); h/w can differ
                across views
            keypoints_orig: shape (num_views, num_keypoints, 2)
            keypoints_aug: shape (num_views, num_keypoints, 2)
            bboxes: shape (num_view, 4) -> x, y, h, w

        Returns:
            each element corresponds to an augmented version of the input images; same device
        """

        device = images[0].device
        images_transformed = []

        for orig_img, kps_og, kps_aug, bbox in zip(
            images, keypoints_orig, keypoints_aug, bboxes, strict=True
        ):

            _, img_height, img_width = orig_img.shape

            # create a mask for valid keypoints (not NaN in either original or augmented)
            valid_mask = ~(np.isnan(kps_og).any(axis=1) | np.isnan(kps_aug).any(axis=1))

            # apply the same mask to both original and augmented keypoints
            orig_pts = kps_og[valid_mask]
            new_pts = kps_aug[valid_mask]

            # ensure we have enough points for transformation
            if len(orig_pts) < 3:
                raise RuntimeError(
                    "Fewer than 3 valid keypoints in 3d data augmentation; "
                    "this error should have been caught earlier!"
                )

            # transform data points from original coordinate space to frame coordinate
            orig_pts[:, 0] = (orig_pts[:, 0] - bbox[0]) / bbox[3] * img_width
            new_pts[:, 0] = (new_pts[:, 0] - bbox[0]) / bbox[3] * img_width
            orig_pts[:, 1] = (orig_pts[:, 1] - bbox[1]) / bbox[2] * img_height
            new_pts[:, 1] = (new_pts[:, 1] - bbox[1]) / bbox[2] * img_height

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
        """Extract 2D keypoints from a per-view example dict in absolute frame coordinates.

        Args:
            data_dict: mapping from view name to its labeled example dictionary.
            clone: if True, clone keypoint tensors before modification to avoid in-place changes.

        Returns:
            Array of shape ``(num_views, num_keypoints_per_view, 2)`` with (x, y) coordinates
            in the original (un-cropped) frame coordinate system.
        """
        num_keypoints = cast(int, self.num_keypoints)
        keypoints_2d = np.zeros((self.num_views, num_keypoints // self.num_views, 2))
        for idx_view, (_view, example_dict) in enumerate(data_dict.items()):
            if clone:
                keypoints_curr = example_dict["keypoints"].reshape(
                    num_keypoints // self.num_views, 2
                ).clone()
            else:
                keypoints_curr = example_dict["keypoints"].reshape(
                    num_keypoints // self.num_views, 2
                )
            # transform keypoints from bbox coordinates to absolute frame coordinates
            # 1. divide by image dims to get 0-1 normalized coords
            keypoints_curr[:, 0] = keypoints_curr[:, 0] / example_dict["images"].shape[-1]  # -1 x
            keypoints_curr[:, 1] = keypoints_curr[:, 1] / example_dict["images"].shape[-2]  # -2 y
            # 2. multiply and add by bbox dims
            keypoints_2d[idx_view] = norm_to_frame(
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

    def apply_3d_transforms(
        self,
        data_dict: dict,
        camgroup: CameraGroup,
        scale_params: tuple = (0.8, 1.2),
        shift_param: float = 0.25,
    ) -> tuple:
        """Apply 3D transforms to keypoint and image data (scale, translate)."""

        # extract keypoints and images from each view
        keypoints_2d = self._get_2d_keypoints_from_example_dict_absolute_coords(data_dict)
        images = []
        bboxes = []
        for _idx_view, (_view, example_dict) in enumerate(data_dict.items()):
            images.append(example_dict["images"])
            bboxes.append(example_dict["bbox"])

        num_keypoints = cast(int, self.num_keypoints)
        if np.all(np.isnan(keypoints_2d)):
            keypoints_3d_aug = np.nan * np.zeros((num_keypoints // self.num_views, 3))
            keypoints_2d_aug_resize = [
                torch.tensor(
                    np.nan * np.zeros(num_keypoints // self.num_views * 2),
                    dtype=example_dict["keypoints"].dtype,
                    device=example_dict["keypoints"].device,
                )
                for _, example_dict in data_dict.items()
            ]
            images_aug = [im.unsqueeze(0) for im in images]
        else:

            # triangulate keypoints (2D -> 3D)
            keypoints_3d = camgroup.triangulate_fast(keypoints_2d.copy())

            # check number of properly triangulated points
            valid_kps = np.sum(~(np.isnan(keypoints_3d).any(axis=1)))

            # if fewer than 3 valid triangulated keypoints, cannot perform augmentation
            if valid_kps < 3:

                # keep 3d keypoints the same
                keypoints_3d_aug = keypoints_3d.copy()

                # keep 2d keypoints the same
                keypoints_2d_aug = keypoints_2d.copy()

                # keep images the same
                images_aug = [im.unsqueeze(0) for im in images]

            else:

                # scale and translate keypoints in 3D
                keypoints_3d_aug = self._scale_translate_keypoints(
                    keypoints_3d, scale_params=scale_params, shift_param=shift_param,
                )

                # project 3D keypoints to 2D using the rotated cameras
                keypoints_2d_aug = camgroup.project(keypoints_3d_aug)

                # transform images to match keypoint augmentations
                images_aug = self._transform_images(
                    images=images,
                    keypoints_orig=keypoints_2d.copy(),
                    keypoints_aug=keypoints_2d_aug.copy(),
                    bboxes=[b.cpu().numpy() for b in bboxes],
                )

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
                visibility=data_dict[view]["visibility"],
            )
            example_dict["heatmaps"] = self.dataset[view].compute_heatmap(example_dict)  # type: ignore[typeddict-unknown-key]
            data_dict_aug[view] = example_dict

        return data_dict_aug, torch.tensor(keypoints_3d_aug)

    def fusion(self, datadict: dict) -> tuple[
        (
            Float[torch.Tensor, "num_views RGB image_height image_width"]
            | Float[torch.Tensor, "num_views frames RGB image_height image_width"]
        ),
        Float[torch.Tensor, "keypoints"],
        Float[torch.Tensor, "num_views heatmap_height heatmap_width"],
        Float[torch.Tensor, "num_views_x_xyhw"],
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
            assert self.cam_params_df is not None
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
                        np.nan * np.zeros((cast(int, self.num_keypoints) // self.num_views, 3)),
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
