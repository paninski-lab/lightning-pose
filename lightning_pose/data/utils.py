"""Dataset/data module utilities."""

from typing import Any, List, Literal, Optional, Tuple, TypedDict, Union

import imgaug.augmenters as iaa
import lightning.pytorch as pl
import numpy as np
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torchtyping import TensorType
from typeguard import typechecked


# below are a bunch of classes that streamline data typechecking
class BaseLabeledExampleDict(TypedDict):
    """Return type when calling __getitem__() on BaseTrackingDataset."""

    images: Union[
        TensorType["RGB":3, "image_height", "image_width"],
        TensorType["frames", "RGB":3, "image_height", "image_width"],
    ]
    keypoints: TensorType["num_targets"]
    idxs: int


class HeatmapLabeledExampleDict(BaseLabeledExampleDict):
    """Return type when calling __getitem__() on HeatmapTrackingDataset."""

    heatmaps: TensorType["num_keypoints", "heatmap_height", "heatmap_width"]


class BaseLabeledBatchDict(TypedDict):
    """Batch type for base labeled data."""

    images: Union[
        TensorType["batch", "RGB":3, "image_height", "image_width", float],
        TensorType["batch", "frames", "RGB":3, "image_height", "image_width", float],
    ]
    keypoints: TensorType["batch", "num_targets", float]
    idxs: TensorType["batch", int]


class HeatmapLabeledBatchDict(BaseLabeledBatchDict):
    """Batch type for heatmap labeled data."""

    heatmaps: TensorType[
        "batch", "num_keypoints", "heatmap_height", "heatmap_width", float
    ]


class UnlabeledBatchDict(TypedDict):
    """Batch type for unlabeled data."""

    frames: Union[
        TensorType["seq_len", "RGB":3, "image_height", "image_width", float],
        TensorType[
            "seq_len", "context":5, "RGB":3, "image_height", "image_width", float
        ],
    ]
    transforms: Union[
        TensorType["seq_len", "h":2, "w":3, float],
        TensorType["h":2, "w":3, float],
        TensorType["seq_len", "null":1, float],
        TensorType["null":1, float],
        torch.Tensor,
    ]
    # transforms shapes
    # (seq_len, 2, 3): different transform for each sequence
    # (2, 3): same transform for all returned frames/keypoints
    # (seq_len, 1): no transforms
    # (1,): no transforms
    # torch.Tensor: necessary, getting error about torch.AnnotatedAlias that I don't understand


class SemiSupervisedBatchDict(TypedDict):
    """Batch type for base labeled+unlabeled data."""

    labeled: BaseLabeledBatchDict
    unlabeled: UnlabeledBatchDict


class SemiSupervisedHeatmapBatchDict(TypedDict):
    """Batch type for heatmap labeled+unlabeled data."""

    labeled: HeatmapLabeledBatchDict
    unlabeled: UnlabeledBatchDict


class SemiSupervisedDataLoaderDict(TypedDict):
    """Return type when calling train/val/test_dataloader() on semi-supervised models."""

    labeled: torch.utils.data.DataLoader
    unlabeled: DALIGenericIterator


class DataExtractor(object):
    """Helper class to extract all data from a data module."""

    def __init__(
        self,
        data_module: pl.LightningDataModule,
        cond: Literal["train", "test", "val"] = "train",
        extract_images: bool = False,
        remove_augmentations: bool = True,
    ) -> None:
        self.cond = cond
        self.extract_images = extract_images
        self.remove_augmentations = remove_augmentations

        if self.remove_augmentations:
            imgaug_curr = data_module.dataset.imgaug_transform
            if len(imgaug_curr) == 1 and isinstance(imgaug_curr[0], iaa.Resize):
                # current augmentation just resizes; keep this
                self.data_module = data_module
            else:
                from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
                from lightning_pose.data.datasets import BaseTrackingDataset, HeatmapDataset

                # make new augmentation pipeline that just resizes
                if not isinstance(imgaug_curr[-1], iaa.Resize):
                    # we currently assume the last transform is resizing
                    raise NotImplementedError
                # keep the resizing aug
                imgaug_new = iaa.Sequential([imgaug_curr[-1]])

                # rebuild dataset with new aug pipeline
                dataset_old = data_module.dataset
                if isinstance(data_module.dataset, HeatmapDataset):
                    dataset_new = HeatmapDataset(
                        root_directory=dataset_old.root_directory,
                        csv_path=dataset_old.csv_path,
                        imgaug_transform=imgaug_new,
                        downsample_factor=dataset_old.downsample_factor,
                        do_context=dataset_old.do_context,
                    )
                elif isinstance(dataset_old, BaseTrackingDataset):
                    dataset_new = BaseTrackingDataset(
                        root_directory=dataset_old.root_directory,
                        csv_path=dataset_old.csv_path,
                        imgaug_transform=imgaug_new,
                        do_context=dataset_old.do_context,
                    )

                # rebuild data_module with new dataset
                if isinstance(data_module, UnlabeledDataModule):
                    data_module_new = UnlabeledDataModule(
                        dataset=dataset_new,
                        video_paths_list=data_module.video_paths_list,
                        train_batch_size=data_module.train_batch_size,
                        val_batch_size=data_module.val_batch_size,
                        test_batch_size=data_module.test_batch_size,
                        num_workers=data_module.num_workers,
                        train_probability=data_module.train_probability,
                        val_probability=data_module.val_probability,
                        train_frames=data_module.train_frames,
                        dali_config=data_module.dali_config,
                        torch_seed=data_module.torch_seed,
                    )
                elif isinstance(data_module, BaseDataModule):
                    data_module_new = BaseDataModule(
                        dataset=dataset_new,
                        train_batch_size=data_module.train_batch_size,
                        val_batch_size=data_module.val_batch_size,
                        test_batch_size=data_module.test_batch_size,
                        num_workers=data_module.num_workers,
                        train_probability=data_module.train_probability,
                        val_probability=data_module.val_probability,
                        train_frames=data_module.train_frames,
                        torch_seed=data_module.torch_seed,
                    )
                    # data_module_new.setup()

                self.data_module = data_module_new

        else:
            self.data_module = data_module

    @property
    def dataset_length(self) -> int:
        name = "%s_dataset" % self.cond
        return len(getattr(self.data_module, name))

    def get_loader(
        self,
    ) -> Union[torch.utils.data.DataLoader, SemiSupervisedDataLoaderDict]:
        if self.cond == "train":
            return self.data_module.train_dataloader()
        if self.cond == "val":
            return self.data_module.val_dataloader()
        if self.cond == "test":
            return self.data_module.test_dataloader()

    @staticmethod
    def verify_labeled_loader(
        loader: Union[torch.utils.data.DataLoader, SemiSupervisedDataLoaderDict]
    ) -> torch.utils.data.DataLoader:
        if isinstance(loader, torch.utils.data.DataLoader):
            labeled_loader = loader
        else:
            # if we have a dictionary of dataloaders, we take the loader called
            # "labeled" (the loader called "unlabeled" doesn't have keypoints)
            labeled_loader = loader.iterables["labeled"]
        return labeled_loader

    def iterate_over_dataloader(
        self, loader: torch.utils.data.DataLoader
    ) -> Tuple[
        TensorType["num_examples", Any],
        Union[
            TensorType["num_examples", 3, "image_width", "image_height"],
            TensorType["num_examples", "frames", 3, "image_width", "image_height"],
            None,
        ],
    ]:
        keypoints_list = []
        images_list = []
        for ind, batch in enumerate(loader):
            keypoints_list.append(batch["keypoints"])
            if self.extract_images:
                images_list.append(batch["images"])
        concat_keypoints = torch.cat(keypoints_list, dim=0)
        if self.extract_images:
            concat_images = torch.cat(images_list, dim=0)
        else:
            concat_images = None
        # assert that indeed the number of columns does not change after concatenation,
        # and that the number of rows is the dataset length.
        assert concat_keypoints.shape == (
            self.dataset_length,
            keypoints_list[0].shape[1],
        )
        return concat_keypoints, concat_images

    def __call__(
        self,
    ) -> Tuple[
        TensorType["num_examples", Any],
        Union[
            TensorType["num_examples", 3, "image_width", "image_height"],
            TensorType["num_examples", "frames", 3, "image_width", "image_height"],
            None,
        ],
    ]:
        loader = self.get_loader()
        loader = self.verify_labeled_loader(loader)
        return self.iterate_over_dataloader(loader)


@typechecked
def split_sizes_from_probabilities(
    total_number: int,
    train_probability: float,
    val_probability: Optional[float] = None,
    test_probability: Optional[float] = None,
) -> List[int]:
    """Returns the number of examples for train, val and test given split probs.

    Args:
        total_number: total number of examples in dataset
        train_probability: fraction of examples used for training
        val_probability: fraction of examples used for validation
        test_probability: fraction of examples used for test. Defaults to None.
            Can be computed as the remaining examples.

    Returns:
        [num training examples, num validation examples, num test examples]

    """

    if test_probability is None and val_probability is None:
        remaining_probability = 1.0 - train_probability
        # round each to 5 decimal places (issue with floating point precision)
        val_probability = round(remaining_probability / 2, 5)
        test_probability = round(remaining_probability / 2, 5)
    elif test_probability is None:
        test_probability = 1.0 - train_probability - val_probability

    assert (
        test_probability + train_probability + val_probability == 1.0
    )  # probabilities should add to one
    train_number = int(np.floor(train_probability * total_number))
    val_number = int(np.floor(val_probability * total_number))
    test_number = (
        total_number - train_number - val_number
    )  # if we lose extra examples by flooring, send these to test_number
    assert (
        train_number + test_number + val_number == total_number
    )  # assert that we're using all datapoints
    return [train_number, val_number, test_number]


@typechecked
def clean_any_nans(data: torch.Tensor, dim: int) -> torch.Tensor:
    """Remove samples from a data array that contain nans."""
    # currently supports only 2D arrays
    nan_bool = (
        torch.sum(torch.isnan(data), dim=dim) > 0
    )  # e.g., when dim == 0, those columns (keypoints) that have >0 nans
    if dim == 0:
        return data[:, ~nan_bool]
    elif dim == 1:
        return data[~nan_bool]


@typechecked
def count_frames(video_list: Union[List[str], str]) -> int:
    """Simple function to count the number of frames in a video or a list of videos."""

    import cv2

    if isinstance(video_list, str):
        video_list = [video_list]
    num_frames = 0
    for video_file in video_list:
        cap = cv2.VideoCapture(video_file)
        num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return num_frames


@typechecked
def compute_num_train_frames(
    len_train_dataset: int,
    train_frames: Optional[Union[int, float]] = None,
) -> int:
    """Quickly compute number of training frames for a given dataset.

    Args:
        len_train_dataset: total number of frames in training dataset
        train_frames:
            <=1 - fraction of total train frames used for training
            >1 - number of total train frames used for training

    Returns:
    int
        total number of train frames

    """
    if train_frames is None:
        n_train_frames = len_train_dataset
    else:
        if train_frames >= len_train_dataset:
            # take max number of train frames
            print("Warning! Requested training frames exceeds training set size; using all")
            n_train_frames = len_train_dataset
        elif train_frames == 1:
            # assume this is a fraction; use full dataset
            n_train_frames = len_train_dataset
        elif train_frames > 1:
            # take this number of train frames
            n_train_frames = int(train_frames)
        elif train_frames > 0:
            # take this fraction of train frames
            n_train_frames = int(train_frames * len_train_dataset)
        else:
            raise ValueError("train_frames must be >0")

    return n_train_frames


# @typechecked
def generate_heatmaps(
    keypoints: TensorType["batch", "num_keypoints", 2],
    height: int,
    width: int,
    output_shape: Tuple[int, int],
    sigma: Union[float, int] = 1.25,
    uniform_heatmaps: bool = False,
) -> TensorType["batch", "num_keypoints", "height", "width"]:
    """Generate 2D Gaussian heatmaps from mean and sigma.

    Args:
        keypoints: coordinates that serve as mean of gaussian bump
        height: height of reshaped image (pixels, e.g., 128, 256, 512...)
        width: width of reshaped image (pixels, e.g., 128, 256, 512...)
        output_shape: dimensions of downsampled heatmap, (height, width)
        sigma: control spread of gaussian

    Returns:
        batch of 2D heatmaps

    """
    keypoints = keypoints.detach().clone()
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, :, 1] *= out_height / height
    keypoints[:, :, 0] *= out_width / width
    nan_idxs = torch.isnan(keypoints)[:, :, 0]
    xv = torch.arange(out_width, device=keypoints.device)
    yv = torch.arange(out_height, device=keypoints.device)
    # note flipped order because of pytorch's ij and numpy's xy indexing for meshgrid
    xx, yy = torch.meshgrid(yv, xv, indexing="ij")
    # adds batch and num_keypoints dimensions to grids
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)
    # adds dimension corresponding to the first dimension of the 2d grid
    keypoints = keypoints.unsqueeze(2)
    # evaluates 2d gaussian with mean equal to the keypoint and var equal to sigma^2
    heatmaps = (yy - keypoints[:, :, :, :1]) ** 2  # also flipped order here
    heatmaps += (xx - keypoints[:, :, :, 1:]) ** 2  # also flipped order here
    heatmaps *= -1
    heatmaps /= 2 * sigma**2
    heatmaps = torch.exp(heatmaps)
    # normalize all heatmaps to one
    heatmaps = heatmaps / torch.sum(heatmaps, dim=(2, 3), keepdim=True)
    # replace nans with zeros heatmaps
    # (all zeros heatmaps are ignored in the supervised heatmap loss)
    if uniform_heatmaps:
        filler_heatmap = torch.ones(
            (out_height, out_width), device=keypoints.device
        ) / (out_height * out_width)
    else:
        filler_heatmap = torch.zeros((out_height, out_width), device=keypoints.device)

    heatmaps[nan_idxs] = filler_heatmap
    return heatmaps


# @typechecked
def evaluate_heatmaps_at_location(
    heatmaps: TensorType["batch", "num_keypoints", "heatmap_height", "heatmap_width"],
    locs: TensorType["batch", "num_keypoints", 2],
    sigma: Union[float, int] = 1.25,  # sigma used for generating heatmaps
    num_stds: int = 2,  # num standard deviations of pixels to compute confidence
) -> TensorType["batch", "num_keypoints"]:
    """Evaluate 4D heatmaps using a 3D location tensor (last dim is x, y coords). Since
    the model outputs heatmaps with a standard deviation of sigma, confidence will be
    spread across neighboring pixels. To account for this, confidence is computed by
    taking all pixels within two standard deviations of the predicted pixel."""
    pix_to_consider = int(np.floor(sigma * num_stds))  # get all pixels within num_stds.
    num_pad = pix_to_consider
    heatmaps_padded = torch.zeros(
        (
            heatmaps.shape[0],
            heatmaps.shape[1],
            heatmaps.shape[2] + num_pad * 2,
            heatmaps.shape[3] + num_pad * 2,
        ),
        device=heatmaps.device,
    )
    heatmaps_padded[:, :, num_pad:-num_pad, num_pad:-num_pad] = heatmaps
    i = torch.arange(heatmaps_padded.shape[0], device=heatmaps_padded.device).reshape(
        -1, 1, 1, 1
    )
    j = torch.arange(heatmaps_padded.shape[1], device=heatmaps_padded.device).reshape(
        1, -1, 1, 1
    )
    k = locs[:, :, None, 1, None].type(torch.int64) + num_pad
    m = locs[:, :, 0, None, None].type(torch.int64) + num_pad
    offsets = list(np.arange(-pix_to_consider, pix_to_consider + 1))
    vals_all = []
    for offset in offsets:
        k_offset = k + offset
        for offset_2 in offsets:
            m_offset = m + offset_2
            # get rid of singleton dims
            vals = heatmaps_padded[i, j, k_offset, m_offset].squeeze(-1).squeeze(-1)
            vals_all.append(vals)
    vals = torch.stack(vals_all, 0).sum(0)
    return vals


# @typechecked
def undo_affine_transform(
    keypoints: TensorType["seq_len", "num_keypoints", 2],
    transform: Union[TensorType["seq_len", 2, 3], TensorType[2, 3]],
) -> TensorType["seq_len", "num_keypoints", 2]:
    # add 1s to get keypoints in projective geometry coords
    ones = torch.ones(
        (keypoints.shape[0], keypoints.shape[1], 1),
        dtype=keypoints.dtype,
        device=keypoints.device,
        requires_grad=True,
    )
    kps_aff = torch.concat([keypoints, ones], axis=2)

    mat = torch.clone(transform).detach()
    if len(transform.shape) == 2:
        # single transform for all frames; add batch dim
        mat = mat.unsqueeze(0)

    # create inverse matrices
    mats_inv_torch = []
    for idx in range(mat.shape[0]):
        mat_inv_ = torch.linalg.inv(mat[idx, :, :2])
        mat_inv = torch.concat(
            [mat_inv_, torch.matmul(-mat_inv_, mat[idx, :, -1, None])], dim=1
        )
        mats_inv_torch.append(
            torch.tensor(
                torch.transpose(mat_inv, 1, 0),
                dtype=keypoints.dtype,
                device=keypoints.device,
                requires_grad=True,
            )
        )

    # make a single block of inverse matrices
    if len(mats_inv_torch) == 1:
        # replicate this inverse matrix for each element of the batch
        mat_inv_torch = torch.tile(
            mats_inv_torch[0].unsqueeze(0), dims=(keypoints.shape[0], 1, 1)
        )
    else:
        # different transformation for each element of the batch
        mat_inv_torch = torch.stack(mats_inv_torch, dim=0)

    # apply inverse matrix to each element individually using batch matrix multiply
    kps_noaug = torch.bmm(kps_aff, mat_inv_torch)

    return kps_noaug
