"""Data modules split a dataset into train, val, and test modules."""

from nvidia.dali.plugin.pytorch import LastBatchPolicy
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from typeguard import typechecked
from typing import List, Literal, Optional, Tuple, Union

from lightning_pose.data.dali import video_pipe, LightningWrapper
from lightning_pose.data.utils import split_sizes_from_probabilities

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseDataModule(pl.LightningDataModule):
    """Splits a labeled dataset into train, val, and test data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[Union[float, int]] = None,
        torch_seed: int = 42,
    ) -> None:
        """Data module splits a dataset into train, val, and test data loaders.

        Args:
            dataset: base dataset to be split into train/val/test
            use_deterministic: TODO: use deterministic split of data...?
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed: control data splits

        """
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # maybe can make the view information more general when deciding on a
        # specific format for csv files
        self.use_deterministic = use_deterministic
        # info about dataset splits
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_dataset = None  # populated by self.setup()
        self.val_dataset = None  # populated by self.setup()
        self.test_dataset = None  # populated by self.setup()
        self.torch_seed = torch_seed

    def setup(self, stage: Optional[str] = None):  # stage arg needed for ptl

        datalen = self.dataset.__len__()
        print(
            "Number of labeled images in the full dataset (train+val+test): {}".format(
                datalen
            )
        )

        if self.use_deterministic:
            return

        # split data based on provided probabilities
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=self.train_probability,
            val_probability=self.val_probability,
            test_probability=self.test_probability,
        )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            data_splits_list,
            generator=torch.Generator().manual_seed(self.torch_seed),
        )

        # further subsample training data if desired
        if self.train_frames is not None:
            split = True
            if self.train_frames >= len(self.train_dataset):
                # take max number of train frames
                print(
                    "Warning! Requested training frames exceeds training "
                    + "set size; using all"
                )
                n_frames = len(self.train_dataset)
                split = False
            elif self.train_frames == 1:
                # assume this is a fraction; use full dataset
                n_frames = len(self.train_dataset)
                split = False
            elif self.train_frames > 1:
                # take this number of train frames
                n_frames = int(self.train_frames)
            elif self.train_frames > 0:
                # take this fraction of train frames
                n_frames = int(self.train_frames * len(self.train_dataset))
            else:
                raise ValueError("train_frames must be >0")
            if split:  # a second split
                self.train_dataset.indices = self.train_dataset.indices[
                    :n_frames
                ]  # this works well

        print(
            "Size of -- train set: {}, val set: {}, test set: {}".format(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )


class UnlabeledDataModule(BaseDataModule):
    """Data module that contains labeled and unlabled data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        video_paths_list: Union[List[str], str],
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[float] = None,
        unlabeled_batch_size: int = 1,
        unlabeled_sequence_length: int = 16,
        dali_seed: int = 123456,
        torch_seed: int = 42,
        device_id: int = 0,
    ) -> None:
        """Data module that contains labeled and unlabeled data loaders.

        Args:
            dataset: pytorch Dataset for labeled data
            video_paths_list: absolute paths of videos ("unlabeled" data)
            use_deterministic: TODO: use deterministic split of data...?
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed: control data splits
            unlabeled_batch_size: number of sequences to load per unlabeled
                batch
            unlabeled_sequence_length: number of frames per sequence of
                unlabeled data
            dali_seed: control randomness of unlabeled data loading
            torch_seed: control randomness of labeled data loading
            device_id: gpu for unlabeled data loading

        """
        super().__init__(
            dataset=dataset,
            use_deterministic=use_deterministic,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            train_probability=train_probability,
            val_probability=val_probability,
            test_probability=test_probability,
            train_frames=train_frames,
            torch_seed=torch_seed,
        )
        self.video_paths_list = video_paths_list
        self.num_workers_for_unlabeled = num_workers // 2
        self.num_workers_for_labeled = num_workers // 2
        assert unlabeled_batch_size == 1, "LightningWrapper expects batch size of 1"
        self.unlabeled_batch_size = unlabeled_batch_size
        self.unlabeled_sequence_length = unlabeled_sequence_length
        self.dali_seed = dali_seed
        self.torch_seed = torch_seed
        self.device_id = device_id
        self.unlabeled_dataloader = None  # initialized in setup_unlabeled
        super().setup()
        self.setup_unlabeled()

    def setup_unlabeled(self):

        from lightning_pose.data.utils import count_frames

        # get input data
        if isinstance(self.video_paths_list, list):
            # presumably a list of files
            filenames = self.video_paths_list
        elif isinstance(self.video_paths_list, str) and os.path.isfile(
            self.video_paths_list
        ):
            # single video file
            filenames = self.video_paths_list
        elif isinstance(self.video_paths_list, str) and os.path.isdir(
            self.video_paths_list
        ):
            # directory of videos
            import glob

            extensions = ["mp4"]  # allowed file extensions
            filenames = []
            for extension in extensions:
                filenames.extend(
                    glob.glob(os.path.join(self.video_paths_list, "*.%s" % extension))
                )
        else:
            raise ValueError(
                "`video_paths_list` must be a list of files, a single file, "
                + "or a directory name"
            )

        data_pipe = video_pipe(
            filenames=filenames,
            resize_dims=[self.dataset.height, self.dataset.width],
            random_shuffle=True,
            seed=self.dali_seed,
            sequence_length=self.unlabeled_sequence_length,
            batch_size=self.unlabeled_batch_size,
            num_threads=self.num_workers_for_unlabeled,
            device_id=self.device_id,
        )

        # compute number of batches
        total_frames = count_frames(filenames)  # sum across vids
        num_batches = int(
            total_frames // self.unlabeled_sequence_length
        )  # assuming batch_size==1

        self.unlabeled_dataloader = LightningWrapper(
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,  # TODO: auto reset on each epoch - is this random?
            num_batches=num_batches,
        )

    def train_dataloader(self):
        loader = {
            "labeled": DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers_for_labeled,
                persistent_workers=True,
            ),
            "unlabeled": self.unlabeled_dataloader,
        }
        return loader

    # TODO: check if necessary
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers_for_labeled,
        )
