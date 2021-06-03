import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from PIL import Image
from typing import Callable, Optional, Tuple, List
import os
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from tqdm import tqdm


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the Tracking Dataset
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
        csv_data = pd.read_csv(
            os.path.join(root_directory, csv_path), header=header_rows
        )
        self.image_names = list(csv_data.iloc[:, 0])
        self.labels = torch.tensor(csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
        self.transform = transform
        self.root_directory = root_directory
        self.num_targets = self.labels.shape[1]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        # get img_name from self.image_names
        img_name = self.image_names[idx]
        # read image from file and apply transformations (if any)
        x = Image.open(os.path.join(self.root_directory, img_name)).convert(
            "RGB"
        )  # Rick's images have 1 color channel; change to 3.
        if self.transform:
            x = self.transform(x)

        # get labels from self.labels
        y = self.labels[idx]

        return x, y


class HeatmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory: str,
        csv_path: str,
        header_rows: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the Tracking Dataset
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
        csv_data = pd.read_csv(
            os.path.join(root_directory, csv_path), header=header_rows
        )
        self.image_names = list(csv_data.iloc[:, 0])
        self.labels = torch.tensor(csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
        self.labels = torch.reshape(self.labels, (self.labels.shape[0], -1, 2))

        # Compute heatmaps as preprocessing step
        label_heatmaps = []
        for idx, y in enumerate(tqdm(self.labels)):
            x = Image.open(os.path.join(root_directory, self.image_names[idx])).convert(
                "RGB"
            )  # Rick's images have 1 color channel; change to 3.
            if transform:
                x = transform(x)
            y_heatmap = np.zeros((y.shape[0], x.shape[-2], x.shape[-1]))
            # TODO: vectorize this operation
            # TODO: Compute these in preprocessing rather than on the fly
            for bp_idx in range(y.shape[0]):
                if not np.any(np.isnan(y[bp_idx].detach().cpu().numpy())):
                    y_heatmap[bp_idx] = self.gaussian(
                        y_heatmap[bp_idx],
                        y[bp_idx].detach().cpu().numpy(),
                    )
            label_heatmaps.append(y_heatmap)
        self.label_heatmaps = torch.from_numpy(np.asarray(label_heatmaps)).float()

        self.transform = transform
        self.root_directory = root_directory
        self.num_targets = self.labels.shape[1]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        # get img_name from self.image_names
        img_name = self.image_names[idx]
        # read image from file and apply transformations (if any)
        x = Image.open(os.path.join(self.root_directory, img_name)).convert(
            "RGB"
        )  # Rick's images have 1 color channel; change to 3.
        if self.transform:
            x = self.transform(x)
        
        y_heatmap = self.label_heatmaps[idx]
        return x, y_heatmap

    # TODO: Add link for function
    def gaussian(self, img, pt, sigma=10):
        # Draw a 2D gaussian
        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

        if ul[0] > img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2

        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])
        img[img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
            g_y[0] : g_y[1], g_x[0] : g_x[1]
        ]
        return img


class TrackingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        train_batch_size,
        validation_batch_size,
        test_batch_size,
        num_workers,
    ):
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        datalen = self.fulldataset.__len__()
        self.train_set, self.valid_set, self.test_set = random_split(
            self.fulldataset,
            [round(datalen * 0.7), round(datalen * 0.1), round(datalen * 0.2)],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.validation_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(
        self,
    ):  # change this, should go through the whole dataset, and maybe make an external function to work with an external dataset
        # return [pair[0] for pair in DataLoader(self.test_set, batch_size = self.validation_batch_size)]
        return DataLoader(
            self.test_set, batch_size=self.test_batch_size, num_workers=0
        )  # set to 1 for testing purposes

