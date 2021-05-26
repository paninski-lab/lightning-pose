import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
from typing import Callable, Optional, Tuple, List
import os

class TrackingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_directory: str,
                 csv_path: str,
                 header_rows: Optional[List[int]] = None,
                 transform: Optional[Callable] = None
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
        csv_data = pd.read_csv(os.path.join(root_directory, csv_path), header=header_rows)
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
            'RGB')  # Rick's images have 1 color channel; change to 3.
        if self.transform:
            x = self.transform(x)

        # get labels from self.labels
        y = self.labels[idx]

        return x, y


class TrackingDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_batch_size, validation_batch_size, test_batch_size, num_workers):
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        data_len = self.fulldataset.__len__()
        self.train_set, self.valid_set, self.test_set = random_split(self.fulldataset,
                                                                     [round(data_len * 0.7), round(data_len * 0.1),
                                                                      round(data_len * 0.2)],
                                                                     generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.validation_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(
            self):  # TODO: change this, should go through the whole dataset, and maybe make an external function to work with an external dataset
        # return [pair[0] for pair in DataLoader(self.test_set, batch_size = self.validation_batch_size)]
        return DataLoader(self.test_set, batch_size=self.test_batch_size,
                          num_workers=0)  # set to 1 for testing purposes
