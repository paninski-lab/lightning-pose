import torch
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from typing import Callable, Optional, Tuple, List


class DGPDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_directory: str,
                 csv_path: str,
                 header_rows: Optional[List[int]] = None,
                 transform: Optional[Callable] = None
                 ) -> None:
        """
        Initializes the DGPDataset
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
        csv_data = pd.read_csv(root_directory + csv_path, header=header_rows)
        self.image_names = list(csv_data.iloc[:,0])
        self.labels = torch.tensor(csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float64)
        self.transform = transform
        self.root_directory = root_directory
        self.num_targets = self.labels.shape[1]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        # get img_name from self.image_names
        img_name = self.image_names[idx]
        # read image from file and apply transformations (if any)
        x = Image.open(self.root_directory + img_name).convert('RGB')
        if self.transform:
            x = self.transform(x)

        # get labels from self.labels
        y = self.labels[idx]

        return x, y
