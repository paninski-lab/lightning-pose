import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from typing import Callable, Optional, Tuple, List
import os
import numpy as np
from PIL import Image
from deepposekit.utils.keypoints import draw_keypoints
from tqdm import tqdm
from sklearn.decomposition import PCA
import h5py


class TrackingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_directory: str,
                 csv_path: str,
                 header_rows: Optional[List[int]] = None,
                 transform: Optional[Callable] = None
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
        csv_data = pd.read_csv(os.path.join(root_directory, csv_path), header=header_rows)
        self.image_names = list(csv_data.iloc[:, 0])
        self.labels = torch.tensor(csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
        self.transform = transform
        self.root_directory = root_directory
        self.num_targets = self.labels.shape[1]
    def __len__(self) -> int: #something is wrong here
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
 
class DLCHeatmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory: str,
        data_path: str,
        mode: str,
        header_rows: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        noNans: Optional[bool] = False
    ) -> None:
        """
        Initializes the DLC Heatmap Dataset
        Parameters:
            root_directory (str): path to data directory
            data_path (str): path to CSV or h5 file  (within root_directory). CSV file should be
                in the form (image_path, bodypart_1_x, bodypart_1_y, ..., bodypart_n_y)
                Note: image_path is relative to the given root_directory
            mode (str): 'csv' or 'h5'
            header_rows (List[int]): (optional) which rows in the csv are header rows
            transform (torchvision.transforms): (optional) transform to resize the images, image dimensions must be repeatably divisible by 2
            noNans (bool): whether or not to throw out all frames that have occluded keypoints
        Returns:
            None
        """
        self.root_directory = root_directory
        self.transform = transform
        if (mode == 'csv'):
            csv_data = pd.read_csv(
                os.path.join(root_directory, data_path), header=header_rows     
            )
            self.image_names = list(csv_data.iloc[:, 0])
            self.labels = torch.tensor(csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32)
            test_img = Image.open(os.path.join(self.root_directory, self.image_names[0])).convert(
                    "RGB" #didn't do this for DLC
            )  # Rick's images have 1 color channel; change to 3.
            
        elif (mode == 'h5'):
            hf = h5py.File(os.path.join(root_directory, data_path), 'r')
            self.images = np.array(hf['images'])
            self.images = self.images[:,:,:,0]
            self.labels = torch.tensor(hf["annotations"])
            test_img = Image.fromarray(self.images[idx]).convert("RGB")
            
        else:
            raise ValueError("mode must be 'csv' or 'h5'")

        self.labels = torch.reshape(self.labels, (self.labels.shape[0], -1, 2)) 
        print(test_img.size)
        test_label = self.labels[0]

        if self.transform:
            test_img_transformed, test_label_transformed = self.transform(images = np.expand_dims(test_img, axis = 0), keypoints = np.expand_dims(test_label, axis = 0))
            test_img_transformed = test_img_transformed.squeeze(0)
            test_label_transformed = test_label_transformed.squeeze(0)
        print(test_img_transformed.shape)
        self.height = test_img_transformed.shape[0] 
        self.width = test_img_transformed.shape[1]

        if (self.height % 128 != 0 or self.height % 128 != 0):
            print("image dimensions (after transformation) must be repeatably divisible by 2!")
            print("current image dimensions after transformation are:")
            print(test_img_transformed.shape[:2])
            exit()
   
        if noNans:
            #Checks for images with set of keypoints that include any nan, so that they can be excluded from the data entirely, like DeepPoseKit does
            ##########################################################
            self.fully_labeled_idxs = self.get_fully_labeled_idxs()
            print(self.fully_labeled_idxs)
            if (mode == 'csv'):
                self.image_names = [self.image_names[idx] for idx in self.fully_labeled_idxs]
            else:
                self.images = [self.images[idx] for idx in self.fully_labeled_idxs]
            #self.labels = [self.labels[idx] for idx in self.fully_labeled_idxs]
            self.labels = torch.index_select(self.labels, 0, self.fully_labeled_idxs)
            if (mode == 'csv'):
                print(len(self.image_names), len(self.labels))
            else:
                print(len(self.images), len(self.labels))
            self.labels = torch.tensor(self.labels)
            print(self.labels.shape)
            ##########################################################

        self.downsample_factor = 2 #could change to 0, 2, 3, or 4
        self.sigma = 5
        self.output_sigma = 1.25 #should be sigma/2 ^downsample factor
        self.output_shape = (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )
        self.half_output_shape = (int(self.output_shape[0] / 2), int(self.output_shape[1] / 2))
        print(self.half_output_shape)

        imgnet_mean = [0.485, 0.456, 0.406]
        imgnet_std = [0.229, 0.224, 0.225]
        self.torch_transform = transforms.Compose([ #imagenet normalization
            transforms.ToTensor(),
            transforms.Normalize(mean = imgnet_mean, std = imgnet_std)
        ])
        self.mode = mode
        # Compute heatmaps as preprocessing step
        #check that max of heatmaps look good
        self.compute_heatmaps()
        self.num_targets = self.labels[0].shape[0]
        print(self.num_targets)

    def compute_heatmaps(self):
        label_heatmaps = []
        for idx, y in enumerate(tqdm(self.labels)):
            if (self.mode == 'csv'):
                x = Image.open(os.path.join(self.root_directory, self.image_names[idx])).convert(
                    "RGB" #didn't do this for DLC
                )  # Rick's images have 1 color channel; change to 3.
            else:
                x = Image.fromarray(self.images[idx]).convert("RGB")
            if self.transform:
                x, y = self.transform(images = np.expand_dims(x, axis = 0), keypoints = np.expand_dims(y, axis = 0)) #check transform and normalization
                x = x.squeeze(0)
                y = y.squeeze(0)
            else:
                y = y.numpy()
            x = self.torch_transform(x)
            y_heatmap = draw_keypoints(y, x.shape[-2], x.shape[-1], self.half_output_shape, sigma = self.output_sigma) #output shape is smaller
            label_heatmaps.append(y_heatmap)
        self.label_heatmaps = torch.from_numpy(np.asarray(label_heatmaps)).float()
        self.label_heatmaps = self.label_heatmaps.permute(0, 3, 1, 2)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        
        # read image from file and apply transformations (if any)
        if (self.mode == 'csv'):
            # get img_name from self.image_names
            img_name = self.image_names[idx]
            x = Image.open(os.path.join(self.root_directory, img_name)).convert( #didn't do this for dlc
                "RGB"
            )  # Rick's images have 1 color channel; change to 3.
        else:
            x = Image.fromarray(self.images[idx]).convert("RGB") #make sure this works with the transformations

        if self.transform:
            x = self.transform(images = np.expand_dims(x, axis = 0)) #check this
            x = x.squeeze(0)
        x = self.torch_transform(x)
        y_heatmap = self.label_heatmaps[idx]
        #y_keypoint = self.labels[idx]
        return x, y_heatmap
        #return x, y_heatmap, y_keypoint

    def get_fully_labeled_idxs(self):
        nan_check = torch.isnan(self.labels)
        nan_check = nan_check[:,:,0]
        nan_check = ~nan_check
        annotated = torch.all(nan_check, dim = 1)
        annotated_index = torch.where(annotated)
        return annotated_index[0]

#taken from https://github.com/jgraving/DeepPoseKit/blob/master/deepposekit/utils/keypoints.py
def draw_keypoints(keypoints, height, width, output_shape, sigma=1, normalize=True):
    keypoints = keypoints.copy()
    n_keypoints = keypoints.shape[0]
    out_height = output_shape[0]
    out_width = output_shape[1]
    keypoints[:, 1] *= out_height / height
    keypoints[:, 0] *= out_width / width
    confidence = np.zeros((out_height, out_width, n_keypoints))
    xv = np.arange(out_width)
    yv = np.arange(out_height)
    xx, yy = np.meshgrid(xv, yv)
    for idx in range(n_keypoints):
        keypoint = keypoints[idx]
        if np.any(keypoint != keypoint): #keeps heatmaps with nans as all zeros
            continue
        gaussian = (yy - keypoint[1]) ** 2
        gaussian += (xx - keypoint[0]) ** 2
        gaussian *= -1
        gaussian /= 2 * sigma ** 2
        gaussian = np.exp(gaussian)
        confidence[..., idx] = gaussian
    if not normalize:
        confidence /= sigma * np.sqrt(2 * np.pi)
    return confidence

class TrackingDataModule(pl.LightningDataModule):
    def __init__(self, dataset, mode, train_batch_size, validation_batch_size, test_batch_size, num_workers):
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.num_views = 2 #changes with dataset, 2 for mouse, 3 for fish
        self.mode = mode
        #self.ppca_params = self.computePPCA_params(self.num_views)
    
    
    def setup(self, stage: Optional[str] = None): 
        datalen = self.fulldataset.__len__()
        print("datalen:")
        print(datalen)
        if (self.mode == 'deterministic'):
            return  

        if ((round(datalen * 0.8) + round(datalen * 0.1) + round(datalen * 0.1)) > datalen):
            self.train_set, self.valid_set, self.test_set = random_split(
                self.fulldataset, [round(datalen * 0.8) - 1, round(datalen * 0.1), round(datalen * 0.1)], #hardcoded solution to rounding error
                generator=torch.Generator().manual_seed(42)
            )
        elif ((round(datalen * 0.8) + round(datalen * 0.1) + round(datalen * 0.1)) < datalen):
            self.train_set, self.valid_set, self.test_set = random_split(
                self.fulldataset, [round(datalen * 0.8) + 1, round(datalen * 0.1), round(datalen * 0.1)], #hardcoded solution to rounding error
                generator=torch.Generator().manual_seed(42)
            )
        else:
            self.train_set, self.valid_set, self.test_set = random_split(
                self.fulldataset, [round(datalen * 0.8), round(datalen * 0.1), round(datalen * 0.1)],
                generator=torch.Generator().manual_seed(42)
            )

    def computePPCA_params(self):
        param_dict = {}
        data_arr = self.train_set.labels #won't work for random splitting
        print(data_arr.shape)
        num_body_parts = self.train_set.num_targets
        arr_for_pca = torch.reshape(data_arr, shape = (2 * self.num_views, num_body_parts * len(self.train_set)))
        pca = PCA(num_components = 6, svd_solver = 'full')
        pca.fit(arr_for_pca.T)
        mu = torch.mean(arr_for_pca, axis=1)
        print(mu.shape)
        print(mu)
        param_dict["obs_offset"] = mu
        exit()
        param_dict["top_3_eigenvectors"] 
        param_dict["bot_3_eigenvectors"] = e
        self.param_dict = param_dict
        #return param_dict
        

    def full_dataloader(self):
        return DataLoader(self.fulldataset, batch_size = 1, num_workers = self.num_workers)
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.validation_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):  
        return DataLoader(self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers)  

