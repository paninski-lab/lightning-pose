import torch
import pandas as pd
from torch import cuda
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from typing import Callable, Optional, Tuple, List
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from pose_est_nets.utils.heatmap_tracker_utils import format_mouse_data
import h5py
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typeguard import typechecked
import sklearn

# set the random seed as input here.
# TODO: when moving to runs, we would like different random seeds, so consider eliminating.
# TODO: review the transforms -- resize is done by imgaug.augmenters coming from the main script. it is fed as input. internally, we always normalize to imagenet params.
TORCH_MANUAL_SEED = 42
_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# statistics of imagenet dataset on which the resnet was trained on
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# Dali parameters
_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
_SEQUENCE_LENGTH_UNSUPERVISED = 7
_INITIAL_PREFETCH_SIZE = 16
_BATCH_SIZE_UNSUPERVISED = 1  # sequence_length * batch_size = num_images passed
_DALI_RANDOM_SEED = 123456
video_directory = os.path.join(
    "/home/jovyan/mouseRunningData/unlabeled_videos"
)  # TODO: should go as input to the class.
assert os.path.isdir(video_directory)
video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]
num_processes = os.cpu_count()


@typechecked
def PCA_prints(pca: sklearn.decomposition._pca.PCA, components_to_keep: int) -> None:
    print("Results of running PCA on labels:")
    print(
        "explained_variance_ratio_: {}".format(
            np.round(pca.explained_variance_ratio_, 3)
        )
    )
    print(
        "total_explained_var: {}".format(
            np.round(np.sum(pca.explained_variance_ratio_[:components_to_keep]), 3)
        )
    )


@pipeline_def
def video_pipe(
    filenames: list, resize_dims: Optional[list]
):  # TODO: what does it return? more typechecking
    video = fn.readers.video(
        device=_DALI_DEVICE,
        filenames=filenames,
        sequence_length=_SEQUENCE_LENGTH_UNSUPERVISED,
        random_shuffle=True,
        initial_fill=_INITIAL_PREFETCH_SIZE,
        normalized=False,
        dtype=types.DALIDataType.FLOAT,
    )
    video = fn.resize(video, size=resize_dims)
    video = (
        video / 255.0
    )  # original videos (at least Rick's) range from 0-255. transform it to 0,1. # TODO: not sure that we need that, make sure it's the same as the supervised ones
    transform = fn.crop_mirror_normalize(
        video,
        output_layout="FCHW",
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
    )
    return transform


class DLCHeatmapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory: str,
        data_path: str,
        mode: str,
        header_rows: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        noNans: Optional[bool] = False,
        downsample_factor: Optional[int] = 2,
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
        if mode == "csv":
            csv_data = pd.read_csv(
                os.path.join(root_directory, data_path), header=header_rows
            )
            self.image_names = list(csv_data.iloc[:, 0])
            self.labels = torch.tensor(
                csv_data.iloc[:, 1:].to_numpy(), dtype=torch.float32
            )
            test_img = Image.open(
                os.path.join(self.root_directory, self.image_names[0])
            ).convert(
                "RGB"  # didn't do this for DLC
            )  # Rick's images have 1 color channel; change to 3.

        elif mode == "h5":
            hf = h5py.File(os.path.join(root_directory, data_path), "r")
            self.images = np.array(hf["images"])
            self.images = self.images[:, :, :, 0]
            self.labels = torch.tensor(hf["annotations"])
            test_img = Image.fromarray(self.images[0]).convert("RGB")

        else:
            raise ValueError("mode must be 'csv' or 'h5'")

        self.labels = torch.reshape(self.labels, (self.labels.shape[0], -1, 2))
        print(test_img.size)
        test_label = self.labels[0]
        # TODO: all the test images can be removed. add assertions.
        if self.transform:
            test_img_transformed, test_label_transformed = self.transform(
                images=np.expand_dims(test_img, axis=0),
                keypoints=np.expand_dims(test_label, axis=0),
            )
            test_img_transformed = test_img_transformed.squeeze(0)
            test_label_transformed = test_label_transformed.squeeze(0)
        # TODO: not great, remove
        print(test_img_transformed.shape)
        self.height = test_img_transformed.shape[0]
        self.width = test_img_transformed.shape[1]

        if self.height % 128 != 0 or self.height % 128 != 0:
            print(
                "image dimensions (after transformation) must be repeatably divisible by 2!"
            )
            print("current image dimensions after transformation are:")
            print(test_img_transformed.shape[:2])
            exit()

        if noNans:
            # Checks for images with set of keypoints that include any nan, so that they can be excluded from the data entirely, like DeepPoseKit does
            ##########################################################
            self.fully_labeled_idxs = self.get_fully_labeled_idxs()
            if mode == "csv":
                self.image_names = [
                    self.image_names[idx] for idx in self.fully_labeled_idxs
                ]
            else:
                self.images = [self.images[idx] for idx in self.fully_labeled_idxs]
            # self.labels = [self.labels[idx] for idx in self.fully_labeled_idxs]
            self.labels = torch.index_select(self.labels, 0, self.fully_labeled_idxs)
            if mode == "csv":
                print(len(self.image_names), len(self.labels))
            else:
                print(len(self.images), len(self.labels))
            self.labels = torch.tensor(self.labels)
            print(self.labels.shape)
            ##########################################################

        self.downsample_factor = downsample_factor
        self.sigma = 5
        self.output_sigma = 1.25  # should be sigma/2 ^downsample factor
        self.output_shape = (
            self.height // 2 ** self.downsample_factor,
            self.width // 2 ** self.downsample_factor,
        )
        # self.half_output_shape = (int(self.output_shape[0] / 2), int(self.output_shape[1] / 2))
        # print(self.half_output_shape)

        self.torch_transform = transforms.Compose(
            [  # imagenet normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        self.mode = mode
        # Compute heatmaps as preprocessing step
        # check that max of heatmaps look good
        self.compute_heatmaps()  # TODO: here we're computing the LABEL heatmaps which are saved to self. maybe explicitly have the outputs here
        self.num_targets = torch.numel(self.labels[0])
        print(self.num_targets)

    def compute_heatmaps(self):
        label_heatmaps = []
        for idx, y in enumerate(tqdm(self.labels)):
            if self.mode == "csv":
                x = Image.open(
                    os.path.join(self.root_directory, self.image_names[idx])
                ).convert(
                    "RGB"  # didn't do this for DLC
                )  # Rick's images have 1 color channel; change to 3.
            else:
                x = Image.fromarray(self.images[idx]).convert("RGB")
            if self.transform:
                x, y = self.transform(
                    images=np.expand_dims(x, axis=0),
                    keypoints=np.expand_dims(y, axis=0),
                )  # check transform and normalization
                x = x.squeeze(0)
                y = y.squeeze(0)
            else:
                y = y.numpy()
            x = self.torch_transform(x)
            y_heatmap = draw_keypoints(
                y, x.shape[-2], x.shape[-1], self.output_shape, sigma=self.output_sigma
            )
            label_heatmaps.append(y_heatmap)
        self.label_heatmaps = torch.from_numpy(np.asarray(label_heatmaps)).float()
        self.label_heatmaps = self.label_heatmaps.permute(0, 3, 1, 2)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:

        # read image from file and apply transformations (if any)
        if self.mode == "csv":
            # get img_name from self.image_names
            img_name = self.image_names[idx]
            x = Image.open(
                os.path.join(self.root_directory, img_name)
            ).convert(  # didn't do this for dlc
                "RGB"
            )  # Rick's images have 1 color channel; change to 3.
        else:
            x = Image.fromarray(self.images[idx]).convert(
                "RGB"
            )  # make sure this works with the transformations

        if self.transform:
            x = self.transform(
                images=np.expand_dims(x, axis=0)
            )  # TODO: check this. can be torch.unsqueeze(0)
            x = x.squeeze(0)
        x = self.torch_transform(x)
        y_heatmap = self.label_heatmaps[idx]
        # y_keypoint = self.labels[idx]
        return x, y_heatmap
        # return x, y_heatmap, y_keypoint

    def get_fully_labeled_idxs(self):  # TODO: make shorter
        nan_check = torch.isnan(self.labels)
        nan_check = nan_check[:, :, 0]
        nan_check = ~nan_check
        annotated = torch.all(nan_check, dim=1)
        annotated_index = torch.where(annotated)
        return annotated_index[0]


# taken from https://github.com/jgraving/DeepPoseKit/blob/master/deepposekit/utils/keypoints.py
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
        if np.any(keypoint != keypoint):  # keeps heatmaps with nans as all zeros
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


# TODO: let the unlabeled data module inherit from TrackingDataModule, just add the relevant components


class TrackingDataModule(pl.LightningDataModule):
    def __init__(  # TODO: add documentation and args
        self,
        dataset,
        mode,
        train_batch_size,
        validation_batch_size,
        test_batch_size,
        num_workers: Optional[int] = 8,
        use_unlabeled_frames: Optional[bool] = False,
        unlabeled_video_path: Optional[str] = None,
    ):
        super().__init__()
        self.fulldataset = dataset
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.num_views = 2  # changes with dataset, 2 for mouse, 3 for fish
        self.mode = mode
        self.use_unlabeled_frames = use_unlabeled_frames
        self.unlabeled_video_path = unlabeled_video_path

    def setup(self, stage: Optional[str] = None):  # TODO: clean up
        datalen = self.fulldataset.__len__()
        print("datalen:")
        print(datalen)
        if self.mode == "deterministic":
            return

        if (
            round(datalen * 0.8) + round(datalen * 0.1) + round(datalen * 0.1)
        ) > datalen:
            self.train_set, self.valid_set, self.test_set = random_split(
                self.fulldataset,
                [
                    round(datalen * 0.8) - 1,
                    round(datalen * 0.1),
                    round(datalen * 0.1),
                ],  # hardcoded solution to rounding error
                generator=torch.Generator().manual_seed(TORCH_MANUAL_SEED),
            )
        elif (
            round(datalen * 0.8) + round(datalen * 0.1) + round(datalen * 0.1)
        ) < datalen:
            self.train_set, self.valid_set, self.test_set = random_split(
                self.fulldataset,
                [
                    round(datalen * 0.8) + 1,
                    round(datalen * 0.1),
                    round(datalen * 0.1),
                ],  # hardcoded solution to rounding error
                generator=torch.Generator().manual_seed(TORCH_MANUAL_SEED),
            )
        else:
            self.train_set, self.valid_set, self.test_set = random_split(
                self.fulldataset,
                [round(datalen * 0.8), round(datalen * 0.1), round(datalen * 0.1)],
                generator=torch.Generator().manual_seed(TORCH_MANUAL_SEED),
            )
        # self.train_set = self.train_set.dataset
        # self.valid_set = self.valid_set.dataset
        # self.test_set = self.test_set.dataset
        print(len(self.train_set), len(self.valid_set), len(self.test_set))

    def setup_unlabeled(self, video_path):
        # device_id = self.local_rank
        # shard_id = self.global_rank
        # num_shards = self.trainer.world_size
        data_pipe = video_pipe(
            batch_size=_BATCH_SIZE_UNSUPERVISED,
            num_threads=self.num_workers,
            device_id=0,  # TODO: be careful when scaling to multinode
            resize_dims=[self.fulldataset.height, self.fulldataset.width],
            # shard_id=shard_id,
            # num_shards=num_shards,
            filenames=video_files,
            seed=_DALI_RANDOM_SEED,
        )

        class LightningWrapper(DALIGenericIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __len__(self):  # just to avoid ptl err check
                return 1  # num frames = len * batch_size; TODO: WHY 64? UNCLEAR

            def __next__(self):
                out = super().__next__()
                return torch.tensor(
                    # out[0]["x"][:, 0, :, :, :], dtype=torch.float  # , device="cuda"
                    out[0]["x"][
                        0, :, :, :, :
                    ],  # should be batch_size, W, H, 3. TODO: valid for one sequence.
                    dtype=torch.float,  # , device="cuda"
                )  # TODO: removed device. verify that it is not needed

        # data_pipe.build()
        # for i in range(startup_len):
        #     data_pipe.run()
        # print(data_pipe._api_type)
        # data_pipe._api_type = types.PipelineAPIType.ITERATOR
        # print(data_pipe._api_type)
        self.semi_supervised_loader = LightningWrapper(  # TODO: why write to self?
            data_pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )  # changed output_map to account for dummy labels

    # TODO: could be separated from this class
    # TODO: return something?
    def computePPCA_params(
        self,
        components_to_keep: Optional[int] = 3,
        empirical_epsilon_percentile: Optional[float] = 90.0,
    ) -> None:
        print("Computing PCA on the labels...")
        param_dict = {}
        # TODO: I don't follow the ifs, clarify with Nick
        if type(self.train_set) == torch.utils.data.dataset.Subset:
            indxs = torch.tensor(self.train_set.indices)
            data_arr = torch.index_select(self.train_set.dataset.labels, 0, indxs)
            num_body_parts = self.train_set.dataset.num_targets
        else:
            data_arr = self.train_set.labels  # won't work for random splitting
            num_body_parts = self.train_set.num_targets
        arr_for_pca = format_mouse_data(data_arr)
        pca = PCA(n_components=4, svd_solver="full")
        pca.fit(arr_for_pca.T)
        print("Done!")

        print(
            "arr_for_pca shape: {}".format(arr_for_pca.shape)
        )  # TODO: have prints as tests
        print("type of pca: {} ".format(type(pca)))

        PCA_prints(pca, components_to_keep)  # print important params
        # mu = torch.mean(arr_for_pca, axis=1) # TODO: needed only for probabilistic version
        # param_dict["obs_offset"] = mu  # TODO: needed only for probabilistic version
        param_dict["kept_eigenvectors"] = torch.tensor(
            pca.components_[:components_to_keep],
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )
        param_dict["discarded_eigenvectors"] = torch.tensor(
            pca.components_[components_to_keep:],
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )

        # absolute value is important -- projections can be negative.
        proj_discarded = torch.abs(
            torch.matmul(
                arr_for_pca.T,
                param_dict["discarded_eigenvectors"].clone().detach().cpu().T,
            )
        )
        print("proj_discarded.shape: {}".format(proj_discarded.shape))
        print("min of proj: {}".format(torch.min(proj_discarded)))
        print("max of proj: {}".format(torch.max(proj_discarded)))
        # TODO: generalize to more dims, we may have multiple discarded components.
        epsilon = np.percentile(
            proj_discarded.numpy(), empirical_epsilon_percentile, axis=0
        )
        param_dict["epsilon"] = torch.tensor(
            torch.tensor(epsilon),
            dtype=torch.float32,
            device=_TORCH_DEVICE,  # TODO: be careful for multinode
        )
        print("pre epsilon loss: {}".format(torch.linalg.norm(proj_discarded)))
        proj_discarded = proj_discarded.masked_fill(
            mask=proj_discarded > torch.tensor(epsilon), value=0.0
        )
        print("norm of proj: {}".format(torch.linalg.norm(proj_discarded)))

        print("post epsilon loss: {}".format(torch.linalg.norm(proj_discarded)))

        print("percentile: {}".format(epsilon))

        print("proj_discarded.shape: {}".format(proj_discarded.shape))

        self.pca_param_dict = param_dict

    def full_dataloader(self):
        return DataLoader(self.fulldataset, batch_size=1, num_workers=self.num_workers)

    def unlabeled_dataloader(self):
        return self.semi_supervised_loader

    ## That's the clean train_dataloader that works. can revert to it if needed
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_set,
    #         batch_size=self.train_batch_size,
    #         num_workers=self.num_workers,
    #     )

    def train_dataloader(  # TODO: verify that indeed the semi_supervised_loader does its job
        self,
    ):  # TODO: I don't like that the function returns a list or a dataloader.
        # if self.trainer.current_epoch % 2 == 0:
        #    return self.semi_supervised_loader
        # else:
        # return DataLoader(self.train_set, batch_size = self.train_batch_size, num_workers = self.num_workers)
        if self.use_unlabeled_frames:
            loader = {
                "labeled": DataLoader(
                    self.train_set,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers
                    // 2,  # TODO: keep track of num_workers
                ),
                "unlabeled": self.unlabeled_dataloader(),
            }
            return loader
        else:
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

    def predict_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.test_batch_size, num_workers=self.num_workers
        )
