import os
from typing import Dict, List, Optional

import fiftyone as fo
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from typeguard import typechecked

from lightning_pose.utils import pretty_print_str
from lightning_pose.utils.io import return_absolute_data_paths, return_absolute_path

# to ignore imports for sphix-autoapidoc
__all__ = [
    "check_lists_equal",
    "remove_string_w_substring_from_list",
    "check_dataset",
    "get_image_tags",
    "FiftyOneImagePlotter",
    "dfConverter",
]


@typechecked
def check_lists_equal(list_1: list, list_2: list) -> bool:
    return len(list_1) == len(list_2) and sorted(list_1) == sorted(list_2)


@typechecked
def remove_string_w_substring_from_list(strings: List[str], substring: str) -> List[str]:
    for s in strings:
        if substring in s:
            strings.remove(s)
    return strings


@typechecked
def check_dataset(dataset: fo.Dataset) -> None:
    pretty_print_str("Checking FiftyOne.Dataset by computing metadata... ")
    try:
        dataset.compute_metadata(skip_failures=False)
    except ValueError:
        print("Encountered error in metadata computation. See print:")
        print(dataset.exists("metadata", False))
        print("The above print should indicate bad image samples, e.g., with bad paths.")


@typechecked
def get_image_tags(pred_df: pd.DataFrame) -> pd.Series:
    # last column indicates if the image was used for training, testing, validation or
    # unused at all
    # zero -> unused, so explicitly replace
    # NOTE: can delete this at some point, pred_df now initialized w/ "unused"
    data_pt_tags = pred_df.iloc[:, -1].replace("0.0", "unused")
    return data_pt_tags


# #@typechecked
# force typechecking over the entire class. right now fails due to some
# list/listconfig issue
class FiftyOneImagePlotter:

    def __init__(
        self,
        cfg: DictConfig,
        keypoints_to_plot: Optional[List[str]] = None,
        csv_filename: str = "predictions.csv",
    ) -> None:

        self.cfg = cfg
        self.keypoints_to_plot = keypoints_to_plot
        self.dataset_name = self.cfg.eval.fiftyone.dataset_name
        self.data_dir, self.video_dir = return_absolute_data_paths(
            cfg.data, cfg.eval.fiftyone.get("n_dirs_back", 3))
        # hard-code this for now
        self.df_header_rows: List[int] = [1, 2]
        # ground_truth_df is not necessary but useful for keypoint names
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            df_tmp = []
            csv_files = [os.path.join(self.data_dir, f) for f in self.cfg.data.csv_file]
            for csv_file in csv_files:
                df_tmp.append(pd.read_csv(csv_file, header=self.df_header_rows))
            self.ground_truth_df = pd.concat(df_tmp)
        else:
            self.ground_truth_df: pd.DataFrame = pd.read_csv(
                os.path.join(self.data_dir, self.cfg.data.csv_file),
                header=self.df_header_rows,
            )
        if self.keypoints_to_plot is None:
            # plot all keypoints that appear in the ground-truth dataframe
            self.keypoints_to_plot: List[str] = list(self.ground_truth_df.columns.levels[0])
            # remove "bodyparts"
            if "bodyparts" in self.keypoints_to_plot:
                self.keypoints_to_plot.remove("bodyparts")
            # remove an "Unnamed" string if exists
            self.keypoints_to_plot = remove_string_w_substring_from_list(
                strings=self.keypoints_to_plot, substring="Unnamed"
            )
            print("Plotting: ", self.keypoints_to_plot)
            # make sure that bodyparts and unnamed arguments aren't there:

        # for faster fiftyone access, convert gt data to dict of dicts
        self.gt_data_dict: Dict[str, Dict[str, np.array]] = dfConverter(
            df=self.ground_truth_df, keypoint_names=self.keypoints_to_plot
        )()

        # get list of image paths
        relative_list = list(self.ground_truth_df.iloc[:, 0])
        self.image_paths = [os.path.join(self.data_dir, im_path) for im_path in relative_list]
        # assert that the images are indeed files
        for im in self.image_paths:
            if not os.path.isfile(im):
                raise FileNotFoundError(im)

        # collect predicted csv files; this will be a list of lists. The length of the first list
        # corresponds to the number of models, the length of the sublists corresponds to the number
        # of views
        model_abs_paths = self.get_model_abs_paths()
        if cfg.data.get("view_names", None) and len(cfg.data.view_names) > 1:
            self.pred_csv_files = []
            for model_dir in model_abs_paths:
                csv_list = [
                    os.path.join(model_dir, csv_filename.replace(".csv", f"_{v}.csv"))
                    for v in cfg.data.view_names
                ]
                self.pred_csv_files.append(csv_list)
        else:
            self.pred_csv_files = [
                [os.path.join(model_dir, csv_filename)] for model_dir in model_abs_paths
            ]

        # populate this variable after model predictions have been loaded
        self.data_tags = None

    @property
    def num_keypoints(self) -> int:
        return self.cfg.data.num_keypoints

    @property
    def model_names(self) -> List[str]:
        model_display_names = self.cfg.eval.fiftyone.model_display_names
        if model_display_names is None:  # model_0, model_1, ...
            model_display_names = [
                "model_%i" % i for i in range(len(self.pred_csv_files))
            ]
        return model_display_names

    def img_height_width(self, idx) -> int:
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        return image.height, image.width

    def dataset_info_print(self) -> str:
        # run after creating the dataset
        pretty_print_str(
            'Created FiftyOne dataset called: %s. To access it in python: fo.load_dataset("%s")'
            % (self.dataset_name, self.dataset_name)
        )

    def get_model_abs_paths(self) -> List[str]:
        model_maybe_relative_paths = self.cfg.eval.hydra_paths
        model_abs_paths = [
            return_absolute_path(m, n_dirs_back=2) for m in model_maybe_relative_paths
        ]
        # assert that the model folders exist
        for mod_path in model_abs_paths:
            assert os.path.isdir(mod_path)
        return model_abs_paths

    def get_gt_keypoints_list(self) -> List[fo.Keypoints]:
        # for each frame, extract ground-truth keypoint information
        print("Collecting ground-truth keypoints...")
        return self.get_keypoints_per_image(self.gt_data_dict)

    def load_model_predictions(self) -> None:
        # take the abs paths, and load the models into a dictionary
        self.model_preds_dict = {}
        self.preds_pandas_df_dict = {}
        for model_name, pred_csv_file_list in zip(self.model_names, self.pred_csv_files):
            # assuming that each path of saved logs has a predictions.csv file in it
            # always assume [1, 2] since our code generated the predictions
            temp_df = []
            for pred_csv_file in pred_csv_file_list:
                temp_df.append(pd.read_csv(pred_csv_file, header=[1, 2]))
            temp_df = pd.concat(temp_df)
            self.model_preds_dict[model_name] = dfConverter(temp_df, self.keypoints_to_plot)()
            self.preds_pandas_df_dict[model_name] = temp_df

    def build_single_frame_keypoints(
        self,
        data_dict: Dict[str, Dict[str, np.array]],
        frame_idx: int,
        height: int,
        width: int,
    ) -> List[fo.Keypoint]:
        # output: the positions of all keypoints in a single frame for a single model
        keypoints_list = []
        for kp_name in self.keypoints_to_plot:  # loop over names
            # write a single keypoint's position, confidence, and name
            keypoints_list.append(
                fo.Keypoint(
                    points=[
                        [
                            data_dict[kp_name]["coords"][frame_idx, 0] / width,
                            data_dict[kp_name]["coords"][frame_idx, 1] / height,
                        ]
                    ],
                    confidence=[data_dict[kp_name]["likelihood"][frame_idx]],
                    label=kp_name,  # sometimes plotted aggresively
                )
            )
        return keypoints_list

    def get_keypoints_per_image(
        self, data_dict: Dict[str, Dict[str, np.array]]
    ) -> List[fo.Keypoints]:
        """iterates over the rows of the dataframe and gathers keypoints in fiftyone format"""
        dataset_length = data_dict[self.keypoints_to_plot[0]]["coords"].shape[0]
        keypoints_list = []
        for img_idx in tqdm(range(dataset_length)):
            img_height, img_width = self.img_height_width(img_idx)
            single_frame_keypoints_list = self.build_single_frame_keypoints(
                data_dict=data_dict, frame_idx=img_idx, height=img_height, width=img_width,
            )
            keypoints_list.append(fo.Keypoints(keypoints=single_frame_keypoints_list))
        return keypoints_list

    def get_pred_keypoints_dict(self) -> Dict[str, List[fo.Keypoints]]:
        pred_keypoints_dict = {}
        # loop over the dictionary with predictions per model
        for model_name, model_dict in self.model_preds_dict.items():
            print("Collecting predicted keypoints for model: %s..." % model_name)
            pred_keypoints_dict[model_name] = self.get_keypoints_per_image(model_dict)

        return pred_keypoints_dict

    def create_dataset(self) -> fo.Dataset:
        samples = []
        # read each model's csv into a pandas dataframe
        self.load_model_predictions()
        # assumes that train,test,val split is identical for all the different models
        # may be different with ensembling
        self.data_tags = get_image_tags(self.preds_pandas_df_dict[self.model_names[0]]).values
        # build the ground-truth keypoints per image
        gt_keypoints_list = self.get_gt_keypoints_list()
        # do the same for each model's predictions (lists are stored in a dict)
        pred_keypoints_dict = self.get_pred_keypoints_dict()
        pretty_print_str("Appending fo.Keypoints to fo.Sample objects for each image...")
        for img_idx, img_path in enumerate(tqdm(self.image_paths)):
            # create a "sample" with an image and a tag (should be appended to self.samples)
            sample = fo.Sample(filepath=img_path, tags=[self.data_tags[img_idx]])
            # add ground truth keypoints to the sample (won't happen for video)
            sample["ground_truth"] = gt_keypoints_list[img_idx]  # previously created
            # add model-predicted keypoints to the sample
            for model_field_name, model_preds in pred_keypoints_dict.items():
                sample[model_field_name + "_preds"] = model_preds[img_idx]

            samples.append(sample)

        fiftyone_dataset = fo.Dataset(self.dataset_name, persistent=True)
        pretty_print_str("Adding samples to the dataset...")
        fiftyone_dataset.add_samples(samples)
        pretty_print_str("Done!")
        return fiftyone_dataset


# @typechecked
class dfConverter:

    def __init__(self, df: pd.DataFrame, keypoint_names: List[str]) -> None:
        self.df = df
        self.keypoint_names = keypoint_names

    def dict_per_bp(self, keypoint_name: str) -> Dict[str, np.array]:
        bp_df = self.df[keypoint_name]
        coords = bp_df[["x", "y"]].to_numpy()
        if "likelihood" in bp_df:
            likelihood = bp_df["likelihood"].to_numpy()
        else:
            likelihood = np.ones(shape=coords.shape[0])

        return {"coords": coords, "likelihood": likelihood}

    def __call__(self) -> Dict[str, Dict[str, np.array]]:
        full_dict = {}
        for kp_name in self.keypoint_names:
            full_dict[kp_name] = self.dict_per_bp(kp_name)

        return full_dict
