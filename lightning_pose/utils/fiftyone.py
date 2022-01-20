import fiftyone as fo
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
from lightning_pose.utils.io import return_absolute_path, return_absolute_data_paths
import os
from typeguard import typechecked

from lightning_pose.utils.plotting_utils import get_videos_in_dir


@typechecked
def check_lists_equal(list_1: list, list_2: list) -> bool:
    return len(list_1) == len(list_2) and sorted(list_1) == sorted(list_2)


@typechecked
def check_unique_tags(data_pt_tags: List[str]) -> bool:
    uniques = list(np.unique(data_pt_tags))
    cond_list = ["test", "train", "validation"]
    cond_list_with_unused_images = ["test", "train", "validation", "unused"]
    flag = check_lists_equal(uniques, cond_list) or check_lists_equal(
        uniques, cond_list_with_unused_images
    )
    return flag


@typechecked
def get_image_tags(pred_df: pd.DataFrame) -> pd.Series:
    # last column indicates if the image was used for training, testing, validation or unused at all
    # zero -> unused, so explicitly replace
    data_pt_tags = pred_df.iloc[:, -1].replace("0.0", "unused")
    assert check_unique_tags(data_pt_tags=list(data_pt_tags))
    return data_pt_tags


# @typechecked # force typechecking over the entire class. right now fails due to some list/listconfig issue
class FiftyOneKeypointBase:
    def __init__(
        self, cfg: DictConfig, keypoints_to_plot: Optional[List[str]] = None
    ) -> None:
        self.cfg = cfg
        self.keypoints_to_plot = keypoints_to_plot
        self.data_dir, self.video_dir = return_absolute_data_paths(cfg.data)
        self.df_header_rows: List[int] = OmegaConf.to_object(cfg.data.header_rows)
        # TODO: [0, 1] in toy dataset, [1,2] in actual ones, standardize
        # ground_truth_df is not necessary but useful for keypoint names
        self.ground_truth_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_dir, self.cfg.data.csv_file),
            header=self.df_header_rows,
        )
        if self.keypoints_to_plot is None:
            # plot all keypoints that appear in the ground-truth dataframe
            self.keypoints_to_plot: List[str] = list(
                self.ground_truth_df.columns.levels[0][1:]
            )

    @property
    def img_width(self) -> int:
        return self.cfg.data.image_orig_dims.width

    @property
    def img_height(self) -> int:
        return self.cfg.data.image_orig_dims.height

    @property
    def num_keypoints(self) -> int:
        return self.cfg.data.num_keypoints

    @property
    def model_names(self) -> List[str]:
        return self.cfg.eval.model_display_names

    @property
    def dataset_name(self) -> str:
        return self.cfg.eval.fifty_one_dataset_name

    def get_model_abs_paths(self) -> List[str]:
        model_maybe_relative_paths = self.cfg.eval.hydra_paths
        model_abs_paths = [
            return_absolute_path(m, n_dirs_back=2) for m in model_maybe_relative_paths
        ]
        # assert that the model folders exist
        for mod_path in model_abs_paths:
            assert os.path.isdir(mod_path)
        return model_abs_paths

    def load_model_predictions(self) -> None:
        # TODO: we have to specify the paths differently in the init method?
        # take the abs paths, and load the models into a dictionary
        model_abs_paths = self.get_model_abs_paths()
        self.model_preds_dict = {}
        for model_idx, model_dir in enumerate(model_abs_paths):
            # assuming that each path of saved logs has a predictions.csv file in it
            self.model_preds_dict[self.model_names[model_idx]] = pd.read_csv(
                os.path.join(model_dir, "predictions.csv"), header=self.df_header_rows
            )

    @typechecked
    def build_single_frame_keypoint_list(
        self,
        df: pd.DataFrame,
        frame_idx: int,
    ) -> List[fo.Keypoint]:
        # the output of this, is a the positions of all keypoints in a single frame for a single model.
        keypoints_list = []
        for kp_name in self.keypoints_to_plot:  # loop over names
            if "likelihood" in df[kp_name]:
                confidence = df[kp_name]["likelihood"][frame_idx]
            else:  # gt data has no confidence, but we call it 1.0 for simplicity
                confidence = 1.0  # also works if we make it None
            # "bodyparts" it appears in the csv as we read it right now, but should be ignored
            if kp_name == "bodyparts":
                continue
            # write a single keypoint's position, confidence, and name
            keypoints_list.append(
                fo.Keypoint(
                    points=[
                        [
                            df[kp_name]["x"][frame_idx] / self.img_width,
                            df[kp_name]["y"][frame_idx] / self.img_height,
                        ]
                    ],
                    confidence=confidence,
                    label=kp_name,  # sometimes plotted aggresively
                )
            )
        return keypoints_list

    @typechecked
    def get_keypoints_per_image(self, df: pd.DataFrame) -> List[fo.Keypoints]:
        """iterates over the rows of the dataframe and gathers keypoints in fiftyone format"""
        keypoints_list = []
        for img_idx in tqdm(range(df.shape[0])):
            single_frame_keypoints_list = self.build_single_frame_keypoint_list(
                df=df, frame_idx=img_idx
            )
            keypoints_list.append(fo.Keypoints(keypoints=single_frame_keypoints_list))
        return keypoints_list

    @typechecked
    def get_pred_keypoints_dict(self) -> Dict[str, List[fo.Keypoints]]:
        pred_keypoints_dict = {}
        # loop over the dictionary with predictions per model
        for model_name, model_df in self.model_preds_dict.items():
            print("Collecting predicted keypoints for model: %s..." % model_name)
            pred_keypoints_dict[model_name] = self.get_keypoints_per_image(model_df)

        return pred_keypoints_dict

    def create_dataset(self):
        # subclasses build their own
        raise NotImplementedError


class FiftyOneImagePlotter(FiftyOneKeypointBase):
    def __init__(
        self, cfg: DictConfig, keypoints_to_plot: Optional[List[str]] = None
    ) -> None:
        super().__init__(cfg=cfg, keypoints_to_plot=keypoints_to_plot)

    @property
    def image_paths(self) -> List[str]:
        """extract absolute paths for all the images in the ground truth csv file

        Returns:
            List[str]: absolute paths per image, checked before returning.
        """
        relative_list = list(self.ground_truth_df.iloc[:, 0])
        absolute_list = [
            os.path.join(self.data_dir, im_path) for im_path in relative_list
        ]
        # assert that the images are indeed files
        for im in absolute_list:
            assert os.path.isfile(im)

        return absolute_list

    @typechecked
    def get_gt_keypoints_list(self) -> List[fo.Keypoints]:
        # for each frame, extract ground-truth keypoint information
        print("Collecting ground-truth keypoints...")
        return self.get_keypoints_per_image(self.ground_truth_df)

    @typechecked
    def create_dataset(self) -> fo.Dataset:
        samples = []
        # read each model's csv into a pandas dataframe
        self.load_model_predictions()
        # assumes that train,test,val split is identical for all the different models. may be different with ensembling.
        self.data_tags = get_image_tags(self.model_preds_dict[self.model_names[0]])
        # build the ground-truth keypoints per image
        gt_keypoints_list = self.get_gt_keypoints_list()
        # do the same for each model's predictions (lists are stored in a dict)
        pred_keypoints_dict = self.get_pred_keypoints_dict()
        for img_idx, img_path in enumerate(tqdm(self.image_paths)):
            # create a "sample" with an image and a tag (should be appended to self.samples)
            sample = fo.Sample(filepath=img_path, tags=[self.data_tags[img_idx]])
            # add ground truth keypoints to the sample (won't happen for video)
            sample["ground_truth"] = gt_keypoints_list[img_idx]  # previously created
            # add model-predicted keypoints the sample
            for model_field_name, model_preds in pred_keypoints_dict.items():
                sample[model_field_name + "_preds"] = model_preds[img_idx]

            samples.append(sample)

        fiftyone_dataset = fo.Dataset(self.dataset_name)
        fiftyone_dataset.add_samples(samples)
        return fiftyone_dataset


""" 
what's shared between the two?
certain properties of the image; keypoint names; obtaining of csvs
creation of dataset.

different: 
in video each sample is a video. there is basically one sample if we analyze one video.
should also use get_pred_keypoints_dict (assuming that the preds for a new vid look the same as the ones in train hydra)
"""


class FiftyOneKeypointVideoPlotter(FiftyOneKeypointBase):
    def __init__(
        self, cfg: DictConfig, keypoints_to_plot: Optional[List[str]] = None
    ) -> None:
        super().__init__(cfg=cfg, keypoints_to_plot=keypoints_to_plot)

        self.new_videos = get_videos_in_dir(cfg.eval.path_to_test_videos[0])
