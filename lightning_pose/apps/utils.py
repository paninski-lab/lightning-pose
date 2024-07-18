"""Utility functions for streamlit apps."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

pix_error_key = "pixel error"
conf_error_key = "confidence"
temp_norm_error_key = "temporal norm"
pcamv_error_key = "pca multiview"
pcasv_error_key = "pca singleview"


@st.cache_resource
def update_labeled_file_list(model_preds_folders: List[str], use_ood: bool = False) -> List[list]:
    per_model_preds = []
    for model_pred_folder in model_preds_folders:
        # pull labeled results from each model folder
        # wrap in Path so that it looks like an UploadedFile object
        model_preds = [
            f
            for f in os.listdir(model_pred_folder)
            if os.path.isfile(os.path.join(model_pred_folder, f))
        ]
        ret_files = []
        for file in model_preds:
            if "predictions" in file:
                if "new" not in file and not use_ood:
                    ret_files.append(Path(file))
                elif "new" in file and use_ood:
                    ret_files.append(Path(file))
                else:
                    pass
        per_model_preds.append(ret_files)
    return per_model_preds


@st.cache_resource
def update_vid_metric_files_list(
    video: str,
    model_preds_folders: List[str],
    video_subdir: str = "video_preds",
) -> List[list]:
    per_vid_preds = []
    for model_preds_folder in model_preds_folders:
        # pull each prediction file associated with a particular video
        # wrap in Path so that it looks like an UploadedFile object
        video_dir = os.path.join(model_preds_folder, video_subdir)
        if not os.path.isdir(video_dir):
            continue
        model_preds = [
            f for f in os.listdir(video_dir) if
            (os.path.isfile(os.path.join(video_dir, f)) and f.endswith(".csv"))
        ]
        ret_files = []
        for file in model_preds:
            if video in file:
                ret_files.append(Path(file))
        per_vid_preds.append(ret_files)
    return per_vid_preds


@st.cache_resource
def get_all_videos(model_preds_folders: List[str], video_subdir: str = "video_preds") -> list:
    # find each video that is predicted on by the models
    # wrap in Path so that it looks like an UploadedFile object
    # returned by streamlit's file_uploader
    ret_videos = set()
    for model_preds_folder in model_preds_folders:
        video_dir = os.path.join(model_preds_folder, video_subdir)
        if not os.path.isdir(video_dir):
            continue
        model_preds = [
            f for f in os.listdir(video_dir) if
            (os.path.isfile(os.path.join(video_dir, f)) and f.endswith(".csv"))
        ]
        for file in model_preds:
            if "temporal" in file:
                vid_file = file.split("_temporal_norm.csv")[0]
                ret_videos.add(vid_file)
            elif "temporal" not in file and "pca" not in file:
                vid_file = file.split(".csv")[0]
                ret_videos.add(vid_file)
    return list(ret_videos)


@st.cache_data
def concat_dfs(dframes: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    counter = 0
    for model_name, dframe in dframes.items():
        mask = dframe.columns.get_level_values("coords").isin(["x", "y", "likelihood"])
        if counter == 0:
            df_concat = dframe.loc[:, mask]
            # base_colnames = list(df_concat.columns.levels[0])  # <-- sorts names, bad!
            base_colnames = list([c[0] for c in df_concat.columns[1::3]])
            df_concat = strip_cols_append_name(df_concat, model_name)
        else:
            df = strip_cols_append_name(dframe.loc[:, mask], model_name)
            df_concat = pd.concat([df_concat, df], axis=1)
        counter += 1
    return df_concat, base_colnames


@st.cache_data
def get_df_box(df_orig: pd.DataFrame, keypoint_names: list, model_names: list) -> pd.DataFrame:
    df_boxes = []
    for keypoint in keypoint_names:
        for model_curr in model_names:
            tmp_dict = {
                "keypoint": keypoint,
                "metric": "Pixel error",
                "value": df_orig[df_orig.model_name == model_curr][keypoint],
                "model_name": model_curr,
            }
            df_boxes.append(pd.DataFrame(tmp_dict))
    return pd.concat(df_boxes)


@st.cache_data
def get_df_scatter(
    df_0: pd.DataFrame,
    df_1: pd.DataFrame,
    data_type: str,
    model_names: list,
    keypoint_names: list
) -> pd.DataFrame:
    df_scatters = []
    for keypoint in keypoint_names:
        df_scatters.append(
            pd.DataFrame(
                {
                    "img_file": df_0.index[df_0.set == data_type],
                    "keypoint": keypoint,
                    model_names[0]: df_0[keypoint][df_0.set == data_type],
                    model_names[1]: df_1[keypoint][df_1.set == data_type],
                }
            )
        )
    return pd.concat(df_scatters)


def get_col_names(keypoint: str, coordinate: str, models: List[str]) -> List[str]:
    return [get_full_name(keypoint, coordinate, model) for model in models]


def strip_cols_append_name(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col + "_" + name for col in df.columns.values]
    return df


def get_full_name(keypoint: str, coordinate: str, model: str) -> str:
    return "_".join([keypoint, coordinate, model])


# ----------------------------------------------
# compute metrics
# ----------------------------------------------
@st.cache_data
def build_precomputed_metrics_df(
    dframes: Dict[str, pd.DataFrame], keypoint_names: List[str], **kwargs,
) -> dict:
    concat_dfs = defaultdict(list)
    for model_name, df_dict in dframes.items():
        for metric_name, df in df_dict.items():
            if "confidence" in metric_name:
                df_ = compute_confidence(
                    df=df,
                    keypoint_names=keypoint_names,
                    model_name=model_name,
                    **kwargs
                )
                concat_dfs[conf_error_key].append(df_)

            df_ = get_precomputed_error(df, keypoint_names, model_name, **kwargs)
            if "single" in metric_name:
                concat_dfs[pcasv_error_key].append(df_)
            elif "multi" in metric_name:
                concat_dfs[pcamv_error_key].append(df_)
            elif "temporal" in metric_name:
                concat_dfs[temp_norm_error_key].append(df_)
            elif "pixel" in metric_name:
                concat_dfs[pix_error_key].append(df_)

    for key in concat_dfs.keys():
        concat_dfs[key] = pd.concat(concat_dfs[key])

    return concat_dfs


@st.cache_data
def get_precomputed_error(
    df: pd.DataFrame, keypoint_names: List[str], model_name: str,
) -> pd.DataFrame:
    # collect results
    df_ = df
    df_["model_name"] = model_name
    df_["mean"] = df_[keypoint_names[:-1]].mean(axis=1)

    return df_


@st.cache_data
def compute_confidence(
    df: pd.DataFrame, keypoint_names: List[str], model_name: str, **kwargs,
) -> pd.DataFrame:

    if df.shape[1] % 3 == 1:
        # collect "set" column if present
        set = df.iloc[:, -1].to_numpy()
    else:
        set = None

    mask = df.columns.get_level_values("coords").isin(["likelihood"])
    results = df.loc[:, mask].to_numpy()

    # collect results
    df_ = pd.DataFrame(columns=keypoint_names)
    for c, col in enumerate(keypoint_names):  # loop over keypoints
        df_[col] = results[:, c]
    df_["model_name"] = model_name
    df_["mean"] = df_[keypoint_names[:-1]].mean(axis=1)
    if set is not None:
        df_["set"] = set
        df_.index = df.index

    return df_


# ------------ utils related to model finding in dir ---------
def get_model_folders(
    model_dir: str,
    require_predictions: bool = True,
    require_tb_logs: bool = False,
) -> List[str]:
    """Find all model folders in a higher-level directory, conditional on directory contents."""
    # strip trailing slash if present
    if model_dir[-1] == os.sep:
        model_dir = model_dir[:-1]
    model_folders = []
    # find all directories two levels deep
    for root, dirs, files in os.walk(model_dir):
        if root.count(os.sep) - model_dir.count(os.sep) == 2:
            # only include directory if it has predictions.csv file (model training finished)
            if require_predictions or require_tb_logs:
                append = True
                if require_predictions and ("predictions.csv" not in os.listdir(root)):
                    append &= False
                if require_tb_logs and ("tb_logs" not in os.listdir(root)):
                    append &= False
                if append:
                    model_folders.append(root)
            else:
                model_folders.append(root)
    return model_folders


# just to get the last two levels of the path
def get_model_folders_vis(model_folders: List[str]) -> List[str]:
    fs = []
    for f in model_folders:
        fs.append(f.split("/")[-2:])
    model_folders_vis = [os.path.join(*f) for f in fs]
    return model_folders_vis
