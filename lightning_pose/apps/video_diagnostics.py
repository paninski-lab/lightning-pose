"""Analyze predictions on video data.

Refer to apps.md for information on how to use this file.

streamlit run video_diagnostics.py -- --model_dir "/home/zeus/content/Pose-app/data/demo/models"


"""

import argparse
import copy
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from lightning_pose.apps.plots import get_y_label, make_plotly_catplot, plot_precomputed_traces
from lightning_pose.apps.utils import (
    build_precomputed_metrics_df,
    concat_dfs,
    get_all_videos,
    get_col_names,
    get_model_folders,
    get_model_folders_vis,
    update_vid_metric_files_list,
)

catplot_options = ["boxen", "box", "violin", "strip", "hist"]
scale_options = ["linear", "log"]


def run():

    args = parser.parse_args()

    st.title("Video Diagnostics")

    # check if args.model_dir is a dir, if not, raise an error
    if args.make_dir:
        os.makedirs(args.model_dir, exist_ok=True)
    if not os.path.isdir(args.model_dir):
        st.text(
            f"--model_dir {args.model_dir} does not exist."
            f"\nPlease check the path and try again."
        )

    st.sidebar.header("Data Settings")

    # ----- selecting which models to use -----
    model_folders = get_model_folders(args.model_dir, require_predictions=False)

    # get the last two levels of each path to be presented to user
    model_folders_vis = get_model_folders_vis(model_folders)

    selected_models_vis = st.sidebar.multiselect("Select models", model_folders_vis, default=None)

    # append this to full path
    selected_models = [os.path.join(args.model_dir, f) for f in selected_models_vis]

    # ----- selecting videos to analyze -----
    all_videos_: list = get_all_videos(selected_models, video_subdir=args.video_subdir)

    # choose from the different videos that were predicted
    video_to_plot = st.sidebar.selectbox("Select a video:", [*all_videos_], key="video")

    if video_to_plot:
        prediction_files = update_vid_metric_files_list(
            video=video_to_plot,
            model_preds_folders=selected_models,
            video_subdir=args.video_subdir,
        )
    else:
        prediction_files = []

    if len(prediction_files) > 0:  # otherwise don't try to proceed

        model_names = copy.copy(selected_models_vis)

        # ---------------------------------------------------
        # load data
        # ---------------------------------------------------
        dframes_metrics = defaultdict(dict)
        dframes_traces = {}
        for p, model_pred_files in enumerate(prediction_files):
            # use provided names from cli
            if len(model_names) > 0:
                model_name = model_names[p]
                model_folder = selected_models[p]

            for model_pred_file in model_pred_files:
                model_pred_file_path = os.path.join(
                    model_folder, args.video_subdir, model_pred_file
                )
                if not isinstance(model_pred_file, Path):
                    model_pred_file.seek(0)  # reset buffer after reading
                if (
                    "pca" in str(model_pred_file)
                    or "temporal" in str(model_pred_file)
                    or "pixel" in str(model_pred_file)
                ):
                    dframe = pd.read_csv(model_pred_file_path, index_col=None)
                    dframes_metrics[model_name][str(model_pred_file)] = dframe
                else:
                    dframe = pd.read_csv(model_pred_file_path, header=[1, 2], index_col=0)
                    dframes_traces[model_name] = dframe
                    dframes_metrics[model_name]["confidence"] = dframe
                # data_types = dframe.iloc[:, -1].unique()

        # edit model names if desired, to simplify plotting
        st.sidebar.write("Model display names (editable)")
        new_names = []
        og_names = list(dframes_metrics.keys())
        for name in og_names:
            new_name = st.sidebar.text_input(
                label="name", value=name, label_visibility="hidden"
            )
            new_names.append(new_name)

        # change dframes key names to new ones
        for n_name, o_name in zip(new_names, og_names):
            dframes_metrics[n_name] = dframes_metrics.pop(o_name)
            dframes_traces[n_name] = dframes_traces.pop(o_name)

        # ---------------------------------------------------
        # compute metrics
        # ---------------------------------------------------

        # concat dataframes, collapsing hierarchy and making df fatter.
        df_concat, keypoint_names = concat_dfs(dframes_traces)
        df_metrics = build_precomputed_metrics_df(
            dframes=dframes_metrics, keypoint_names=keypoint_names
        )
        metric_options = list(df_metrics.keys())

        # ---------------------------------------------------
        # plot diagnostics
        # ---------------------------------------------------

        col00, col01, col02 = st.columns(3)

        with col00:
            # choose which metric to plot
            metric_to_plot = st.selectbox("Metric:", metric_options, key="metric")

        with col01:
            # plot diagnostic averaged overall all keypoints
            plot_type = st.selectbox("Plot style:", catplot_options, key="plot_type")

        with col02:
            plot_scale = st.radio(
                "Y-axis scale", scale_options, key="plot_scale", horizontal=True
            )

        x_label = "Model Name"
        y_label = get_y_label(metric_to_plot)
        log_y = False if plot_scale == "linear" else True

        # DB: commented out seaborn for visual coherence
        # fig_cat = make_seaborn_catplot(
        #     x="model_name", y="mean", data=df_metrics[metric_to_plot], log_y=log_y,
        #     x_label=x_label, y_label=y_label, title="Average over all keypoints",
        #     plot_type=plot_type
        # )
        # st.pyplot(fig_cat)

        # select keypoint to plot
        keypoint_to_plot = st.selectbox(
            "Select a keypoint:",
            pd.Series([*keypoint_names, "mean"]),
            key="keypoint_to_plot",
        )

        if plot_type != "hist":
            # show plot per keypoint
            plotly_flex_fig = make_plotly_catplot(
                x="model_name",
                y=keypoint_to_plot,
                data=df_metrics[metric_to_plot],
                x_label=x_label,
                y_label=y_label,
                title=keypoint_to_plot,
                plot_type=plot_type,
                log_y=log_y,
            )
        else:
            plotly_flex_fig = make_plotly_catplot(
                x=keypoint_to_plot,
                y=None,
                data=df_metrics[metric_to_plot],
                x_label=y_label,
                y_label="Frame count",
                title=keypoint_to_plot,
                plot_type="hist",
            )
        st.plotly_chart(plotly_flex_fig)

        # ---------------------------------------------------
        # plot traces
        # ---------------------------------------------------
        st.header("Trace diagnostics")

        col10, col11 = st.columns(2)

        with col10:
            models = st.multiselect(
                "Models:",
                pd.Series(list(dframes_metrics.keys())),
                default=list(dframes_metrics.keys()),
            )

        with col11:
            keypoint = st.selectbox("Keypoint:", pd.Series(keypoint_names))

        cols = get_col_names(keypoint, "x", models)
        fig_traces = plot_precomputed_traces(df_metrics, df_concat, cols)
        st.plotly_chart(fig_traces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=[])
    parser.add_argument("--video_subdir", type=str, default="video_preds")
    parser.add_argument("--make_dir", action="store_true", default=False)

    run()
