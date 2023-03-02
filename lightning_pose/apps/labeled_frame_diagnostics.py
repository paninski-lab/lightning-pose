"""Analyze predictions on labeled frames.

Refer to apps.md for information on how to use this file.

"""

import argparse
import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict
import os

from lightning_pose.apps.utils import build_precomputed_metrics_df, concat_dfs, get_df_box, get_df_scatter
from lightning_pose.apps.utils import update_labeled_file_list
from lightning_pose.apps.plots import make_seaborn_catplot, make_plotly_scatterplot, get_y_label

st.session_state["n_submits"] = 0

catplot_options = ["boxen", "box", "bar", "violin", "strip"]
scale_options = ["linear", "log"]


def run():

    args = parser.parse_args()

    st.title("Labeled Frame Diagnostics")

    st.sidebar.header("Data Settings")

    prediction_files = update_labeled_file_list(args.model_folders)

    # col wrap when plotting results from all keypoints
    n_cols = 3

    if len(prediction_files) > 0:  # otherwise don't try to proceed

        # ---------------------------------------------------
        # load data
        # ---------------------------------------------------
        dframes_metrics = defaultdict(dict)
        dframes_traces = {}
        for p, model_pred_files in enumerate(prediction_files):
            # use provided names from cli
            if len(args.model_names) > 0:
                model_name = args.model_names[p]
                model_folder = args.model_folders[p]

            for model_pred_file in model_pred_files:
                model_pred_file_path = os.path.join(model_folder, model_pred_file)
                if not isinstance(model_pred_file, Path):
                    model_pred_file.seek(0)  # reset buffer after reading
                if 'pca' in str(model_pred_file) or 'temporal' in str(model_pred_file) or 'pixel' in str(model_pred_file):
                    dframe = pd.read_csv(model_pred_file_path, index_col=None)
                    dframes_metrics[model_name][str(model_pred_file)] = dframe
                else:
                    dframe = pd.read_csv(model_pred_file_path, header=[1, 2], index_col=0)
                    dframes_traces[model_name] = dframe
                    dframes_metrics[model_name]['confidence'] = dframe
                data_types = dframe.iloc[:, -1].unique()

        # edit model names if desired, to simplify plotting
        st.sidebar.write("Model display names (editable)")
        new_names = []
        og_names = list(dframes_metrics.keys())
        for name in og_names:
            new_name = st.sidebar.text_input(label="", value=name)
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
            dframes=dframes_metrics, keypoint_names=keypoint_names)
        metric_options = list(df_metrics.keys())

        # ---------------------------------------------------'
        # user options
        # ---------------------------------------------------
        st.header("Select data to plot")

        # choose from individual keypoints, their mean, or all at once
        keypoint_to_plot = st.selectbox(
            "Select a keypoint:", ["mean", "ALL", *keypoint_names], key="keypoint")

        # choose which metric to plot
        metric_to_plot = st.selectbox("Select a metric:", metric_options, key="metric")
        y_label = get_y_label(metric_to_plot)

        # choose data split - train/val/test/unused
        data_type = st.selectbox("Select data partition:", data_types, key="data partition")

        # ---------------------------------------------------
        # plot metrics for all modelsz
        # ---------------------------------------------------

        st.header("Compare multiple models")

        # enumerate plotting options
        plot_type = st.selectbox("Pick a plot type:", catplot_options)
        plot_epsilon = st.text_input("Epsilon", 0)
        plot_scale = st.radio("Select y-axis scale", scale_options)

        # filter data
        df_metrics_filt = df_metrics[metric_to_plot][df_metrics[metric_to_plot].set == data_type]
        n_frames_per_dtype = df_metrics_filt.shape[0] // len(args.model_folders)

        # plot data
        title = '%s (%i %s frames)' % (keypoint_to_plot, n_frames_per_dtype, data_type)

        log_y = False if plot_scale == "linear" else True

        if keypoint_to_plot == "ALL":

            df_box = get_df_box(df_metrics_filt, keypoint_names, new_names)
            sns.set_context("paper")
            fig_box = sns.catplot(
                x="model_name", y="value", col="keypoint", col_wrap=n_cols, sharey=False,
                kind=plot_type, data=df_box[df_box["value"] > int(plot_epsilon)], height=2)
            fig_box.set_axis_labels("Model Name", y_label)
            fig_box.set_xticklabels(rotation=45, ha="right")
            if len(keypoint_names) > 9:
                top = 0.94
            else:
                top = 0.9
            fig_box.fig.subplots_adjust(top=top)
            fig_box.fig.suptitle("All keypoints (%i %s frames)" % (n_frames_per_dtype, data_type))
            st.pyplot(fig_box)

        else:

            fig_box = make_seaborn_catplot(
                x="model_name", y=keypoint_to_plot,
                data=df_metrics_filt[df_metrics_filt[keypoint_to_plot] > int(plot_epsilon)],
                x_label="Model Name",
                y_label=y_label, title=title, log_y=log_y, plot_type=plot_type)
            st.pyplot(fig_box)

        # ---------------------------------------------------
        # scatterplots
        # ---------------------------------------------------

        st.header("Compare two models")
        model_0 = st.selectbox(
            "Model 0 (x-axis):", new_names, key="model_0")
        model_1 = st.selectbox(
            "Model 1 (y-axis):", [n for n in new_names if n != model_0], key="model_1")

        df_tmp0 = df_metrics[metric_to_plot][df_metrics[metric_to_plot].model_name == model_0]
        df_tmp1 = df_metrics[metric_to_plot][df_metrics[metric_to_plot].model_name == model_1]

        plot_scatter_scale = st.radio("Select axes scale", ["linear", "log"])

        if keypoint_to_plot == "ALL":

            df_scatter = get_df_scatter(
                df_tmp0, df_tmp1, data_type, [model_0, model_1], keypoint_names)
            fig_scatter = make_plotly_scatterplot(
                model_0=model_0, model_1=model_1, df=df_scatter,
                metric_name=y_label, title=title,
                axes_scale=plot_scatter_scale,
                facet_col="keypoint", n_cols=n_cols, hover_data=["img_file"],
                fig_height=300 * np.ceil(len(keypoint_names) / n_cols), fig_width=900,
            )

        else:

            df_scatter = pd.DataFrame({
                model_0: df_tmp0[keypoint_to_plot][df_tmp0.set == data_type],
                model_1: df_tmp1[keypoint_to_plot][df_tmp1.set == data_type],
                "img_file": df_tmp0.img_file[df_tmp0.set == data_type]
            })
            fig_scatter = make_plotly_scatterplot(
                model_0=model_0, model_1=model_1, df=df_scatter,
                metric_name=y_label, title=title,
                axes_scale=plot_scatter_scale,
                hover_data=["img_file"],
                fig_height=500, fig_width=500,
            )

        st.plotly_chart(fig_scatter)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_names', action='append', default=[])
    parser.add_argument('--model_folders', action='append', default=[])

    run()
