"""A collection of visualizations for various pose estimation performance metrics."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

pix_error_key = "pixel error"
conf_error_key = "confidence"
temp_norm_error_key = "temporal norm"
pcamv_error_key = "pca multiview"
pcasv_error_key = "pca singleview"


def get_y_label(to_compute: str) -> str:
    if (
        to_compute == "rmse"
        or to_compute == "pixel_error"
        or to_compute == "pixel error"
    ):
        return "Pixel Error"
    if to_compute == "temporal_norm" or to_compute == "temporal norm":
        return "Temporal norm (pix.)"
    elif to_compute == "pca_multiview" or to_compute == "pca multiview":
        return "Multiview PCA\nrecon error (pix.)"
    elif to_compute == "pca_singleview" or to_compute == "pca singleview":
        return "Low-dimensional PCA\nrecon error (pix.)"
    elif to_compute == "conf" or to_compute == "confidence":
        return "Confidence"


def make_seaborn_catplot(
    x, y, data, x_label, y_label, title, log_y=False, plot_type="box", figsize=(5, 5)
):
    sns.set_context("paper")
    fig = plt.figure(figsize=figsize)
    if plot_type == "box":
        sns.boxplot(x=x, y=y, data=data)
    elif plot_type == "boxen":
        sns.boxenplot(x=x, y=y, data=data)
    elif plot_type == "bar":
        sns.barplot(x=x, y=y, data=data)
    elif plot_type == "violin":
        sns.violinplot(x=x, y=y, data=data)
    elif plot_type == "strip":
        sns.stripplot(x=x, y=y, data=data)
    else:
        raise NotImplementedError
    ax = fig.gca()
    ax.set_yscale("log") if log_y else ax.set_yscale("linear")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title)
    return fig


def make_plotly_catplot(
    x,
    y,
    data,
    x_label,
    y_label,
    title,
    log_y=False,
    plot_type="box",
    fig_height=500,
    fig_width=500,
):
    if plot_type == "box" or plot_type == "boxen":
        fig = px.box(data, x=x, y=y, log_y=log_y)
    elif plot_type == "violin":
        fig = px.violin(data, x=x, y=y, log_y=log_y, box=True)
    elif plot_type == "strip":
        fig = px.strip(data, x=x, y=y, log_y=log_y)
    # elif plot_type == "bar":
    #     fig = px.bar(data, x=x, y=y, log_y=log_y)
    elif plot_type == "hist":
        fig = px.histogram(
            data,
            x=x,
            color="model_name",
            marginal="rug",
            barmode="overlay",
        )
    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title=x_label,
        title=title,
        height=fig_height,
        width=fig_width,
    )

    return fig


def make_plotly_scatterplot(
    model_0,
    model_1,
    df,
    metric_name,
    title,
    axes_scale="linear",
    facet_col=None,
    n_cols=0,
    opacity=0.5,
    hover_data=None,
    fig_height=500,
    fig_width=500,
):
    xlabel = "%s<br>(%s)" % (metric_name, model_0)
    ylabel = "%s<br>(%s)" % (metric_name, model_1)

    log_scatter = False if axes_scale == "linear" else True

    fig_scatter = px.scatter(
        df,
        x=model_0,
        y=model_1,
        facet_col=facet_col,
        facet_col_wrap=n_cols,
        log_x=log_scatter,
        log_y=log_scatter,
        opacity=opacity,
        hover_data=hover_data,
        # trendline="ols",
        title=title,
        labels={model_0: xlabel, model_1: ylabel},
    )

    mn = np.min(df[[model_0, model_1]].min(skipna=True).to_numpy())
    mx = np.max(df[[model_0, model_1]].max(skipna=True).to_numpy())
    trace = go.Scatter(x=[mn, mx], y=[mn, mx], line_color="black", mode="lines")
    trace.update(legendgroup="trendline", showlegend=False)
    fig_scatter.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)
    fig_scatter.update_layout(title=title, width=fig_width, height=fig_height)
    fig_scatter.update_traces(marker={"size": 5})

    return fig_scatter


def plot_precomputed_traces(df_metrics, df_traces, cols):
    # -------------------------------------------------------------
    # setup
    # -------------------------------------------------------------
    coordinate = "x"  # placeholder
    keypoint = cols[0].split("_%s_" % coordinate)[0]
    colors = px.colors.qualitative.Plotly

    rows = 3
    row_heights = [2, 2, 0.75]
    if temp_norm_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)
    if pcamv_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)
    if pcasv_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)

    fig_traces = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        x_title="Frame number",
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    yaxis_labels = {}
    row = 1

    # -------------------------------------------------------------
    # plot temporal norms, pcamv reproj errors, pcasv reproj errors
    # -------------------------------------------------------------
    for error_key in [temp_norm_error_key, pcamv_error_key, pcasv_error_key]:
        if error_key in df_metrics.keys():
            for c, col in enumerate(cols):
                # col = <keypoint>_<coord>_<model_name>.csv
                pieces = col.split("_%s_" % coordinate)
                if len(pieces) != 2:
                    # otherwise "_[x/y]_" appears in keypoint or model name :(
                    raise ValueError("invalid column name %s" % col)
                kp = pieces[0]
                model = pieces[1]
                fig_traces.add_trace(
                    go.Scatter(
                        name=col,
                        x=np.arange(df_traces.shape[0]),
                        y=df_metrics[error_key][kp][
                            df_metrics[error_key].model_name == model
                        ],
                        mode="lines",
                        line=dict(color=colors[c]),
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            if error_key == temp_norm_error_key:
                yaxis_labels["yaxis%i" % row] = "temporal<br>norm"
            elif error_key == pcamv_error_key:
                yaxis_labels["yaxis%i" % row] = "pca multi<br>error"
            elif error_key == pcasv_error_key:
                yaxis_labels["yaxis%i" % row] = "pca single<br>error"
            row += 1

    # -------------------------------------------------------------
    # plot traces
    # -------------------------------------------------------------
    for coord in ["x", "y"]:
        for c, col in enumerate(cols):
            pieces = col.split("_%s_" % coordinate)
            assert (
                len(pieces) == 2
            )  # otherwise "_[x/y]_" appears in keypoint or model name :(
            kp = pieces[0]
            model = pieces[1]
            new_col = col.replace("_%s_" % coordinate, "_%s_" % coord)
            fig_traces.add_trace(
                go.Scatter(
                    name=model,
                    x=np.arange(df_traces.shape[0]),
                    y=df_traces[new_col],
                    mode="lines",
                    line=dict(color=colors[c]),
                    showlegend=False if coord == "x" else True,
                ),
                row=row,
                col=1,
            )
        yaxis_labels["yaxis%i" % row] = "%s coordinate" % coord
        row += 1

    # -------------------------------------------------------------
    # plot likelihoods
    # -------------------------------------------------------------
    for c, col in enumerate(cols):
        col_l = col.replace("_%s_" % coordinate, "_likelihood_")
        fig_traces.add_trace(
            go.Scatter(
                name=col_l,
                x=np.arange(df_traces.shape[0]),
                y=df_traces[col_l],
                mode="lines",
                line=dict(color=colors[c]),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    yaxis_labels["yaxis%i" % row] = "confidence"
    row += 1

    # -------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------
    for k, v in yaxis_labels.items():
        fig_traces["layout"][k]["title"] = v
    fig_traces.update_layout(
        width=800,
        height=np.sum(row_heights) * 125,
        title_text="Timeseries of %s" % keypoint,
    )

    return fig_traces
