"""A collection of visualizations for various pose estimation performance metrics."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve

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


def plot_calibration_diagram(
    confidences,
    accuracies,
    n_bins=10,
    model_name="Model",
    keypoint_name="",
    data_type="",
    error_threshold=5.0,
):
    """
    Plot calibration diagram for pose estimation model using Plotly.

    Args:
        confidences: predicted confidence scores (0-1)
        accuracies: binary array indicating if prediction was accurate (1) or not (0)
        n_bins: number of bins for grouping confidences
        model_name: name of the model for title
        keypoint_name: name of the keypoint being analyzed
        data_type: train/val/test data split
        error_threshold: pixel error threshold used to determine accuracy

    Returns:
        Plotly figure object
    """
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        accuracies, confidences, n_bins=n_bins, strategy='uniform'
    )

    # Calculate expected calibration error (ECE) - simplified version
    # ECE is the weighted average of the absolute differences between accuracy and confidence
    if len(mean_predicted_value) > 0 and len(confidences) > 0:
        # For each bin, calculate |accuracy - confidence| weighted by bin size
        # We'll recompute bins to ensure consistency
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_count = 0

        for i in range(n_bins):
            # Find points in this bin
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

            bin_count = np.sum(in_bin)
            if bin_count > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                ece += bin_count * np.abs(bin_accuracy - bin_confidence)
                total_count += bin_count

        ece = ece / total_count if total_count > 0 else 0
    else:
        ece = 0

    # Create Plotly figure
    fig = go.Figure()

    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect calibration',
        line=dict(dash='dash', color='black'),
        showlegend=True
    ))

    # Add model calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='markers+lines',
        name=f'{model_name}',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue'),
        showlegend=True
    ))

    # Add confidence histogram as marginal
    fig.add_trace(go.Histogram(
        x=confidences,
        name='Confidence distribution',
        yaxis='y2',
        opacity=0.3,
        showlegend=False,
        marker_color='gray',
        nbinsx=20
    ))

    # Update layout
    title_text = f'Calibration Plot - {model_name}'
    if keypoint_name:
        title_text += f' ({keypoint_name})'
    if data_type:
        title_text += f' - {data_type} set'
    title_text += f'<br>Error threshold: {error_threshold:.1f} pixels | ECE: {ece:.3f}'

    fig.update_layout(
        title=title_text,
        xaxis=dict(
            title='Mean Predicted Confidence',
            range=[0, 1],
            tickmode='linear',
            tick0=0,
            dtick=0.1
        ),
        yaxis=dict(
            title='Fraction of Accurate Predictions',
            range=[0, 1],
            tickmode='linear',
            tick0=0,
            dtick=0.1
        ),
        yaxis2=dict(
            title='Count',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        width=700,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def plot_calibration_diagram_multi(
    models_data,
    n_bins=10,
    keypoint_name="",
    data_type="",
    error_threshold=5.0,
):
    """
    Plot calibration diagram for multiple pose estimation models using Plotly.

    Args:
        models_data: list of dicts with keys 'model_name', 'confidences', 'accuracies'
        n_bins: number of bins for grouping confidences
        keypoint_name: name of the keypoint being analyzed
        data_type: train/val/test data split
        error_threshold: pixel error threshold used to determine accuracy

    Returns:
        Plotly figure object
    """
    # Define colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Create Plotly figure
    fig = go.Figure()

    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect calibration',
        line=dict(dash='dash', color='black', width=2),
        showlegend=True
    ))

    # Add calibration curves for each model
    ece_values = []
    for i, model_data in enumerate(models_data):
        model_name = model_data['model_name']
        confidences = model_data['confidences']
        accuracies = model_data['accuracies']
        color = colors[i % len(colors)]

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=n_bins, strategy='uniform'
        )

        # Calculate ECE for this model
        if len(mean_predicted_value) > 0 and len(confidences) > 0:
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            total_count = 0

            for j in range(n_bins):
                in_bin = (confidences >= bin_edges[j]) & (confidences < bin_edges[j + 1])
                if j == n_bins - 1:  # Include right edge in last bin
                    in_bin = (confidences >= bin_edges[j]) & (confidences <= bin_edges[j + 1])

                bin_count = np.sum(in_bin)
                if bin_count > 0:
                    bin_accuracy = np.mean(accuracies[in_bin])
                    bin_confidence = np.mean(confidences[in_bin])
                    ece += bin_count * np.abs(bin_accuracy - bin_confidence)
                    total_count += bin_count

            ece = ece / total_count if total_count > 0 else 0
        else:
            ece = 0

        ece_values.append(ece)

        # Add model calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='markers+lines',
            name=f'{model_name} (ECE: {ece:.3f})',
            marker=dict(size=8, color=color),
            line=dict(color=color, width=2),
            showlegend=True
        ))

    # Create title
    title_text = 'Model Calibration Comparison'
    if keypoint_name:
        title_text += f' - {keypoint_name}'
    if data_type:
        title_text += f' ({data_type} set)'
    title_text += f'<br>Error threshold: {error_threshold:.1f} pixels'

    # Update layout
    fig.update_layout(
        title=title_text,
        xaxis=dict(
            title='Mean Predicted Confidence',
            range=[0, 1],
            tickmode='linear',
            tick0=0,
            dtick=0.1
        ),
        yaxis=dict(
            title='Fraction of Accurate Predictions',
            range=[0, 1],
            tickmode='linear',
            tick0=0,
            dtick=0.1
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig
