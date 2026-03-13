"""
Visualization utilities for creating Plotly graphs

Provides generic graph builders to reduce repetition and improve maintainability.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def parse_elapsed_time(timestamps: list[str]) -> list[float]:
    """Convert timestamp strings to elapsed seconds from the first timestamp.

    Args:
        timestamps: List of timestamp strings in format "YYYY-MM-DD HH:MM:SS"

    Returns:
        List of elapsed seconds (float) from the first timestamp
    """
    if not timestamps:
        return []

    try:
        dt_objects = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
        start_time = dt_objects[0]
        elapsed = [(dt - start_time).total_seconds() for dt in dt_objects]
        return elapsed
    except (ValueError, AttributeError) as e:
        # Fallback to sample numbers if parsing fails
        logger.warning(f"Failed to parse timestamps: {e}")
        return list(range(len(timestamps)))


def aggregate_all_nodes(node_metrics_list: list[dict]) -> list[dict]:
    """Aggregate all nodes together and average their metrics.

    Similar to group_by_dp but combines ALL nodes (all DP ranks, all TP workers)
    into a single averaged line per run.

    Args:
        node_metrics_list: List of node data dictionaries

    Returns:
        List with one averaged node entry per run
    """
    run_groups = defaultdict(list)

    for node_data in node_metrics_list:
        if not node_data["prefill_batches"]:
            continue

        run_id = node_data.get("run_id", "Unknown")
        run_groups[run_id].append(node_data)

    # Create averaged node data for each run
    aggregated_nodes = []

    for run_id, nodes in run_groups.items():
        if not nodes:
            continue

        # Collect all timestamps and metrics across ALL nodes in this run
        all_batches = defaultdict(list)  # timestamp -> list of metric values

        for node in nodes:
            for batch in node["prefill_batches"]:
                ts = batch.get("timestamp", "")
                if ts:
                    all_batches[ts].append(batch)

        # Average metrics at each timestamp
        averaged_batches = []
        for timestamp in sorted(all_batches.keys()):
            batches_at_time = all_batches[timestamp]

            # Average all numeric metrics
            avg_batch = {"timestamp": timestamp, "dp": 0}

            # List of metrics to average
            metrics = [
                "input_throughput",
                "gen_throughput",
                "new_seq",
                "new_token",
                "running_req",
                "queue_req",
                "inflight_req",
                "transfer_req",
                "prealloc_req",
                "num_tokens",
                "token_usage",
                "preallocated_usage",
            ]

            for metric in metrics:
                values = [b.get(metric) for b in batches_at_time if metric in b]
                if values:
                    avg_batch[metric] = np.mean(values)

            # Copy type from first batch
            if batches_at_time:
                avg_batch["type"] = batches_at_time[0].get("type", "prefill")

            averaged_batches.append(avg_batch)

        # Create aggregated node data structure
        node_count = len(nodes)
        worker_type = nodes[0]["node_info"]["worker_type"]

        aggregated_node = {
            "node_info": {
                "node": "ALL",
                "worker_type": worker_type,
                "worker_id": f"{node_count}nodes",
            },
            "prefill_batches": averaged_batches,
            "memory_snapshots": [],
            "config": nodes[0].get("config", {}),
            "run_id": run_id,
            "run_metadata": nodes[0].get("run_metadata", {}),  # Preserve metadata
        }

        aggregated_nodes.append(aggregated_node)

    return aggregated_nodes


def group_nodes_by_dp(node_metrics_list: list[dict]) -> list[dict]:
    """Group nodes by DP index and average their metrics across TP workers.

    Args:
        node_metrics_list: List of node data dictionaries

    Returns:
        List of grouped node data, one entry per DP group with averaged metrics
    """
    dp_groups = defaultdict(list)

    for node_data in node_metrics_list:
        if not node_data["prefill_batches"]:
            continue

        # Use first batch's DP value as the group key
        first_dp = node_data["prefill_batches"][0].get("dp", 0)
        run_id = node_data.get("run_id", "Unknown")
        group_key = (run_id, first_dp)

        dp_groups[group_key].append(node_data)

    # Create averaged node data for each DP group
    grouped_nodes = []

    for (run_id, dp_idx), nodes in dp_groups.items():
        if not nodes:
            continue

        # Collect all timestamps and metrics across nodes in this DP group
        all_batches = defaultdict(list)  # timestamp -> list of metric values

        for node in nodes:
            for batch in node["prefill_batches"]:
                ts = batch.get("timestamp", "")
                if ts:
                    all_batches[ts].append(batch)

        # Average metrics at each timestamp
        averaged_batches = []
        for timestamp in sorted(all_batches.keys()):
            batches_at_time = all_batches[timestamp]

            # Average all numeric metrics
            avg_batch = {"timestamp": timestamp, "dp": dp_idx}

            # List of metrics to average
            metrics = [
                "input_throughput",
                "gen_throughput",
                "new_seq",
                "new_token",
                "running_req",
                "queue_req",
                "inflight_req",
                "transfer_req",
                "prealloc_req",
                "num_tokens",
                "token_usage",
                "preallocated_usage",
            ]

            for metric in metrics:
                values = [b.get(metric) for b in batches_at_time if metric in b]
                if values:
                    avg_batch[metric] = np.mean(values)

            # Copy type from first batch
            if batches_at_time:
                avg_batch["type"] = batches_at_time[0].get("type", "prefill")

            averaged_batches.append(avg_batch)

        # Create grouped node data structure
        tp_count = len(nodes)
        worker_type = nodes[0]["node_info"]["worker_type"]

        grouped_node = {
            "node_info": {
                "node": f"DP{dp_idx}",
                "worker_type": worker_type,
                "worker_id": f"{tp_count}workers",
            },
            "prefill_batches": averaged_batches,
            "memory_snapshots": [],
            "config": nodes[0]["config"],
            "run_id": run_id,
            "run_metadata": nodes[0].get("run_metadata", {}),  # Preserve metadata
        }

        grouped_nodes.append(grouped_node)

    return grouped_nodes


def create_node_metric_graph(
    node_metrics_list: list[dict],
    title: str,
    y_label: str,
    metric_key: str | None,
    batch_filter: Callable | None = None,
    value_extractor: Callable | None = None,
    mode: str = "lines+markers",
    stackgroup: str | None = None,
    group_by_dp: bool = False,
    aggregate_all: bool = False,
) -> go.Figure:
    """Generic function to create node metric graphs.

    Args:
        node_metrics_list: List of node data
        title: Graph title
        y_label: Y-axis label
        metric_key: Key to extract from batch data (e.g., 'input_throughput')
        batch_filter: Optional function to filter batches (e.g., lambda b: b.get('type') == 'decode')
        value_extractor: Optional function to compute value from batch (default: extract metric_key)
        mode: Plotly trace mode ('lines+markers', 'lines', etc.)
        stackgroup: Stack group name for stacked charts (None for non-stacked)
        group_by_dp: If True, group nodes by DP and show averaged lines per DP group
        aggregate_all: If True, aggregate ALL nodes into a single averaged line per run

    Returns:
        Plotly Figure
    """
    if aggregate_all:
        node_metrics_list = aggregate_all_nodes(node_metrics_list)
    elif group_by_dp:
        node_metrics_list = group_nodes_by_dp(node_metrics_list)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data["prefill_batches"]:
            continue

        # Import get_node_label from log_parser
        from analysis.srtlog.log_parser import get_node_label

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract data
        timestamps = []
        values = []

        for batch in node_data["prefill_batches"]:
            # Apply filter if provided
            if batch_filter and not batch_filter(batch):
                continue

            if "timestamp" not in batch:
                continue

            # Extract value
            if value_extractor:
                value = value_extractor(batch)
            else:
                value = batch.get(metric_key)

            if value is not None:
                timestamps.append(batch["timestamp"])
                values.append(value)

        if timestamps:
            # Convert timestamps to elapsed seconds
            elapsed_seconds = parse_elapsed_time(timestamps)

            trace_config = {
                "x": elapsed_seconds,
                "y": values,
                "mode": mode,
                "name": label,
                "line": {"color": color, "width": 2},
            }

            if "markers" in mode:
                trace_config["marker"] = {"size": 6}

            if stackgroup:
                trace_config["stackgroup"] = stackgroup

            trace_config["hovertemplate"] = (
                f"<b>{label}</b><br>" + "Time: %{x:.1f}s<br>" + f"{y_label}: %{{y:.2f}}<extra></extra>"
            )

            fig.add_trace(go.Scatter(**trace_config))

    fig.update_layout(
        title=title,
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title=y_label,
        hovermode="closest",
        height=400,
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    return fig


def create_stacked_metric_graph(
    node_metrics_list: list[dict],
    title: str,
    metrics_config: list[dict[str, str]],
    batch_filter: Callable | None = None,
    group_by_dp: bool = False,
    aggregate_all: bool = False,
) -> go.Figure:
    """Create stacked area chart for multiple metrics.

    Args:
        node_metrics_list: List of node data
        title: Graph title
        metrics_config: List of dicts with 'key', 'name', 'color' for each metric
        batch_filter: Optional function to filter batches
        group_by_dp: If True, group nodes by DP and show averaged lines per DP group
        aggregate_all: If True, aggregate ALL nodes into a single averaged line per run

    Returns:
        Plotly Figure
    """
    if aggregate_all:
        node_metrics_list = aggregate_all_nodes(node_metrics_list)
    elif group_by_dp:
        node_metrics_list = group_nodes_by_dp(node_metrics_list)

    fig = go.Figure()

    # Aggregate data across all nodes by timestamp
    data_by_time = defaultdict(lambda: {m["key"]: 0 for m in metrics_config})

    for node_data in node_metrics_list:
        if not node_data["prefill_batches"]:
            continue

        for batch in node_data["prefill_batches"]:
            # Apply filter if provided
            if batch_filter and not batch_filter(batch):
                continue

            ts = batch.get("timestamp", "")
            if ts:
                for metric_cfg in metrics_config:
                    if metric_cfg["key"] in batch:
                        data_by_time[ts][metric_cfg["key"]] += batch[metric_cfg["key"]]

    # Sort by timestamp and create arrays
    sorted_times = sorted(data_by_time.keys())
    # Convert timestamps to elapsed seconds
    x_vals = parse_elapsed_time(sorted_times)

    # Add traces in order (bottom to top of stack)
    for metric_cfg in metrics_config:
        values = [data_by_time[ts][metric_cfg["key"]] for ts in sorted_times]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=values,
                mode="lines",
                name=metric_cfg["name"],
                line={"width": 0},
                fillcolor=metric_cfg["color"],
                stackgroup="one",
                hovertemplate=f"{metric_cfg['name']}: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title="Number of Requests",
        hovermode="x unified",
        height=400,
        template="plotly_white",
    )

    return fig


def create_latency_vs_concurrency_graph(
    df: pd.DataFrame, selected_runs: list[str], metric_name: str, metric_col: str, y_label: str
) -> go.Figure:
    """Create latency vs concurrency graph for a specific metric.

    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs to plot
        metric_name: Display name for the metric (e.g., "TTFT")
        metric_col: Column name in dataframe (e.g., "Mean TTFT (ms)")
        y_label: Y-axis label
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for idx, run_id in enumerate(selected_runs):
        run_data = df[df["Run ID"] == run_id].sort_values("Concurrency")

        if len(run_data) == 0:
            continue

        color = colors[idx % len(colors)]

        # Filter out N/A values
        valid_data = run_data[run_data[metric_col] != "N/A"].copy()

        if len(valid_data) == 0:
            continue

        # Get run date if available
        run_date = valid_data.iloc[0].get("Run Date", "N/A") if len(valid_data) > 0 else "N/A"

        # Create hover text
        hover_text = [
            f"<b>{run_id}</b><br>"
            + f"Date: {run_date}<br>"
            + f"Concurrency: {row['Concurrency']}<br>"
            + f"{metric_name}: {row[metric_col]:.2f} ms"
            for _, row in valid_data.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=valid_data["Concurrency"],
                y=valid_data[metric_col],
                mode="lines+markers",
                name=run_id,
                line={"color": color, "width": 2},
                marker={"size": 8, "color": color},
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title={
            "text": f"{metric_name} vs Concurrency",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="Concurrency",
        yaxis_title=y_label,
        hovermode="closest",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        height=450,
        template="plotly_white",
    )

    return fig


def calculate_pareto_frontier(df: pd.DataFrame, y_metric: str = "Output TPS/GPU") -> list[tuple]:
    """Calculate the Pareto frontier points.

    A point is on the Pareto frontier if no other point is strictly better
    in both dimensions (higher TPS/User AND higher y_metric).

    Args:
        df: DataFrame with 'Output TPS/User' and y_metric columns
        y_metric: Y-axis metric to use ("Output TPS/GPU" or "Total TPS/GPU")
                 Both metrics are normalized per GPU

    Returns:
        List of (x, y) tuples representing frontier points, sorted by x
    """
    points = df[["Output TPS/User", y_metric]].values.tolist()

    if len(points) == 0:
        return []

    # Find all non-dominated points
    frontier = []

    for i, (x1, y1) in enumerate(points):
        is_dominated = False

        # Check if this point is dominated by any other point
        for j, (x2, y2) in enumerate(points):
            if i != j:
                # Point (x2, y2) dominates (x1, y1) if it's better in both dimensions
                if x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                    is_dominated = True
                    break

        if not is_dominated:
            frontier.append((x1, y1))

    # Sort frontier points by x coordinate for proper line drawing
    frontier_sorted = sorted(frontier, key=lambda p: p[0])

    return frontier_sorted


def create_pareto_graph(
    df: pd.DataFrame,
    selected_runs: list[str],
    show_cutoff: bool = False,
    cutoff_value: float = 30.0,
    show_frontier: bool = False,
    y_metric: str = "Output TPS/GPU",
    run_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Create interactive Pareto graph with optional cutoff line and frontier.

    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs to plot
        show_cutoff: Whether to show vertical cutoff line
        cutoff_value: X-axis value for cutoff line (TPS/User)
        show_frontier: Whether to show the Pareto frontier
        y_metric: Y-axis metric to plot ("Output TPS/GPU" or "Total TPS/GPU")
                 Both metrics are normalized per GPU
        run_labels: Optional dict mapping run_id to display label for legend
    """
    fig = go.Figure()

    # Add Pareto frontier FIRST if enabled (so it appears behind data points)
    if show_frontier and len(df) > 0:
        frontier_points = calculate_pareto_frontier(df, y_metric)

        if len(frontier_points) > 1:
            frontier_x = [p[0] for p in frontier_points]
            frontier_y = [p[1] for p in frontier_points]

            # Outer glow - widest, most transparent
            fig.add_trace(
                go.Scatter(
                    x=frontier_x,
                    y=frontier_y,
                    mode="lines",
                    line={"color": "rgba(255, 223, 0, 0.15)", "width": 20},
                    showlegend=False,
                    hoverinfo="skip",
                    name="Frontier Glow Outer",
                )
            )

            # Middle glow
            fig.add_trace(
                go.Scatter(
                    x=frontier_x,
                    y=frontier_y,
                    mode="lines",
                    line={"color": "rgba(255, 223, 0, 0.25)", "width": 12},
                    showlegend=False,
                    hoverinfo="skip",
                    name="Frontier Glow Middle",
                )
            )

            # Thin highlighted line
            fig.add_trace(
                go.Scatter(
                    x=frontier_x,
                    y=frontier_y,
                    mode="lines",
                    line={"color": "rgba(255, 215, 0, 0.6)", "width": 3, "dash": "dot"},
                    showlegend=False,
                    hoverinfo="skip",
                    name="Frontier Line",
                )
            )

            # Small markers on frontier points
            fig.add_trace(
                go.Scatter(
                    x=frontier_x,
                    y=frontier_y,
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": "rgba(255, 215, 0, 0.4)",
                        "line": {"width": 1, "color": "rgba(255, 215, 0, 0.8)"},
                    },
                    showlegend=False,
                    hoverinfo="skip",
                    name="Frontier Points",
                )
            )

            # Add frontier label
            mid_idx = len(frontier_points) // 2
            fig.add_annotation(
                x=frontier_x[mid_idx],
                y=frontier_y[mid_idx],
                text="⭐ Pareto Frontier",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(255, 215, 0, 0.8)",
                ax=50,
                ay=-50,
                font={"size": 12, "color": "gold"},
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="rgba(255, 215, 0, 0.8)",
                borderwidth=2,
                borderpad=4,
            )
        elif len(frontier_points) == 1:
            # Single point frontier - just highlight it with a subtle ring
            fig.add_trace(
                go.Scatter(
                    x=[frontier_points[0][0]],
                    y=[frontier_points[0][1]],
                    mode="markers",
                    marker={
                        "size": 25,
                        "color": "rgba(255, 215, 0, 0.3)",
                        "line": {"width": 2, "color": "rgba(255, 215, 0, 0.6)"},
                    },
                    showlegend=False,
                    hoverinfo="skip",
                    name="Frontier Point",
                )
            )

    # Now add the actual data points (so they appear on top of frontier)
    colors = px.colors.qualitative.Set1
    for idx, run_id in enumerate(selected_runs):
        run_data = df[df["Run ID"] == run_id].copy()

        # Filter out rows where y_metric is "N/A" (can't plot these)
        run_data = run_data[run_data[y_metric] != "N/A"]

        if run_data.empty:
            continue  # Skip this run if no valid data

        # Choose which throughput to show based on y_metric
        if y_metric == "Total TPS/GPU":
            tps_label = "Total TPS"
            tps_column = "Total TPS"
        else:
            tps_label = "Output TPS"
            tps_column = "Output TPS"

        # Use custom label if provided, otherwise extract job number from run_id
        if run_labels and run_id in run_labels:
            legend_name = run_labels[run_id]
        else:
            job_num = run_id.split("_")[0] if "_" in run_id else run_id
            legend_name = f"Job {job_num}"

        # Helper function to format values that might be NaN or "N/A"
        def format_value(val):
            if val == "N/A" or val is None or (isinstance(val, float) and pd.isna(val)):
                return "N/A"
            return f"{val:.2f}"

        fig.add_trace(
            go.Scatter(
                x=run_data["Output TPS/User"],
                y=run_data[y_metric],
                mode="markers+lines",
                name=legend_name,
                marker={"size": 10, "color": colors[idx % len(colors)]},
                line={"color": colors[idx % len(colors)], "width": 2},
                text=[
                    f"Run: {row['Run ID']}<br>"
                    f"Concurrency: {row['Concurrency']}<br>"
                    f"Output TPS/User: {format_value(row['Output TPS/User'])}<br>"
                    f"{y_metric}: {format_value(row[y_metric])}<br>"
                    f"{tps_label}: {format_value(row[tps_column])}<br>"
                    f"Mean TTFT: {format_value(row['Mean TTFT (ms)'])} ms<br>"
                    f"Mean TPOT: {format_value(row['Mean TPOT (ms)'])} ms"
                    for _, row in run_data.iterrows()
                ],
                hoverinfo="text",
                customdata=[
                    [legend_name, row["Concurrency"], row["Output TPS/User"], row[y_metric]]
                    for _, row in run_data.iterrows()
                ],
                selected={"marker": {"size": 14}},
                unselected={"marker": {"opacity": 0.4}},
            )
        )

    # Add vertical cutoff line if enabled
    if show_cutoff:
        fig.add_vline(
            x=cutoff_value,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Target: {cutoff_value} TPS/User",
            annotation_position="top right",
            annotation_font_size=12,
            annotation_font_color="red",
        )

    # Determine title and axis labels based on y_metric
    if y_metric == "Total TPS/GPU":
        title_text = "Pareto Frontier: Total TPS/GPU vs Output TPS/User"
        y_axis_title = "Total TPS/GPU"
    else:
        title_text = "Pareto Frontier: Output TPS/GPU vs Output TPS/User"
        y_axis_title = "Output TPS/GPU"

    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        xaxis_title="Output TPS/User",
        yaxis_title=y_axis_title,
        hovermode="closest",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        height=600,
        template="plotly_white",
    )

    return fig
