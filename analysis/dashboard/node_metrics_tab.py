# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Node-Level Metrics Tab
"""

import os

import streamlit as st

from analysis.dashboard.components import (
    create_decode_disagg_stacked_graph,
    create_decode_gen_throughput_graph,
    create_decode_prealloc_req_graph,
    create_decode_running_requests_graph,
    create_decode_transfer_req_graph,
    create_kv_cache_utilization_graph,
    create_node_inflight_requests_graph,
    create_node_throughput_graph,
    create_queue_depth_graph,
    load_node_metrics,
)
from analysis.srtlog.visualizations import aggregate_all_nodes, group_nodes_by_dp


def render(filtered_runs: list, logs_dir: str):
    """Render the node-level metrics tab.

    Args:
        filtered_runs: List of BenchmarkRun objects
        logs_dir: Path to logs directory
    """
    st.subheader("Node-Level Metrics")

    # Check if any runs are aggregated
    has_aggregated = any(run.metadata.is_aggregated for run in filtered_runs)
    has_disaggregated = any(not run.metadata.is_aggregated for run in filtered_runs)

    if has_aggregated and has_disaggregated:
        st.markdown("""
        Runtime metrics extracted from log files. Mixed mode (aggregated + disaggregated runs selected).
        """)
    elif has_aggregated:
        st.markdown("""
        Runtime metrics extracted from log files (aggregated mode - no prefill/decode split).
        """)
    else:
        st.markdown("""
        Runtime metrics extracted from log files, split by prefill and decode nodes.
        """)

    # Parse log files for all selected runs (cached)
    all_node_metrics = []
    with st.spinner(f"Parsing logs for {len(filtered_runs)} run(s)..."):
        for run in filtered_runs:
            run_path = run.metadata.path
            if run.metadata.is_aggregated:
                run_id = f"{run.job_id}_{run.metadata.agg_workers}A_{run.metadata.run_date}"
            else:
                run_id = f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
            if run_path and os.path.exists(run_path):
                node_metrics = load_node_metrics(run_path)
                for node_data in node_metrics:
                    node_data["run_id"] = run_id
                    # Add run metadata for better labels
                    node_data["run_metadata"] = {
                        "job_id": run.job_id,
                        "is_aggregated": run.metadata.is_aggregated,
                        "prefill_nodes": run.metadata.prefill_nodes,
                        "decode_nodes": run.metadata.decode_nodes,
                        "prefill_workers": run.metadata.prefill_workers,
                        "decode_workers": run.metadata.decode_workers,
                        "agg_nodes": run.metadata.agg_nodes,
                        "agg_workers": run.metadata.agg_workers,
                        "gpus_per_node": run.metadata.gpus_per_node,
                        "total_gpus": run.total_gpus,
                        "isl": run.profiler.isl,
                        "osl": run.profiler.osl,
                        "gpu_type": run.metadata.gpu_type,
                    }
                all_node_metrics.extend(node_metrics)

    if not all_node_metrics:
        st.warning("No log files found for the selected runs.")
        st.info(
            "Node metrics are extracted from files like `*_prefill_*.err`, `*_decode_*.err`, `*_prefill_*.out`, or `*_decode_*.out`"
        )
        return

    # Split by prefill vs decode (for disaggregated) or agg (for aggregated)
    prefill_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] == "prefill"]
    decode_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] == "decode"]
    agg_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] in ["agg", "aggregated"]]

    st.caption(
        f"📊 Found {len(prefill_nodes)} prefill nodes, {len(decode_nodes)} decode nodes, {len(agg_nodes)} aggregated nodes"
    )

    # Add toggle for aggregation
    aggregation_mode = st.radio(
        "📊 Node Aggregation",
        options=[
            "Show individual nodes",
            "Group by DP rank (average per DP)",
            "Aggregate all nodes (single averaged line)",
        ],
        index=2,
        horizontal=True,
        help="Control how node metrics are displayed: individual lines, grouped by DP rank, or fully aggregated across all nodes.",
    )

    # Pre-aggregate nodes ONCE to avoid recomputing for each graph
    if aggregation_mode == "Aggregate all nodes (single averaged line)":
        with st.spinner("Aggregating nodes..."):
            if prefill_nodes:
                prefill_nodes = aggregate_all_nodes(prefill_nodes)
            if decode_nodes:
                decode_nodes = aggregate_all_nodes(decode_nodes)
            if agg_nodes:
                agg_nodes = aggregate_all_nodes(agg_nodes)
        group_by_dp = False
        aggregate_all = False  # Already aggregated
    elif aggregation_mode == "Group by DP rank (average per DP)":
        with st.spinner("Grouping by DP..."):
            if prefill_nodes:
                prefill_nodes = group_nodes_by_dp(prefill_nodes)
            if decode_nodes:
                decode_nodes = group_nodes_by_dp(decode_nodes)
            if agg_nodes:
                agg_nodes = group_nodes_by_dp(agg_nodes)
        group_by_dp = False
        aggregate_all = False  # Already grouped
    else:
        # Individual nodes - no preprocessing
        group_by_dp = False
        aggregate_all = False

    # Aggregated Metrics Section (for aggregated mode runs)
    if agg_nodes:
        st.markdown("### ⚙️ Aggregated Node Metrics")
        _render_agg_metrics(agg_nodes, group_by_dp, aggregate_all)
        if prefill_nodes or decode_nodes:
            st.divider()

    # Prefill Metrics Section
    if prefill_nodes:
        st.markdown("### 📤 Prefill Node Metrics")
        _render_prefill_metrics(prefill_nodes, group_by_dp, aggregate_all)

    # Decode Metrics Section
    if decode_nodes:
        if prefill_nodes:
            st.divider()
        st.markdown("### 📥 Decode Node Metrics")
        _render_decode_metrics(decode_nodes, group_by_dp, aggregate_all)


def _render_agg_metrics(agg_nodes, group_by_dp, aggregate_all):
    """Render aggregated mode node metrics."""
    # Check if nodes have batch data
    has_data = any(node_data["prefill_batches"] for node_data in agg_nodes)
    if not has_data:
        st.warning("⚠️ No batch metrics found for aggregated nodes.")
        return

    throughput_fig = create_node_throughput_graph(agg_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    throughput_fig.update_xaxes(showgrid=True)
    throughput_fig.update_yaxes(showgrid=True)
    st.plotly_chart(throughput_fig, width="stretch", key="agg_throughput")
    st.caption("Shows input throughput in tokens/s - measures how fast the system processes prompts")

    kv_fig = create_kv_cache_utilization_graph(agg_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    kv_fig.update_xaxes(showgrid=True)
    kv_fig.update_yaxes(showgrid=True)
    st.plotly_chart(kv_fig, width="stretch", key="agg_kv_util")
    st.caption(
        "Percentage of KV cache memory currently in use - helps tune max-total-tokens and identify memory pressure"
    )

    queue_fig = create_queue_depth_graph(agg_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    queue_fig.update_layout(title="Queued Requests")
    queue_fig.update_xaxes(showgrid=True)
    queue_fig.update_yaxes(showgrid=True)
    st.plotly_chart(queue_fig, width="stretch", key="agg_queue")
    st.caption("Requests waiting in queue - growing queue indicates backpressure or overload")


def _render_prefill_metrics(prefill_nodes, group_by_dp, aggregate_all):
    """Render prefill node metrics."""
    throughput_fig = create_node_throughput_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    throughput_fig.update_xaxes(showgrid=True)
    throughput_fig.update_yaxes(showgrid=True)
    st.plotly_chart(throughput_fig, width="stretch", key="prefill_throughput")
    st.caption("Shows prefill throughput in tokens/s - measures how fast the system processes input prompts")

    inflight_fig = create_node_inflight_requests_graph(
        prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
    )
    inflight_fig.update_xaxes(showgrid=True)
    inflight_fig.update_yaxes(showgrid=True)
    st.plotly_chart(inflight_fig, width="stretch", key="prefill_inflight")
    st.caption("Requests that have been sent to decode workers in PD disaggregation mode")

    kv_fig = create_kv_cache_utilization_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    kv_fig.update_xaxes(showgrid=True)
    kv_fig.update_yaxes(showgrid=True)
    st.plotly_chart(kv_fig, width="stretch", key="prefill_kv_util")
    st.caption(
        "Percentage of KV cache memory currently in use - helps tune max-total-tokens and identify memory pressure"
    )

    queue_fig = create_queue_depth_graph(prefill_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    queue_fig.update_layout(title="PREFILL Queued Requests")
    queue_fig.update_xaxes(showgrid=True)
    queue_fig.update_yaxes(showgrid=True)
    st.plotly_chart(queue_fig, width="stretch", key="prefill_queue_v2")
    st.caption("Prefill requests waiting in queue - growing queue indicates backpressure or overload")


def _render_decode_metrics(decode_nodes, group_by_dp, aggregate_all):
    """Render decode node metrics."""
    # Check if decode nodes have batch data
    has_data = any(node_data["prefill_batches"] for node_data in decode_nodes)
    if not has_data:
        st.warning(
            "⚠️ No batch metrics found for decode nodes. Decode nodes may not log batch-level metrics in the current setup."
        )
        return

    running_fig = create_decode_running_requests_graph(
        decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
    )
    running_fig.update_xaxes(showgrid=True)
    running_fig.update_yaxes(showgrid=True)
    st.plotly_chart(running_fig, width="stretch", key="decode_running")
    st.caption("Number of requests currently being decoded and generating output tokens")

    gen_fig = create_decode_gen_throughput_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    gen_fig.update_xaxes(showgrid=True)
    gen_fig.update_yaxes(showgrid=True)
    st.plotly_chart(gen_fig, width="stretch", key="decode_gen_throughput")
    st.caption("Output token generation rate in tokens/s - measures decode performance")

    kv_decode_fig = create_kv_cache_utilization_graph(
        decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
    )
    kv_decode_fig.update_xaxes(showgrid=True)
    kv_decode_fig.update_yaxes(showgrid=True)
    st.plotly_chart(kv_decode_fig, width="stretch", key="decode_kv_util")
    st.caption(
        "Total KV cache tokens in use across all running requests - low indicates underutilization, high indicates risk of OOM"
    )

    queue_decode_fig = create_queue_depth_graph(decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all)
    queue_decode_fig.update_layout(title="DECODE Queued Requests")
    queue_decode_fig.update_xaxes(showgrid=True)
    queue_decode_fig.update_yaxes(showgrid=True)
    st.plotly_chart(queue_decode_fig, width="stretch", key="decode_queue_v2")
    st.caption("Decode requests waiting in queue - indicates decode capacity constraints")

    # Disaggregation metrics with toggle
    st.divider()
    st.markdown("#### Disaggregation Metrics")

    disagg_view = st.radio(
        "View mode",
        options=["Stacked (Combined)", "Separate Graphs"],
        index=0,
        horizontal=True,
        help="Stacked view shows request flow through stages. Separate graphs show individual metrics.",
    )

    if disagg_view == "Stacked (Combined)":
        stacked_fig = create_decode_disagg_stacked_graph(
            decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
        )
        stacked_fig.update_xaxes(showgrid=True)
        stacked_fig.update_yaxes(showgrid=True)
        st.plotly_chart(stacked_fig, width="stretch", key="decode_disagg_stacked")
        st.caption(
            "Shows the request flow funnel: Prealloc Queue → Transfer Queue → Running requests in PD disaggregation"
        )
    else:
        transfer_fig = create_decode_transfer_req_graph(
            decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
        )
        transfer_fig.update_xaxes(showgrid=True)
        transfer_fig.update_yaxes(showgrid=True)
        st.plotly_chart(transfer_fig, width="stretch", key="decode_transfer")
        st.caption("Requests waiting for KV cache transfer from prefill to decode workers in PD disaggregation mode")

        prealloc_fig = create_decode_prealloc_req_graph(
            decode_nodes, group_by_dp=group_by_dp, aggregate_all=aggregate_all
        )
        prealloc_fig.update_xaxes(showgrid=True)
        prealloc_fig.update_yaxes(showgrid=True)
        st.plotly_chart(prealloc_fig, width="stretch", key="decode_prealloc")
        st.caption(
            "Requests in pre-allocation queue for PD disaggregation - waiting for memory allocation on decode workers"
        )
