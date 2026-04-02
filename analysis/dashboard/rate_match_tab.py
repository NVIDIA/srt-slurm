# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Rate Match Analysis Tab - Compare prefill input rate vs decode generation rate
"""

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from analysis.dashboard.components import load_node_metrics


def render(filtered_runs: list, logs_dir: str):
    """Render rate match analysis.

    Args:
        filtered_runs: List of BenchmarkRun objects
        logs_dir: Path to logs directory
    """
    st.subheader("Rate Match Analysis")
    st.markdown("""
    Compare prefill input rate vs decode generation rate to verify proper node ratio.
    """)

    # Toggle for request rate approximation
    col1, col2 = st.columns([1, 3])
    with col1:
        show_request_rate = st.toggle(
            "Show Request Rate", value=False, help="Convert tokens/s to requests/s using ISL/OSL"
        )

    st.caption("""
    **What to look for:**
    - Lines should align when system is balanced
    - Decode rate consistently below prefill = decode bottleneck (need more decode nodes)
    - Decode rate above prefill = prefill bottleneck (decode nodes underutilized)
    """)

    # Filter out aggregated runs (rate matching doesn't apply to aggregated mode)
    disagg_runs = [run for run in filtered_runs if not run.metadata.is_aggregated]
    agg_runs = [run for run in filtered_runs if run.metadata.is_aggregated]

    if agg_runs:
        agg_count = len(agg_runs)
        st.info(
            f"ℹ️ Rate match analysis is not applicable to aggregated mode. "
            f"Skipping {agg_count} aggregated run(s): " + ", ".join([f"Job {r.job_id}" for r in agg_runs])
        )

    if not disagg_runs:
        st.warning("No disaggregated runs selected. Rate match analysis requires disaggregated (prefill/decode) runs.")
        return

    st.divider()

    # Render each run in its own section
    for idx, run in enumerate(disagg_runs):
        run_path = run.metadata.path

        # Load node metrics
        with st.spinner(f"Loading metrics for Job {run.job_id}..."):
            node_metrics = load_node_metrics(run_path)

        if not node_metrics:
            st.error(f"No node metrics found for Job {run.job_id}")
            continue

        # Split by type
        prefill_nodes = [n for n in node_metrics if n["node_info"]["worker_type"] == "prefill"]
        decode_nodes = [n for n in node_metrics if n["node_info"]["worker_type"] == "decode"]

        # Show run details
        with st.expander(
            f"🔧 Job {run.job_id} - {run.metadata.prefill_workers}P{run.metadata.decode_workers}D ({run.metadata.formatted_date})",
            expanded=len(filtered_runs) == 1,
        ):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prefill Nodes", len(prefill_nodes))
            with col2:
                st.metric("Decode Nodes", len(decode_nodes))
            with col3:
                prefill_gpus = run.metadata.prefill_nodes * run.metadata.gpus_per_node
                decode_gpus = run.metadata.decode_nodes * run.metadata.gpus_per_node
                st.metric("GPU Split", f"{prefill_gpus} / {decode_gpus}")
            with col4:
                st.metric("ISL/OSL", f"{run.profiler.isl}/{run.profiler.osl}")

            # Create rate match graph
            isl = int(run.profiler.isl) if run.profiler.isl else None
            osl = int(run.profiler.osl) if run.profiler.osl else None
            rate_fig = _create_rate_match_graph(
                prefill_nodes, decode_nodes, run.job_id, show_request_rate=show_request_rate, isl=isl, osl=osl
            )
            st.plotly_chart(rate_fig, width="stretch", key=f"rate_match_{run.job_id}")

        # Add divider between runs except for the last one
        if idx < len(disagg_runs) - 1:
            st.divider()


def _create_rate_match_graph(prefill_nodes, decode_nodes, job_id="", show_request_rate=False, isl=None, osl=None):
    """Create rate matching graph comparing prefill input vs decode generation.

    Args:
        prefill_nodes: List of prefill node metrics
        decode_nodes: List of decode node metrics
        job_id: Job identifier for title
        show_request_rate: If True, convert tokens/s to requests/s using ISL/OSL
        isl: Input sequence length for conversion
        osl: Output sequence length for conversion
    """
    rate_fig = go.Figure()

    title_suffix = f" - Job {job_id}" if job_id else ""

    # Determine divisors for request rate conversion
    prefill_divisor = isl if (show_request_rate and isl) else 1
    decode_divisor = osl if (show_request_rate and osl) else 1

    # Get prefill input throughput over time
    if prefill_nodes:
        all_prefill_batches = {}
        for p_node in prefill_nodes:
            for batch in p_node["prefill_batches"]:
                if batch.get("input_throughput") is not None:
                    ts = batch.get("timestamp", "")
                    if ts:
                        if ts not in all_prefill_batches:
                            all_prefill_batches[ts] = []
                        all_prefill_batches[ts].append(batch["input_throughput"])

        timestamps = []
        avg_input_tps = []

        for ts in sorted(all_prefill_batches.keys()):
            avg = sum(all_prefill_batches[ts]) / len(all_prefill_batches[ts])
            timestamps.append(ts)
            avg_input_tps.append(avg / prefill_divisor)

        if timestamps:
            first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            elapsed = [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time).total_seconds() for ts in timestamps]

            unit = "req/s" if show_request_rate else "tok/s"
            rate_fig.add_trace(
                go.Scatter(
                    x=elapsed,
                    y=avg_input_tps,
                    mode="lines+markers",
                    name=f"Prefill Input Rate [{unit}] (avg {len(prefill_nodes)} nodes)",
                    line={"color": "orange", "width": 3},
                    marker={"size": 6},
                )
            )

    # Get decode gen throughput over time
    if decode_nodes:
        all_decode_batches = {}
        for d_node in decode_nodes:
            for batch in d_node["prefill_batches"]:
                if batch.get("gen_throughput") is not None and batch.get("gen_throughput") > 0:
                    ts = batch.get("timestamp", "")
                    if ts:
                        if ts not in all_decode_batches:
                            all_decode_batches[ts] = []
                        all_decode_batches[ts].append(batch["gen_throughput"])

        timestamps = []
        avg_gen_tps = []

        for ts in sorted(all_decode_batches.keys()):
            avg = sum(all_decode_batches[ts]) / len(all_decode_batches[ts])
            timestamps.append(ts)
            avg_gen_tps.append(avg / decode_divisor)

        if timestamps:
            first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            elapsed = [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time).total_seconds() for ts in timestamps]

            unit = "req/s" if show_request_rate else "tok/s"
            rate_fig.add_trace(
                go.Scatter(
                    x=elapsed,
                    y=avg_gen_tps,
                    mode="lines+markers",
                    name=f"Decode Gen Rate [{unit}] (avg {len(decode_nodes)} nodes)",
                    line={"color": "green", "width": 3},
                    marker={"size": 6},
                )
            )

    y_unit = "requests/s" if show_request_rate else "tokens/s"
    rate_fig.update_layout(
        title=f"Rate Match: Prefill Input vs Decode Generation{title_suffix}",
        xaxis_title="Time Elapsed (seconds)",
        yaxis_title=f"Average Throughput ({y_unit} per node)",
        hovermode="x unified",
        height=500,
        showlegend=True,
    )
    rate_fig.update_xaxes(showgrid=True)
    rate_fig.update_yaxes(showgrid=True)

    return rate_fig
