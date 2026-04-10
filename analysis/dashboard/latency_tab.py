# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Latency Analysis Tab
"""

import pandas as pd
import streamlit as st

from analysis.srtlog.visualizations import create_latency_vs_concurrency_graph


def render(df: pd.DataFrame, selected_runs: list[str]):
    """Render the latency analysis tab.

    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs
    """
    st.subheader("Latency Analysis")

    if len(selected_runs) == 0:
        st.warning("Please select at least one run from the sidebar.")
        return

    # TTFT Graph
    st.markdown("### Time to First Token (TTFT)")
    st.markdown("""
    **TTFT** measures the time from request submission to receiving the first output token.
    This is critical for perceived responsiveness.
    """)
    ttft_fig = create_latency_vs_concurrency_graph(
        df,
        selected_runs,
        metric_name="TTFT",
        metric_col="Mean TTFT (ms)",
        y_label="Mean TTFT (ms)",
    )
    st.plotly_chart(ttft_fig, width="stretch", key="latency_ttft")

    # TPOT Graph
    st.markdown("### Time Per Output Token (TPOT)")
    st.markdown("""
    **TPOT** measures the time between consecutive output tokens during generation.
    Lower TPOT means faster streaming and better user experience.
    """)
    tpot_fig = create_latency_vs_concurrency_graph(
        df,
        selected_runs,
        metric_name="TPOT",
        metric_col="Mean TPOT (ms)",
        y_label="Mean TPOT (ms)",
    )
    st.plotly_chart(tpot_fig, width="stretch", key="latency_tpot")

    # ITL Graph
    st.markdown("### Inter-Token Latency (ITL)")
    st.markdown("""
    **ITL** measures the interval between tokens during generation.
    Similar to TPOT but may include queueing delays.
    """)
    itl_fig = create_latency_vs_concurrency_graph(
        df,
        selected_runs,
        metric_name="ITL",
        metric_col="Mean ITL (ms)",
        y_label="Mean ITL (ms)",
    )
    st.plotly_chart(itl_fig, width="stretch", key="latency_itl")

    # Summary statistics
    st.divider()
    st.markdown("### 📊 Latency Summary Statistics")

    summary_data = []
    for run_id in selected_runs:
        run_data = df[df["Run ID"] == run_id]
        if len(run_data) > 0:
            summary_data.append(
                {
                    "Run ID": run_id,
                    "Min TTFT (ms)": run_data["Mean TTFT (ms)"].min()
                    if "Mean TTFT (ms)" in run_data.columns
                    else "N/A",
                    "Max TTFT (ms)": run_data["Mean TTFT (ms)"].max()
                    if "Mean TTFT (ms)" in run_data.columns
                    else "N/A",
                    "Min TPOT (ms)": run_data["Mean TPOT (ms)"].min()
                    if "Mean TPOT (ms)" in run_data.columns
                    else "N/A",
                    "Max TPOT (ms)": run_data["Mean TPOT (ms)"].max()
                    if "Mean TPOT (ms)" in run_data.columns
                    else "N/A",
                    "Min ITL (ms)": run_data["Mean ITL (ms)"].min() if "Mean ITL (ms)" in run_data.columns else "N/A",
                    "Max ITL (ms)": run_data["Mean ITL (ms)"].max() if "Mean ITL (ms)" in run_data.columns else "N/A",
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width="stretch")
