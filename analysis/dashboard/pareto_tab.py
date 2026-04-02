# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pareto Frontier Analysis Tab
"""

import pandas as pd
import streamlit as st

from analysis.srtlog.visualizations import calculate_pareto_frontier, create_pareto_graph


def render(df: pd.DataFrame, selected_runs: list[str], run_legend_labels: dict, pareto_options: dict):
    """Render the Pareto frontier analysis tab.

    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs
        run_legend_labels: Dict mapping run_id to display label
        pareto_options: Dict with show_cutoff, cutoff_value, show_frontier
    """
    st.subheader("Pareto Frontier Analysis")

    # Y-axis metric toggle
    y_axis_metric = st.radio(
        "Y-axis metric",
        options=["Output TPS/GPU", "Total TPS/GPU"],
        index=0,
        horizontal=True,
        help="Choose between decode throughput per GPU or total throughput per GPU (input + output)",
    )

    if y_axis_metric == "Total TPS/GPU":
        st.markdown("""
        This graph shows the trade-off between **Total TPS/GPU** (input + output tokens/s per GPU) and
        **Output TPS/User** (throughput per user).
        """)
    else:
        st.markdown("""
        This graph shows the trade-off between **Output TPS/GPU** (decode tokens/s per GPU) and
        **Output TPS/User** (throughput per user).
        """)

    pareto_fig = create_pareto_graph(
        df,
        selected_runs,
        pareto_options["show_cutoff"],
        pareto_options["cutoff_value"],
        pareto_options["show_frontier"],
        y_axis_metric,
        run_legend_labels,
    )
    pareto_fig.update_xaxes(showgrid=True)
    pareto_fig.update_yaxes(showgrid=True)

    # Use a key counter so we can reset the chart widget (clears selection)
    if "pareto_chart_gen" not in st.session_state:
        st.session_state.pareto_chart_gen = 0

    event = st.plotly_chart(
        pareto_fig,
        on_select="rerun",
        selection_mode=["points"],
        key=f"pareto_main_{st.session_state.pareto_chart_gen}",
    )

    # Point comparison on selection (shift+click or box/lasso select 2 points)
    points = event.selection.points if event and event.selection else []
    if len(points) == 1:
        st.info("Shift+click another point to compare performance.")
    elif len(points) == 2:
        # Track swap state so user can flip baseline
        if "pareto_swapped" not in st.session_state:
            st.session_state.pareto_swapped = False

        if st.session_state.pareto_swapped:
            a, b = points[1], points[0]
        else:
            a, b = points[0], points[1]

        a_name, a_conc, a_tps_user, a_tps_gpu = a["customdata"]
        b_name, b_conc, b_tps_user, b_tps_gpu = b["customdata"]

        def _pct(base, comp):
            if base == 0 or base is None:
                return None
            return ((comp - base) / base) * 100

        pct_tps_user = _pct(a_tps_user, b_tps_user)
        pct_tps_gpu = _pct(a_tps_gpu, b_tps_gpu)

        def _fmt_delta(pct):
            if pct is None:
                return "N/A"
            color = "green" if pct >= 0 else "red"
            sign = "+" if pct >= 0 else ""
            return f":{color}[{sign}{pct:.1f}%]"

        col_a, col_swap, col_delta, col_b = st.columns([2, 0.5, 1, 2])
        with col_a:
            st.markdown("**Point A (baseline)**")
            st.markdown(
                f"**{a_name}** @ concurrency {a_conc}  \n"
                f"TPS/User: `{a_tps_user:.2f}` · {y_axis_metric}: `{a_tps_gpu:.2f}`"
            )
        with col_swap:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Swap", key="swap_pareto_points"):
                st.session_state.pareto_swapped = not st.session_state.pareto_swapped
                st.rerun()
        with col_delta:
            st.markdown("**% Change (A→B)**")
            st.markdown(f"TPS/User: {_fmt_delta(pct_tps_user)}")
            st.markdown(f"{y_axis_metric}: {_fmt_delta(pct_tps_gpu)}")
        with col_b:
            st.markdown("**Point B**")
            st.markdown(
                f"**{b_name}** @ concurrency {b_conc}  \n"
                f"TPS/User: `{b_tps_user:.2f}` · {y_axis_metric}: `{b_tps_gpu:.2f}`"
            )
    elif len(points) > 2:
        st.warning("Select exactly 2 points to compare. Clear selection and try again.")

    # Clear selection button (visible when any points are selected)
    if len(points) >= 1:
        if st.button("Clear selection", key="clear_pareto_selection"):
            st.session_state.pareto_chart_gen += 1
            st.session_state.pareto_swapped = False
            st.rerun()

    # Debug info for frontier
    if pareto_options["show_frontier"]:
        frontier_points = calculate_pareto_frontier(df, y_axis_metric)
        st.caption(f"🔍 Debug: Frontier has {len(frontier_points)} points across {len(df)} total data points")

        if len(frontier_points) > 0:
            with st.expander("View Frontier Points Details"):
                frontier_df = pd.DataFrame(frontier_points, columns=["Output TPS/User", "Output TPS/GPU"])
                st.dataframe(frontier_df, width="stretch")

    # Data export button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.download_button(
            label="📥 Download Data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_data.csv",
            mime="text/csv",
            width="stretch",
        )

    # Metric calculation documentation
    st.divider()
    st.markdown("### 📊 Metric Calculations")
    st.markdown("""
    **How each metric is calculated:**

    **Output TPS/GPU** (Throughput Efficiency):
    """)
    st.latex(r"\text{Output TPS/GPU} = \frac{\text{Total Output Throughput (tokens/s)}}{\text{Total Number of GPUs}}")
    st.markdown("""
    *This measures how efficiently each GPU is being utilized for token generation.*

    **Output TPS/User** (Per-User Generation Rate):
    """)
    st.latex(r"\text{Output TPS/User} = \frac{1000}{\text{Mean TPOT (ms)}}")
    st.markdown("""
    *Where TPOT (Time Per Output Token) is the average time between consecutive output tokens.
    This represents the actual token generation rate experienced by each user, independent of concurrency.*
    """)
