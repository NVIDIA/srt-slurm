# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Configuration Details Tab
"""

import os

import streamlit as st

from analysis.srtlog import (
    format_config_for_display,
    parse_command_line_from_err,
)
from analysis.srtlog.config_reader import (
    get_all_configs,
    get_command_line_args,
    get_environment_variables,
    parse_command_line_to_dict,
)


def render(filtered_runs: list):
    """Render the configuration details tab.

    Args:
        filtered_runs: List of BenchmarkRun objects
    """
    st.subheader("Run Configuration Details")

    for idx, run in enumerate(filtered_runs):
        # Build run_id based on mode
        if run.metadata.is_aggregated:
            run_id = f"{run.job_id}_{run.metadata.agg_workers}A_{run.metadata.run_date}"
        else:
            run_id = (
                f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
            )

        run_path = run.metadata.path
        run_date = run.metadata.formatted_date

        # Add index to ensure unique keys even if duplicates exist
        unique_key = f"{run_id}_{idx}"

        expander_title = f"🔧 Job {run.job_id}"
        if run_date:
            expander_title += f" ({run_date})"

        with st.expander(expander_title, expanded=len(filtered_runs) == 1):
            if not run_path or not os.path.exists(run_path):
                st.warning("Configuration files not found for this run.")
                continue

            # Get structured config data
            config_data = format_config_for_display(run_path)

            if "error" in config_data:
                st.error(config_data["error"])
                continue

            # Compact overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", config_data["summary"]["num_nodes"])
            with col2:
                st.metric("GPU", config_data["summary"]["gpu_type"])
            with col3:
                st.metric("ISL/OSL", f"{run.profiler.isl}/{run.profiler.osl}")
            with col4:
                gpu_type_suffix = f" ({run.metadata.gpu_type})" if run.metadata.gpu_type else ""
                st.metric("Profiler", f"{run.profiler.profiler_type}{gpu_type_suffix}")

            st.caption(f"Model: {config_data['summary']['model']}")
            st.divider()

            # Use tabs for organization
            config_tab1, config_tab2, config_tab3 = st.tabs(["📋 Topology", "⚙️ Node Config", "🌍 Environment"])

            all_configs = get_all_configs(run_path)

            with config_tab1:
                _render_topology_tab(run_path)

            with config_tab2:
                _render_node_config_tab(all_configs, unique_key)

            with config_tab3:
                _render_environment_tab(all_configs, unique_key)


def _render_topology_tab(run_path):
    """Render topology information."""
    parsed_data = parse_command_line_from_err(run_path)
    physical_nodes = parsed_data.get("services", {})

    if physical_nodes:
        # Count by service type
        service_counts = {}
        for services in physical_nodes.values():
            for svc in set(services):
                service_counts[svc] = service_counts.get(svc, 0) + 1

        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prefill Nodes", service_counts.get("prefill", 0))
        with col2:
            st.metric("Decode Nodes", service_counts.get("decode", 0))
        with col3:
            st.metric("Frontend Nodes", service_counts.get("frontend", 0))

        # Compact node grid (collapsed by default)
        with st.expander(f"View all {len(physical_nodes)} physical nodes", expanded=False):
            cols_per_row = 4
            node_items = sorted(physical_nodes.items())
            for i in range(0, len(node_items), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, (phys_node, service_types) in enumerate(node_items[i : i + cols_per_row]):
                    with cols[j]:
                        st.caption(f"**{phys_node}**")
                        for svc in sorted(set(service_types)):
                            emoji = {
                                "prefill": "📤",
                                "decode": "📥",
                                "frontend": "🌐",
                                "nginx": "🌐",
                            }.get(svc, "⚙️")
                            st.caption(f"{emoji} {svc}")
    else:
        st.info("No topology information available")


def _render_node_config_tab(all_configs, run_id):
    """Render node configuration."""
    if not all_configs:
        st.info("No config files found")
        return

    # Create node selection dropdown
    node_names = [
        config.get("filename", f"Node {i}")
        .replace("_config.json", "")
        .replace("watchtower-aqua-", "")
        .replace("watchtower-navy-", "")
        for i, config in enumerate(all_configs)
    ]

    # Group by type
    prefill_nodes_list = [(i, name) for i, name in enumerate(node_names) if "prefill" in name.lower()]
    decode_nodes_list = [(i, name) for i, name in enumerate(node_names) if "decode" in name.lower()]
    other_nodes = [
        (i, name) for i, name in enumerate(node_names) if "prefill" not in name.lower() and "decode" not in name.lower()
    ]

    # Create categorized options
    node_options = []
    if prefill_nodes_list:
        node_options.extend([f"📤 {name}" for _, name in prefill_nodes_list])
    if decode_nodes_list:
        node_options.extend([f"📥 {name}" for _, name in decode_nodes_list])
    if other_nodes:
        node_options.extend([f"🖥️ {name}" for _, name in other_nodes])

    # Map back to indices
    option_to_idx = {}
    all_indexed = prefill_nodes_list + decode_nodes_list + other_nodes
    for i, (idx, _) in enumerate(all_indexed):
        option_to_idx[node_options[i]] = idx

    selected_option = st.selectbox("Select node", options=node_options, key=f"config_node_{run_id}")
    selected_idx = option_to_idx[selected_option]
    selected_config = all_configs[selected_idx]

    # Show command line args
    cmd_args = get_command_line_args(selected_config)

    if cmd_args:
        cmd_dict = parse_command_line_to_dict(cmd_args)

        st.markdown(f"**Command Line Arguments** ({len(cmd_dict)} flags)")
        st.caption("Actual flags passed on command line for this node")

        # Display all flags in 3 columns
        num_items = len(cmd_dict)
        items_per_col = (num_items + 2) // 3

        col1, col2, col3 = st.columns(3)

        for idx, (key, value) in enumerate(sorted(cmd_dict.items())):
            col_idx = idx // items_per_col
            display_val = str(value)[:60] + "..." if len(str(value)) > 60 else value

            if col_idx == 0:
                with col1:
                    st.caption(f"`{key}`: **{display_val}**")
            elif col_idx == 1:
                with col2:
                    st.caption(f"`{key}`: **{display_val}**")
            else:
                with col3:
                    st.caption(f"`{key}`: **{display_val}**")
    else:
        st.info("No command line args found")


def _render_environment_tab(all_configs, run_id):
    """Render environment variables."""
    if not all_configs:
        st.info("No config files found")
        return

    # Use same node selector
    node_names = [
        config.get("filename", f"Node {i}")
        .replace("_config.json", "")
        .replace("watchtower-aqua-", "")
        .replace("watchtower-navy-", "")
        for i, config in enumerate(all_configs)
    ]

    selected_name = st.selectbox("Select node", options=node_names, key=f"env_node_{run_id}")
    selected_config = all_configs[node_names.index(selected_name)]
    env_vars = get_environment_variables(selected_config)

    if env_vars:
        for category, vars_dict in env_vars.items():
            with st.expander(f"{category} ({len(vars_dict)} vars)", expanded=category in ["NCCL", "SGLANG"]):
                for key, value in sorted(vars_dict.items()):
                    st.caption(f"`{key}`: {value}")
    else:
        st.info("No environment variables found")
