# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Configuration reader module for parsing node config files
"""

import json
import logging
import os
from typing import Any

import pandas as pd

from .cache_manager import CacheManager
from .models import NodeConfig, ParsedCommandInfo

# Configure logging
logger = logging.getLogger(__name__)


def validate_config_structure(config: dict[str, Any], config_path: str) -> None:
    """Validate config structure and log warnings if format changed.

    This helps debug when log structure changes in the future.
    """
    expected_keys = ["config", "gpu_info", "environment"]
    missing_keys = [key for key in expected_keys if key not in config]

    if missing_keys:
        logger.warning(
            f"Config at {config_path} missing expected keys: {missing_keys}. "
            f"Available keys: {list(config.keys())}. "
            f"Log structure may have changed."
        )

    # Validate nested structure
    if "config" in config and "server_args" not in config["config"]:
        logger.warning(
            f"Config at {config_path} missing 'server_args' in 'config'. "
            f"Available keys in config: {list(config['config'].keys())}"
        )

    if "gpu_info" in config and "gpus" not in config["gpu_info"]:
        logger.warning(
            f"Config at {config_path} missing 'gpus' in 'gpu_info'. "
            f"Available keys: {list(config['gpu_info'].keys())}"
        )


def read_config_file(config_path: str) -> NodeConfig | None:
    """Read a single config JSON file and validate structure."""
    try:
        with open(config_path) as f:
            config = json.load(f)
            validate_config_structure(config, config_path)
            return config
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return None


def get_all_configs(run_path: str) -> list[NodeConfig]:
    """Get all config files from a run directory.

    Args:
        run_path: Path to the run directory containing *_config.json files

    Returns:
        List of config dictionaries with 'filename' added
    """
    configs = []
    if not os.path.exists(run_path):
        logger.error(f"Run path does not exist: {run_path}")
        return configs

    for file in os.listdir(run_path):
        if file.endswith("_config.json"):
            config_path = os.path.join(run_path, file)
            config = read_config_file(config_path)
            if config:
                config["filename"] = file
                configs.append(config)

    if not configs:
        logger.warning(f"No config files (*_config.json) found in {run_path}")

    return configs


def extract_node_info(config: dict) -> dict:
    """Extract relevant node information from config."""
    info = {
        "node_name": config.get("filename", "Unknown").replace("_config.json", ""),
    }

    # GPU info
    if "gpu_info" in config:
        gpu_info = config["gpu_info"]
        info["gpu_count"] = gpu_info.get("count", "N/A")
        if "gpus" in gpu_info and len(gpu_info["gpus"]) > 0:
            info["gpu_name"] = gpu_info["gpus"][0].get("name", "N/A")
            info["gpu_memory"] = gpu_info["gpus"][0].get("memory_total", "N/A")
            info["driver_version"] = gpu_info["gpus"][0].get("driver_version", "N/A")

    # Server args
    if "config" in config and "server_args" in config["config"]:
        server_args = config["config"]["server_args"]
        info["tp_size"] = server_args.get("tp_size", "N/A")
        info["dp_size"] = server_args.get("dp_size", "N/A")
        info["pp_size"] = server_args.get("pp_size", "N/A")
        info["attention_backend"] = server_args.get("attention_backend", "N/A")
        info["kv_cache_dtype"] = server_args.get("kv_cache_dtype", "N/A")
        info["max_total_tokens"] = server_args.get("max_total_tokens", "N/A")
        info["chunked_prefill_size"] = server_args.get("chunked_prefill_size", "N/A")
        info["disaggregation_mode"] = server_args.get("disaggregation_mode", "N/A")
        info["context_length"] = server_args.get("context_length", "N/A")

    # Model info
    if "config" in config and "server_args" in config["config"]:
        server_args = config["config"]["server_args"]
        info["model"] = server_args.get("served_model_name", "N/A")

    return info


def get_run_summary(run_path: str) -> dict:
    """Get a comprehensive summary of a run's configuration."""
    configs = get_all_configs(run_path)

    if not configs:
        return {"error": "No config files found"}

    summary: dict[str, Any] = {"num_nodes": len(configs), "nodes": []}

    # Extract info from each node
    for config in configs:
        node_info = extract_node_info(config)
        summary["nodes"].append(node_info)

    # Get common configuration (from first node)
    if configs:
        first_config = configs[0]
        if "config" in first_config and "server_args" in first_config["config"]:
            server_args = first_config["config"]["server_args"]
            summary["model"] = server_args.get("served_model_name", "N/A")
            summary["attention_backend"] = server_args.get("attention_backend", "N/A")
            summary["kv_cache_dtype"] = server_args.get("kv_cache_dtype", "N/A")

        if (
            "gpu_info" in first_config
            and "gpus" in first_config["gpu_info"]
            and len(first_config["gpu_info"]["gpus"]) > 0
        ):
            summary["gpu_type"] = first_config["gpu_info"]["gpus"][0].get("name", "N/A")

    return summary


def format_config_for_display(run_path: str) -> dict:
    """Format configuration information for display in Streamlit.

    Returns a dictionary with structured data for better display.
    """
    summary = get_run_summary(run_path)

    if "error" in summary:
        return {"error": summary["error"]}

    # Group nodes by type
    prefill_nodes = []
    decode_nodes = []
    frontend_nodes = []
    other_nodes = []

    for node in summary.get("nodes", []):
        node_name = node.get("node_name", "Unknown")
        if "prefill" in node_name.lower():
            prefill_nodes.append(node)
        elif "decode" in node_name.lower():
            decode_nodes.append(node)
        elif "frontend" in node_name.lower() or "nginx" in node_name.lower():
            frontend_nodes.append(node)
        else:
            other_nodes.append(node)

    return {
        "summary": {
            "num_nodes": summary.get("num_nodes", "N/A"),
            "model": summary.get("model", "N/A"),
            "gpu_type": summary.get("gpu_type", "N/A"),
            "attention_backend": summary.get("attention_backend", "N/A"),
            "kv_cache_dtype": summary.get("kv_cache_dtype", "N/A"),
        },
        "prefill_nodes": prefill_nodes,
        "decode_nodes": decode_nodes,
        "frontend_nodes": frontend_nodes,
        "other_nodes": other_nodes,
    }


def get_environment_variables(config: dict) -> dict:
    """Extract and categorize environment variables.

    Returns dict organized by category: NCCL, SGLANG, CUDA, MC (Mooncake), etc.
    """
    if "environment" not in config:
        return {}

    env = config["environment"]

    categories = {"NCCL": {}, "SGLANG": {}, "CUDA": {}, "Mooncake": {}, "OMPI": {}, "Other": {}}

    for key, value in env.items():
        if key.startswith("NCCL_"):
            categories["NCCL"][key] = value
        elif key.startswith("SGLANG_") or key == "DYN_SKIP_SGLANG_LOG_FORMATTING" or key == "SGL_FORCE_SHUTDOWN":
            categories["SGLANG"][key] = value
        elif key.startswith("CUDA_"):
            categories["CUDA"][key] = value
        elif key.startswith("MC_"):
            categories["Mooncake"][key] = value
        elif key.startswith("OMPI_"):
            categories["OMPI"][key] = value
        else:
            categories["Other"][key] = value

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def get_command_line_args(config: dict) -> list[str]:
    """Extract actual command line args from runtime_info.

    These are the ACTUAL args that were passed, not the processed server_args
    which may include defaults.

    Args:
        config: Node config dict

    Returns:
        List of command line arguments, empty if not found
    """
    if "runtime_info" not in config:
        return []

    runtime_info = config["runtime_info"]
    return runtime_info.get("command_line_args", [])


def parse_command_line_to_dict(cmd_args: list[str]) -> dict[str, str]:
    """Parse command line args list into a dict of flag->value pairs.

    Args:
        cmd_args: List like ["--flag1", "value1", "--flag2", "--flag3", "value3"]

    Returns:
        Dict like {"flag1": "value1", "flag2": "True", "flag3": "value3"}
    """
    parsed = {}
    i = 0

    while i < len(cmd_args):
        arg = cmd_args[i]

        # Skip non-flag arguments (like script name)
        if not arg.startswith("--"):
            i += 1
            continue

        # Remove -- prefix
        flag = arg[2:]

        # Check if next arg is a value or another flag
        if i + 1 < len(cmd_args) and not cmd_args[i + 1].startswith("--"):
            parsed[flag] = cmd_args[i + 1]
            i += 2
        else:
            # Boolean flag (no value)
            parsed[flag] = "True"
            i += 1

    return parsed


def parse_command_line_from_err(run_path: str) -> ParsedCommandInfo:
    """Parse .err/.out files to find explicitly set flags and service topology.

    Uses parquet caching to avoid re-parsing on subsequent loads.

    Expected file format:
    - Filename pattern: <node>_<service>_<id>.err or .out (e.g., watchtower-navy-cn01_prefill_w0.err)
    - Contains line with: python3 -m ... sglang --flag1 value1 --flag2 value2 ...

    Args:
        run_path: Path to the run directory containing .err/.out files

    Returns:
        {
            'explicit_flags': set of flag names that were explicitly set,
            'services': {node_name: [service_types]}
        }
    """
    import os
    import re

    # Initialize cache manager
    cache_mgr = CacheManager(run_path)
    source_patterns = ["*.err", "*.out"]

    # Try to load from cache first
    if cache_mgr.is_cache_valid("config_topology", source_patterns):
        cached_df = cache_mgr.load_from_cache("config_topology")
        if cached_df is not None and not cached_df.empty:
            # Reconstruct data from cache
            explicit_flags = set(cached_df[cached_df["type"] == "flag"]["name"].tolist())

            # Reconstruct services dict
            services: dict[str, list[str]] = {}
            service_rows = cached_df[cached_df["type"] == "service"]
            for _, row in service_rows.iterrows():
                node_name = row["node_name"]
                service_type = row["name"]
                if node_name not in services:
                    services[node_name] = []
                services[node_name].append(service_type)

            logger.info(f"Loaded {len(explicit_flags)} flags and {len(services)} nodes from cache")
            return {"explicit_flags": explicit_flags, "services": services}

    # Cache miss - parse from .err/.out files
    explicit_flags: set = set()
    services: dict[str, list[str]] = {}
    log_files_found = 0
    commands_found = 0

    if not os.path.exists(run_path):
        logger.error(f"Run path does not exist: {run_path}")
        return {"explicit_flags": explicit_flags, "services": services}

    # Scan all .err and .out files
    for filename in os.listdir(run_path):
        if filename.endswith(".err") or filename.endswith(".out"):
            log_files_found += 1
            filepath = os.path.join(run_path, filename)

            # Extract node name and service type from filename
            # Pattern: watchtower-navy-cn01_prefill_w0.err or r02-p01-dgx-c11_prefill_w0.out -> cn01/c11, prefill
            # Use greedy match (.+) since node names can contain underscores
            match = re.match(r"(.+)_(prefill|decode|frontend|nginx|nats|etcd)", filename)
            if match:
                node_name = match.group(1).replace("watchtower-navy-", "")
                service_type = match.group(2)

                if node_name not in services:
                    services[node_name] = []
                services[node_name].append(service_type)
            else:
                logger.debug(
                    f"Could not parse service type from filename: {filename}. "
                    f"Expected pattern: <node>_<service>_<id>.err or .out"
                )

            # Look for command line to extract explicit flags
            try:
                with open(filepath) as f:
                    for line in f:
                        if "python" in line and "sglang" in line and "--" in line:
                            # Extract all --flag-name patterns
                            flags = re.findall(r"--([a-z0-9-]+)", line)
                            explicit_flags.update(flags)
                            commands_found += 1
                            break  # Only need to find the command once per file
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue

    # Validation warnings
    if log_files_found == 0:
        logger.warning(f"No .err/.out files found in {run_path}. Cannot determine service topology.")

    if commands_found == 0 and log_files_found > 0:
        logger.warning(
            f"Found {log_files_found} log files but no sglang commands. "
            f"Expected format: 'python3 -m ... sglang --flag ...' "
            f"Log structure may have changed."
        )

    logger.info(
        f"Parsed {log_files_found} log files, found {commands_found} commands, "
        f"{len(explicit_flags)} unique flags, {len(services)} nodes"
    )

    # Save to cache
    cache_rows = []
    # Store flags
    for flag in explicit_flags:
        cache_rows.append({"type": "flag", "name": flag, "node_name": None})
    # Store services
    for node_name, service_types in services.items():
        for service_type in service_types:
            cache_rows.append({"type": "service", "name": service_type, "node_name": node_name})

    if cache_rows:
        cache_df = pd.DataFrame(cache_rows)
        cache_mgr.save_to_cache("config_topology", cache_df, source_patterns)

    return {"explicit_flags": explicit_flags, "services": services}
