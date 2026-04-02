# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Cluster configuration reader for SLURM settings
"""

import logging
from pathlib import Path

import tomllib

logger = logging.getLogger(__name__)


def load_cluster_config(config_path: str = "srtslurm.toml") -> dict | None:
    """Load cluster configuration from TOML file.

    Args:
        config_path: Path to srtslurm.toml

    Returns:
        Dict with cluster config, or None if file doesn't exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return None

    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
        return config.get("cluster", {})
    except Exception as e:
        logger.error(f"Failed to load cluster config: {e}")
        return None


def get_cluster_setting(key: str, cli_value=None, config_path: str = "srtslurm.toml"):
    """Get cluster setting with CLI override priority.

    Priority:
    1. CLI argument (if provided and not None)
    2. Config file value
    3. None (caller should handle missing required settings)

    Args:
        key: Setting key (e.g., 'account', 'partition', 'network_interface')
        cli_value: Value from CLI argument (or None if not provided)
        config_path: Path to config file

    Returns:
        Setting value or None if not found
    """
    # CLI override takes precedence
    if cli_value is not None:
        return cli_value

    # Try config file
    config = load_cluster_config(config_path)
    if config:
        return config.get(key)

    return None


def validate_cluster_settings(account, partition, network_interface, config_path="srtslurm.toml"):
    """Validate that required cluster settings are present.

    Args:
        account: SLURM account (from CLI or config)
        partition: SLURM partition (from CLI or config)
        network_interface: Network interface (from CLI or config)
        config_path: Path to config file

    Returns:
        Tuple of (account, partition, network_interface) with config fallbacks applied

    Raises:
        ValueError: If required settings are missing
    """
    # Get values with config fallback
    final_account = get_cluster_setting("account", account, config_path)
    final_partition = get_cluster_setting("partition", partition, config_path)
    final_network_interface = get_cluster_setting("network_interface", network_interface, config_path)

    # Validate required settings
    missing = []
    if not final_account:
        missing.append("--account (or [cluster].account in srtslurm.toml)")
    if not final_partition:
        missing.append("--partition (or [cluster].partition in srtslurm.toml)")
    if not final_network_interface:
        missing.append("--network-interface (or [cluster].network_interface in srtslurm.toml)")

    if missing:
        raise ValueError(f"Missing required cluster settings: {', '.join(missing)}")

    return final_account, final_partition, final_network_interface
