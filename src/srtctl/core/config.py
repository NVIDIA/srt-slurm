#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config loading and resolution with srtslurm.yaml integration.

This module provides:
- load_config(): Load YAML config, apply cluster defaults, return typed SrtConfig
- get_srtslurm_setting(): Get cluster-wide settings
"""

import copy
import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml.comments import CommentedMap

from .schema import ClusterConfig, SrtConfig

logger = logging.getLogger(__name__)


def load_cluster_config() -> dict[str, Any] | None:
    """
    Load cluster configuration from srtslurm.yaml if it exists.

    Searches for srtslurm.yaml in order:
    1. SRTSLURM_CONFIG environment variable (if set)
    2. Current working directory
    3. Parent directories up to 3 levels

    Returns None if file doesn't exist (graceful degradation).
    """
    # Check env var first (highest priority)
    env_config = os.environ.get("SRTSLURM_CONFIG")
    if env_config:
        env_path = Path(env_config)
        if env_path.exists():
            cluster_config_path = env_path
            logger.debug(f"Using srtslurm.yaml from SRTSLURM_CONFIG: {cluster_config_path}")
        else:
            logger.warning(f"SRTSLURM_CONFIG set but file not found: {env_config}")
            return None
    else:
        # Search paths
        search_paths = [
            Path.cwd() / "srtslurm.yaml",
            Path.cwd().parent / "srtslurm.yaml",
            Path.cwd().parent.parent / "srtslurm.yaml",
        ]

        cluster_config_path = None
        for path in search_paths:
            if path.exists():
                cluster_config_path = path
                break

        if not cluster_config_path:
            logger.debug("No srtslurm.yaml found - using config as-is")
            return None

    try:
        with open(cluster_config_path) as f:
            raw_config = yaml.safe_load(f)

        # Validate with marshmallow schema
        schema = ClusterConfig.Schema()
        validated = schema.load(raw_config)
        logger.debug(f"Loaded cluster config from {cluster_config_path}")

        # Dump back to dict for compatibility
        return schema.dump(validated)
    except Exception as e:
        logger.warning(f"Failed to load or validate srtslurm.yaml: {e}")
        return None


def resolve_config_with_defaults(user_config: dict[str, Any], cluster_config: dict[str, Any] | None) -> dict[str, Any]:
    """
    Resolve user config by applying cluster defaults and aliases.

    This applies:
    1. Default SLURM settings (account, partition, time_limit)
    2. Model path alias resolution
    3. Container alias resolution

    Args:
        user_config: User's YAML config as dict
        cluster_config: Cluster defaults from srtslurm.yaml (or None)

    Returns:
        Resolved config dict with all defaults applied
    """
    # Deep copy to avoid mutating original
    config = copy.deepcopy(user_config)

    if cluster_config is None:
        return config

    # Apply SLURM defaults
    slurm = config.setdefault("slurm", {})
    if "account" not in slurm and cluster_config.get("default_account"):
        slurm["account"] = cluster_config["default_account"]
        logger.debug(f"Applied default account: {slurm['account']}")

    if "partition" not in slurm and cluster_config.get("default_partition"):
        slurm["partition"] = cluster_config["default_partition"]
        logger.debug(f"Applied default partition: {slurm['partition']}")

    if "time_limit" not in slurm and cluster_config.get("default_time_limit"):
        slurm["time_limit"] = cluster_config["default_time_limit"]
        logger.debug(f"Applied default time_limit: {slurm['time_limit']}")

    # Resolve model path alias
    model = config.get("model", {})
    model_path = model.get("path", "")

    model_paths = cluster_config.get("model_paths")
    if model_paths and model_path in model_paths:
        resolved_path = model_paths[model_path]
        model["path"] = resolved_path
        logger.debug(f"Resolved model alias '{model_path}' -> '{resolved_path}'")

    # Resolve container alias
    container = model.get("container", "")

    containers = cluster_config.get("containers")
    if containers and container in containers:
        resolved_container = containers[container]
        model["container"] = resolved_container
        logger.debug(f"Resolved container alias '{container}' -> '{resolved_container}'")

    # Apply reporting defaults (if not specified in user config)
    if "reporting" not in config and cluster_config.get("reporting"):
        config["reporting"] = cluster_config["reporting"]
        logger.debug("Applied cluster reporting config")

    # Resolve extra_mount host path aliases through model_paths
    extra_mounts = config.get("extra_mount", [])
    if model_paths and extra_mounts:
        resolved_mounts = []
        for mount_spec in extra_mounts:
            host_path, container_path = mount_spec.split(":", 1)
            if host_path in model_paths:
                resolved_host = model_paths[host_path]
                resolved_mounts.append(f"{resolved_host}:{container_path}")
                logger.debug(f"Resolved extra_mount alias '{host_path}' -> '{resolved_host}'")
            else:
                resolved_mounts.append(mount_spec)
        config["extra_mount"] = resolved_mounts

    # Resolve frontend nginx_container alias
    frontend = config.get("frontend", {})
    nginx_container = frontend.get("nginx_container", "")

    if containers and nginx_container in containers:
        resolved_nginx = containers[nginx_container]
        frontend["nginx_container"] = resolved_nginx
        config["frontend"] = frontend
        logger.debug(f"Resolved nginx_container alias '{nginx_container}' -> '{resolved_nginx}'")

    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep-merge two dicts. Override values take precedence.

    - dict: recursive merge
    - list: full replacement (no append)
    - scalar: override replaces base
    - None value: deletes the key from result
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _collect_list_lengths(d: dict[str, Any]) -> list[int]:
    """Return the length of every list-valued leaf in d (recursive)."""
    lengths: list[int] = []
    for v in d.values():
        if isinstance(v, list):
            lengths.append(len(v))
        elif isinstance(v, dict):
            lengths.extend(_collect_list_lengths(v))
    return lengths


def _determine_zip_length(zip_dict: dict[str, Any]) -> int:
    """Determine N for a zip_override section, enforcing broadcast rules.

    - Length-1 lists are broadcast to N.
    - All other lists must share the same length N.
    - Raises ValueError if incompatible lengths are found.
    """
    lengths = _collect_list_lengths(zip_dict)
    if not lengths:
        raise ValueError("zip_override section contains no list values — nothing to zip")
    if any(n == 0 for n in lengths):
        raise ValueError("zip_override contains an empty list — cannot zip zero-length lists")
    non_broadcast = [n for n in lengths if n != 1]
    if not non_broadcast:
        return 1  # every list has length 1; N=1
    unique = set(non_broadcast)
    if len(unique) > 1:
        raise ValueError(
            f"Incompatible zip lengths {sorted(unique)}. All lists must have the same length or length 1 (broadcast)."
        )
    return unique.pop()


def _apply_zip_slice(d: dict[str, Any], index: int) -> dict[str, Any]:
    """Replace each list-valued leaf with its index-th element.

    Length-1 lists are broadcast (always use element 0).
    Scalar values pass through unchanged (implicitly broadcast).
    List-of-list elements become literal list values in the result.
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, list):
            result[k] = v[0 if len(v) == 1 else index]
        elif isinstance(v, dict):
            result[k] = _apply_zip_slice(v, index)
        else:
            result[k] = v
    return result


def expand_zip_override(
    group_name: str,
    zip_dict: dict[str, Any],
    base: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Expand a zip_override_* section into N (suffix, config_dict) tuples.

    Each list-valued leaf in zip_dict is a zip dimension.
    Length-1 lists are broadcast to N. All other list lengths must equal N.
    Suffix is '{group_name}_{i}' for i in range(N).

    If the zip_dict provides a 'name' list, each variant uses the corresponding
    name. Otherwise the name is auto-generated as '{base_name}_{group_name}_{i}'.
    """
    n = _determine_zip_length(zip_dict)
    base_name = base.get("name", "unnamed")
    # Only suppress auto-naming when the user explicitly provides a name list.
    # A scalar name in zip_dict would broadcast to every variant (duplicates),
    # so we auto-generate in that case too.
    has_name_list = isinstance(zip_dict.get("name"), list)
    results: list[tuple[str, dict[str, Any]]] = []
    for i in range(n):
        sliced = _apply_zip_slice(zip_dict, i)
        merged = deep_merge(base, sliced)
        if not has_name_list:
            merged["name"] = f"{base_name}_{group_name}_{i}"
        suffix = f"{group_name}_{i}"
        results.append((suffix, merged))
    return results


def _expand_wildcard(
    raw_config: dict[str, Any],
    pattern: str,
    base: dict[str, Any],
    override_keys: list[str],
    zip_keys: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Expand a glob pattern against all override_* / zip_override_* keys (base always excluded)."""
    all_keys = sorted(override_keys + zip_keys)
    matched = [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
    if not matched:
        available = ", ".join([*override_keys, *[f"{k}[i]" for k in zip_keys]]) or "(none)"
        raise ValueError(f"No variants match '{pattern}'. Available: {available}")

    configs: list[tuple[str, dict[str, Any]]] = []
    for key in matched:
        if key.startswith("zip_override_"):
            group_name = key[len("zip_override_") :]
            configs.extend(expand_zip_override(group_name, raw_config[key], base))
        else:
            suffix = key[len("override_") :]
            override_dict = raw_config[key]
            merged = deep_merge(base, override_dict)
            if "name" not in override_dict:
                base_name = base.get("name", "unnamed")
                merged["name"] = f"{base_name}_{suffix}"
            configs.append((suffix, merged))

    return configs


def generate_override_configs(
    raw_config: dict[str, Any],
    selector: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Expand a raw config with base + override_* + zip_override_* keys into independent configs.

    Args:
        raw_config: Raw YAML dict containing 'base' and optional 'override_*' /
                    'zip_override_*' keys.
        selector: Optional selector:
                    None                        – all override_* and zip_override_* variants (base excluded)
                    "base"                      – base only
                    "override_<name>"           – single override variant
                    "zip_override_<name>"       – all variants in a zip group
                    "zip_override_<name>[N]"    – single variant by 0-based index
                    "<glob>"                    – all matching keys (fnmatch against all override_* and
                                                  zip_override_* names; base always excluded)

    Returns:
        List of (suffix, config_dict) tuples.

    Raises:
        ValueError: If selector specifies a non-existent key or out-of-range index.
    """
    base = raw_config["base"]
    override_keys = sorted(k for k in raw_config if k.startswith("override_"))
    zip_keys = sorted(k for k in raw_config if k.startswith("zip_override_"))

    if selector is not None:
        # zip_override_foo[N] — single variant by index
        m = re.fullmatch(r"(zip_override_[\w-]+)\[(\d+)\]", selector)
        if m:
            zip_key, idx = m.group(1), int(m.group(2))
            if zip_key not in raw_config:
                available = ", ".join(f"{k}[i]" for k in zip_keys) or "(none)"
                raise ValueError(f"'{zip_key}' not found in config. Available zip groups: {available}")
            group_name = zip_key[len("zip_override_") :]
            variants = expand_zip_override(group_name, raw_config[zip_key], base)
            if idx >= len(variants):
                raise ValueError(
                    f"Index [{idx}] out of range for '{zip_key}' "
                    f"(has {len(variants)} variants, valid: 0–{len(variants) - 1})"
                )
            return [variants[idx]]

        if selector == "base":
            return [("base", copy.deepcopy(base))]

        # Wildcard: delegate to glob matching before exact-key lookups
        if "*" in selector or "?" in selector:
            return _expand_wildcard(raw_config, selector, base, override_keys, zip_keys)

        # zip_override_foo — all variants in the group
        if selector.startswith("zip_override_"):
            if selector not in raw_config:
                available = ", ".join(zip_keys) or "(none)"
                raise ValueError(f"'{selector}' not found in config. Available: {available}")
            group_name = selector[len("zip_override_") :]
            return expand_zip_override(group_name, raw_config[selector], base)

        # override_foo — single override variant
        if selector not in raw_config:
            all_selectors = ", ".join([*override_keys, *[f"{k}[i]" for k in zip_keys]]) or "(none)"
            raise ValueError(f"Override '{selector}' not found in config. Available: {all_selectors}")
        suffix = selector[len("override_") :]
        override_dict = raw_config[selector]
        merged = deep_merge(base, override_dict)
        if "name" not in override_dict:
            base_name = base.get("name", "unnamed")
            merged["name"] = f"{base_name}_{suffix}"
        return [(suffix, merged)]

    # selector=None: all overrides + all zip groups (sorted for determinism); base excluded
    configs: list[tuple[str, dict[str, Any]]] = []
    for key in override_keys:
        suffix = key[len("override_") :]
        override_dict = raw_config[key]
        merged = deep_merge(base, override_dict)
        if "name" not in override_dict:
            base_name = base.get("name", "unnamed")
            merged["name"] = f"{base_name}_{suffix}"
        configs.append((suffix, merged))
    for key in zip_keys:
        group_name = key[len("zip_override_") :]
        configs.extend(expand_zip_override(group_name, raw_config[key], base))

    return configs


def resolve_override_yaml(
    config_path: Path,
    selector: str | None = None,
) -> list[tuple[str, Any]]:
    """Expand an override YAML into variants, preserving field order and comments.

    Like :func:`generate_override_configs` but returns ``ruamel.yaml``
    ``CommentedMap`` objects so the output can be serialised with comments
    intact.

    Field ordering rules (same as the merge):
    - Base fields appear first, in base order.
    - New fields from the override section are appended at the end.

    For ``zip_override_*`` variants the per-variant values come from
    :func:`expand_zip_override` (list slicing); base comments are preserved
    while the zip section comments are not (they reference list elements).

    Args:
        config_path: Path to an override YAML file (must have a ``base`` key).
        selector: Optional selector, same syntax as
                  :func:`generate_override_configs`.

    Returns:
        List of ``(suffix, CommentedMap)`` tuples ready for
        :func:`~srtctl.core.yaml_utils.dump_yaml_with_comments`.
    """
    from .yaml_utils import comment_aware_merge, load_yaml_with_comments

    # Load twice: once with comment preservation, once as plain dicts for the
    # existing expansion logic (zip slicing, wildcard, etc.).
    raw_cm = load_yaml_with_comments(config_path)
    with open(config_path) as f:
        raw_plain = yaml.safe_load(f)

    base_cm: Any = raw_cm["base"]

    # Re-use the existing expansion to get fully merged plain dicts.
    plain_variants = generate_override_configs(raw_plain, selector=selector)

    results: list[tuple[str, Any]] = []
    for suffix, merged_plain in plain_variants:
        if suffix == "base":
            # No override applied — return the base CommentedMap as-is.
            results.append(("base", base_cm))
            continue

        override_key = f"override_{suffix}"
        if override_key in raw_cm and isinstance(raw_cm[override_key], CommentedMap):
            # Regular override: merge CommentedMaps so override comments are kept.
            result_cm = comment_aware_merge(base_cm, raw_cm[override_key])
            # Preserve auto-generated fields from the existing override expansion,
            # such as the synthesized name when the override does not set one.
            if "name" in merged_plain:
                result_cm["name"] = merged_plain["name"]
        else:
            # zip_override variant (values were lists → now scalars) or any
            # other case: merge the plain resolved dict into the base CommentedMap
            # so at least base field order and comments are preserved.
            result_cm = comment_aware_merge(base_cm, merged_plain)

        results.append((suffix, result_cm))

    return results


def validate_config_file(path: Path | str) -> list[str]:
    """Validate a recipe YAML, handling both plain and override-format files.

    For plain configs, validates the single config.
    For override configs (has a ``base`` key), expands all variants and
    validates each one.

    Returns:
        List of error strings. Empty list means all variants are valid.
    """
    path = Path(path)
    if not path.exists():
        return [f"{path}: file not found"]

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"{path}: YAML parse error: {e}"]

    if not isinstance(raw, dict):
        return [f"{path}: not a YAML mapping"]

    errors: list[str] = []

    if "base" in raw:
        # Override format — expand and validate each variant
        try:
            variants = generate_override_configs(raw)
        except Exception as e:
            return [f"{path}: failed to expand overrides: {e}"]

        cluster_config = load_cluster_config()
        schema = SrtConfig.Schema()
        for suffix, config_dict in variants:
            resolved = resolve_config_with_defaults(config_dict, cluster_config)
            try:
                schema.load(resolved)
            except Exception as e:
                errors.append(f"{path} [{suffix}]: {e}")
    else:
        # Plain config
        try:
            load_config(path)
        except Exception as e:
            errors.append(f"{path}: {e}")

    return errors


def get_srtslurm_setting(key: str, default: Any = None) -> Any:
    """
    Get a setting from srtslurm.yaml cluster config.

    Args:
        key: Setting key (e.g., 'gpus_per_node', 'network_interface')
        default: Default value if not found

    Returns:
        Setting value or default if not found
    """
    cluster_config = load_cluster_config()
    if cluster_config and key in cluster_config:
        return cluster_config[key]
    return default


def load_config(path: Path | str) -> SrtConfig:
    """
    Load and validate YAML config, applying cluster defaults.

    Returns a fully typed, frozen SrtConfig dataclass ready for use.

    Args:
        path: Path to the YAML configuration file

    Returns:
        SrtConfig frozen dataclass

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load raw user config
    with open(path) as f:
        user_config = yaml.safe_load(f)

    # Load cluster defaults (optional)
    cluster_config = load_cluster_config()

    # Resolve with defaults (applies aliases and default values)
    resolved_config = resolve_config_with_defaults(user_config, cluster_config)

    # Parse with marshmallow schema to get typed SrtConfig
    try:
        schema = SrtConfig.Schema()
        config = schema.load(resolved_config)
        assert isinstance(config, SrtConfig)
        logger.info(f"Loaded config: {config.name}")
        return config
    except Exception as e:
        raise ValueError(f"Invalid config in {path}: {e}") from e
