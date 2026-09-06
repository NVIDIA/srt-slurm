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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml.comments import CommentedMap

from .lockfile import verify_lock_integrity
from .schema import ClusterConfig, SrtConfig

logger = logging.getLogger(__name__)


def find_cluster_config_path() -> Path | None:
    """Locate srtslurm.yaml using the standard search order."""
    # Check env var first (highest priority)
    env_config = os.environ.get("SRTSLURM_CONFIG")
    if env_config:
        env_path = Path(env_config)
        if env_path.exists():
            logger.debug(f"Using srtslurm.yaml from SRTSLURM_CONFIG: {env_path}")
            return env_path
        logger.warning(f"SRTSLURM_CONFIG set but file not found: {env_config}")
        return None

    search_paths = [
        Path.cwd() / "srtslurm.yaml",
        Path.cwd().parent / "srtslurm.yaml",
        Path.cwd().parent.parent / "srtslurm.yaml",
    ]
    for path in search_paths:
        if path.exists():
            return path

    logger.debug("No srtslurm.yaml found - using config as-is")
    return None


def load_cluster_config() -> dict[str, Any] | None:
    """
    Load cluster configuration from srtslurm.yaml if it exists.

    Returns None if file doesn't exist (graceful degradation).
    """
    cluster_config_path = find_cluster_config_path()
    if not cluster_config_path:
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
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load or validate srtslurm.yaml: {e}")
        return None


# Keys whose string values name a container image. Any such leaf anywhere in a
# recipe resolves against the cluster `containers:` alias map.
CONTAINER_ALIAS_KEYS: frozenset[str] = frozenset({"container", "container_image", "image", "nginx_container"})

# Sub-trees the alias walker never enters: `identity` declares the pullable image a
# run *should* be using (verification only, never an alias); the rest are free-form
# maps (environment variables, engine flags, mounts) where a key happening to be
# called `image` is user data, not a container reference.
_CONTAINER_ALIAS_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "identity",
        "environment",
        "prefill_environment",
        "decode_environment",
        "aggregated_environment",
        "env",
        "args",
        "extra_args",
        "prefill_extra_args",
        "decode_extra_args",
        "aggregated_extra_args",
        "container_mounts",
        "sbatch_directives",
        "srun_options",
        "sglang_config",
        "vllm_config",
        "trtllm_config",
        "mocker_config",
        "store_config",
    }
)


def resolve_container_aliases(config: dict[str, Any], containers: Mapping[str, str]) -> list[str]:
    """Replace every container-alias leaf in ``config`` with its ``containers:`` value, in place.

    Walks the whole recipe once. A leaf is any string under a key in
    :data:`CONTAINER_ALIAS_KEYS` whose value is a key of ``containers``; literal
    paths and registry URIs are left alone. This is the single place container
    aliases resolve (model, frontend, nginx, benchmark, exporters, Mooncake,
    services), so a new block that names an image needs no resolver code.

    Returns one human-readable note per resolved leaf.
    """
    notes: list[str] = []

    def walk(node: Any, path: tuple[Any, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _CONTAINER_ALIAS_SKIP_KEYS:
                    continue
                if key in CONTAINER_ALIAS_KEYS and isinstance(value, str) and value in containers:
                    node[key] = containers[value]
                    dotted = ".".join(str(part) for part in (*path, key))
                    notes.append(f"Resolved container alias {dotted}: '{value}' -> '{containers[value]}'")
                elif isinstance(value, dict | list):
                    walk(value, (*path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*path, index))

    walk(config, ())
    return notes


def resolve_config_with_defaults(user_config: dict[str, Any], cluster_config: dict[str, Any] | None) -> dict[str, Any]:
    """
    Resolve user config by applying cluster defaults and aliases.

    This applies:
    1. Default SLURM settings (account, partition, time_limit)
    2. Model path alias resolution
    3. Container alias resolution for every container-typed key (see
       :func:`resolve_container_aliases`)

    Args:
        user_config: User's YAML config as dict
        cluster_config: Cluster defaults from srtslurm.yaml (or None)

    Returns:
        Resolved config dict with all defaults applied
    """
    # Deep copy to avoid mutating original
    config = copy.deepcopy(user_config)

    # Normalize the 2.0 ``roles:`` authoring block into the existing internal
    # fields (resources.*_workers, backend.*_environment, backend.<engine>_config.*)
    # before anything else reads them. No-op for legacy recipes.
    from srtctl.core.roles import expand_roles

    expand_roles(config)

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

    # GPU-topology facts inherited from the cluster when the recipe omits them.
    # gpu_type and gpus_per_node describe the machine, not the deployment, so a
    # recipe can move between clusters by leaving them to srtslurm.yaml.
    resources_defaults = config.get("resources")
    if isinstance(resources_defaults, dict):
        if not resources_defaults.get("gpu_type") and cluster_config.get("default_gpu_type"):
            resources_defaults["gpu_type"] = cluster_config["default_gpu_type"]
            logger.debug("Applied default gpu_type: %s", resources_defaults["gpu_type"])
        if "gpus_per_node" not in resources_defaults and cluster_config.get("gpus_per_node") is not None:
            resources_defaults["gpus_per_node"] = cluster_config["gpus_per_node"]
            logger.debug("Applied cluster gpus_per_node: %s", resources_defaults["gpus_per_node"])

    default_sbatch_directives = cluster_config.get("default_sbatch_directives")
    if isinstance(default_sbatch_directives, dict):
        sbatch_directives = config.setdefault("sbatch_directives", {})
        for key, value in default_sbatch_directives.items():
            sbatch_directives.setdefault(key, value)
        logger.debug("Applied default sbatch_directives: %s", default_sbatch_directives)

    # Apply cluster-level het-job default. Without this, a recipe with
    # `het_jobs: None` would defer the cluster default at render-time but skip
    # SrtConfig validation (which only fires on `het_jobs is True`). Writing
    # the cluster value into the resolved recipe ensures __post_init__ catches
    # bad combinations (het + trtllm, het + agg, ...) at load time.
    resources = config.get("resources")
    if isinstance(resources, dict) and resources.get("het_jobs") is None:
        cluster_het = cluster_config.get("use_het_jobs")
        if cluster_het is not None:
            resources["het_jobs"] = bool(cluster_het)
            logger.debug("Applied cluster use_het_jobs default: %s", cluster_het)

    # Resolve model path alias
    model = config.get("model", {})
    model_path = model.get("path", "")

    model_paths = cluster_config.get("model_paths")
    if model_paths and model_path in model_paths:
        resolved_path = model_paths[model_path]
        model["path"] = resolved_path
        logger.debug(f"Resolved model alias '{model_path}' -> '{resolved_path}'")

    # Resolve every container alias in one pass (model.container,
    # frontend.container_image / nginx_container, benchmark.container_image,
    # exporter images, mooncake_kv_store.container, services, ...).
    containers = cluster_config.get("containers")
    if containers:
        for note in resolve_container_aliases(config, containers):
            logger.debug(note)

    # Apply reporting defaults (if not specified in user config)
    if "reporting" not in config and cluster_config.get("reporting"):
        config["reporting"] = cluster_config["reporting"]
        logger.debug("Applied cluster reporting config")

    if "health_check" not in config and cluster_config.get("default_health_check"):
        config["health_check"] = cluster_config["default_health_check"]
        logger.debug("Applied default_health_check: %s", config["health_check"])

    # Cluster-wide host setup (e.g. locking GPU clocks on nodes that need it).
    # Whole-block replace, like default_health_check: a recipe that sets
    # host_setup owns it entirely, so `host_setup: {commands: []}` is the way to
    # opt a single run out of the cluster default.
    if "host_setup" not in config and cluster_config.get("default_host_setup"):
        config["host_setup"] = cluster_config["default_host_setup"]
        logger.debug("Applied default_host_setup: %s", config["host_setup"])

    # Cluster-level default for nginx nofile ulimit (job yaml wins if present).
    frontend = config.get("frontend", {})
    if "nginx_raise_ulimit" not in frontend and cluster_config.get("nginx_raise_ulimit") is not None:
        frontend["nginx_raise_ulimit"] = cluster_config["nginx_raise_ulimit"]
        config["frontend"] = frontend
        logger.debug(f"Applied cluster nginx_raise_ulimit: {frontend['nginx_raise_ulimit']}")

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
    """Expand an override-format config into independent variants.

    Wraps :func:`_expand_override_variants` and carries a top-level ``schema``
    key (declared beside ``base``, not inside it) into every variant so each one
    loads at the version the file declares.
    """
    variants = _expand_override_variants(raw_config, selector=selector)
    if "schema" in raw_config:
        for _suffix, config in variants:
            config.setdefault("schema", raw_config["schema"])
    return variants


def _expand_override_variants(
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
            result_cm = base_cm
        else:
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

        # The file-level `schema` key lives beside `base`; each resolved variant
        # is a standalone recipe, so it declares the version itself.
        if "schema" in raw_plain and "schema" not in result_cm:
            result_cm.insert(0, "schema", raw_plain["schema"])

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
        except Exception as e:  # noqa: BLE001
            return [f"{path}: failed to expand overrides: {e}"]

        cluster_config = load_cluster_config()
        schema = SrtConfig.Schema()
        for suffix, config_dict in variants:
            resolved = resolve_config_with_defaults(config_dict, cluster_config)
            try:
                schema.load(resolved)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{path} [{suffix}]: {e}")
    elif "sweep" in raw:
        # Sweep format — expand every combination; the expander validates each one
        from .sweep import generate_sweep_configs

        try:
            expanded = generate_sweep_configs(raw)
        except Exception as e:  # noqa: BLE001
            return [f"{path}: failed to expand sweep: {e}"]
        if not expanded:
            errors.append(f"{path}: sweep expanded to zero jobs")
    else:
        # Plain config
        try:
            load_config(path)
        except Exception as e:  # noqa: BLE001
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


def git_clone_command_prefix() -> list[str]:
    """Return the ``git`` invocation every clone/fetch in srtctl should start from.

    Some clusters see intermittent git smart-HTTP failures negotiating HTTP/2
    against github.com (stalls, or truncated responses that git misreports as
    "could not read Username" auth-prompt failures) on certain network paths --
    observed on both a login host and its compute nodes. Setting
    ``git_http_version: "HTTP/1.1"`` in srtslurm.yaml works around this for
    every git clone/fetch srtctl performs, cluster-wide, without touching
    individual recipes or call sites.
    """
    version = get_srtslurm_setting("git_http_version")
    if not version:
        return ["git"]
    return ["git", "-c", f"http.version={version}"]


def _setdefault_nested(parent: dict, key: str, values: dict) -> None:
    """``parent[key]`` becomes a dict and gains ``values`` without clobbering."""
    child = parent.get(key)
    if not isinstance(child, dict):
        child = {}
        parent[key] = child
    for k, v in values.items():
        child.setdefault(k, v)


def expand_observability(cfg: dict) -> dict:
    """Expand ``observability.enabled`` into the individual launch flags.

    One knob, six effects -- see :class:`~srtctl.core.schema.ObservabilityConfig`
    for the rationale and the full list. Mutates ``cfg`` in place and returns it.

    Every write is a ``setdefault``: an explicit value in the recipe always
    wins. That makes it safe to flip ``observability.enabled`` on globally
    without silently overriding a recipe that deliberately disabled something.

    No-op unless ``observability.enabled`` is truthy.
    """
    from srtctl.core.schema import (
        ANALYTICS_ENGINE_CONFIG,
        ANALYTICS_REQUEST_TRACE_ENV,
        ANALYTICS_SPAN_ENV,
    )

    observability = cfg.get("observability")
    if not isinstance(observability, dict) or not observability.get("enabled"):
        return cfg

    # --- traces leg: SPAN_CLOSED lines on prefill, decode and frontend -------
    backend = cfg.get("backend")
    if not isinstance(backend, dict):
        backend = {}
        cfg["backend"] = backend

    for mode in ("prefill", "decode", "aggregated"):
        _setdefault_nested(backend, f"{mode}_environment", ANALYTICS_SPAN_ENV)

    frontend = cfg.get("frontend")
    if not isinstance(frontend, dict):
        frontend = {}
        cfg["frontend"] = frontend
    _setdefault_nested(frontend, "env", ANALYTICS_SPAN_ENV)

    # --- request-trace leg: per-request phase timings, frontend only ---------
    # Complements the span leg rather than duplicating it. Spans decompose the
    # router but stop at one opaque handle_payload per worker; these records
    # carry prefill_wait / prefill / kv_transfer_estimated for the same request,
    # keyed by x_request_id so all three legs join on one id.
    _setdefault_nested(frontend, "env", ANALYTICS_REQUEST_TRACE_ENV)

    # --- metrics leg: the /metrics surface and what appears on it ------------
    # publish_events_and_metrics is what creates the endpoint; without it the
    # engine-config keys below have nowhere to publish to.
    if backend.get("type", "sglang") == "trtllm":
        # An explicit False here defeats the whole metrics leg -- no /metrics
        # surface means no KV-cache gauges for anyone, including the in-job
        # scraper. setdefault still lets the recipe win (that contract matters),
        # but say so loudly: a recipe written before this knob existed will
        # otherwise silently produce a run with half the data missing.
        if backend.get("publish_events_and_metrics") is False:
            logger.warning(
                "observability.enabled but backend.publish_events_and_metrics is "
                "explicitly false — workers will NOT expose /metrics, so KV-cache "
                "gauges and worker scrapes will be missing. Remove that line or "
                "set it true to get the full analytics capture."
            )
        backend.setdefault("publish_events_and_metrics", True)

        trtllm_config = backend.get("trtllm_config")
        if not isinstance(trtllm_config, dict):
            trtllm_config = {}
            backend["trtllm_config"] = trtllm_config
        for mode in ("prefill", "decode", "aggregated"):
            if isinstance(trtllm_config.get(mode), dict):
                _setdefault_nested(trtllm_config, mode, ANALYTICS_ENGINE_CONFIG)

    logger.info(
        "observability.enabled: expanded span-event env (prefill/decode/frontend), "
        "publish_events_and_metrics and per-iteration engine stats"
    )
    return cfg


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
    if user_config is None:
        raise ValueError(f"Invalid config in {path}: YAML file is empty")
    if not isinstance(user_config, dict):
        raise TypeError(f"Invalid config in {path}: top-level YAML must be a mapping")

    # Strip lock: section if present (lockfiles are valid recipes)
    # Preserved for comparison after the new run completes
    lock_data = user_config.pop("lock", None)
    if lock_data:
        if verify_lock_integrity(lock_data):
            logger.info("Loaded lockfile — integrity verified, will compare after benchmark")
        else:
            logger.warning("Loaded lockfile — integrity check FAILED (lock section may have been edited)")
            logger.warning("Comparison results may not reflect the original run")

    # Load cluster defaults (optional)
    cluster_config = load_cluster_config()

    # Resolve with defaults (applies aliases and default values)
    resolved_config = resolve_config_with_defaults(user_config, cluster_config)

    # Expand the single `observability.enabled` knob into the individual
    # launch flags. Done on the raw dict (before schema.load) so every
    # downstream consumer -- worker command builder, engine YAML writer,
    # frontend env -- sees the expanded values with no extra plumbing.
    expand_observability(resolved_config)

    # Parse with marshmallow schema to get typed SrtConfig
    try:
        schema = SrtConfig.Schema()
        config = schema.load(resolved_config)
        assert isinstance(config, SrtConfig)
        logger.info(f"Loaded config: {config.name}")
        # Attach lock data for post-run comparison. Uses object.__setattr__
        # because SrtConfig is frozen — this is the standard Python pattern for
        # adding metadata to frozen dataclasses without modifying the schema.
        # Retrieved via getattr(config, "_lock_data", None) in postprocess.
        object.__setattr__(config, "_lock_data", lock_data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid config in {path}: {e}") from e
