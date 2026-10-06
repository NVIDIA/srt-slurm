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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml.comments import CommentedMap

from .lockfile import verify_lock_integrity
from .roles import ROLE_NAMES
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
        "env",
        "args",
        "extra_args",
        "container_mounts",
        "sbatch_directives",
        "srun_options",
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


# Recipe keys that only the pre-2.0 (v1) layout had. The schema no longer has
# fields of these names; a recipe that spells them is a v1 recipe and is rejected
# at load with a pointer to ``srtctl migrate``, which rewrites every one of them
# (docs/legacy-v1.md has the key-by-key mapping), instead of marshmallow's
# bare "Unknown field".
LEGACY_TOP_LEVEL_KEYS: tuple[str, ...] = ("backend", "infra")
LEGACY_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "resources": tuple(
        key
        for role in ROLE_NAMES
        for key in (f"{role}_nodes", f"{role}_workers", f"gpus_per_{role}", f"{role}_critical")
    ),
    "frontend": ("orchestrator_placement", "dedicated_node"),
    "benchmark": ("client_placement", "client_dedicated_node"),
    "dynamo": ("version", "hash", "wheel", "cargo_patches"),
}
MIGRATE_HINT = "run `srtctl migrate -f <recipe> --in-place` to rewrite it (docs/legacy-v1.md maps every key)"


def legacy_keys_present(config: Mapping[str, Any]) -> list[str]:
    """Dotted paths of every v1-only key a raw recipe still carries."""
    found = [key for key in LEGACY_TOP_LEVEL_KEYS if key in config]
    for section, keys in LEGACY_SECTION_KEYS.items():
        block = config.get(section)
        if isinstance(block, Mapping):
            found.extend(f"{section}.{key}" for key in keys if key in block)
    return found


def require_current_schema(config: Mapping[str, Any]) -> None:
    """Reject a v1 recipe before any expansion runs.

    A recipe must declare ``schema: 2``: the key was introduced with the 2.0
    layout, so its absence marks a pre-2.0 recipe. Any key that only the v1
    layout had is rejected too, even under ``schema: 2``, so the internal
    fields cannot be reached from a recipe by their old spelling.
    """
    from srtctl.core.schema import CURRENT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

    version = config.get("schema")
    if version is None:
        raise ValueError(
            "recipe has no `schema:` key, which marks the pre-2.0 layout; schema 1 recipes no longer load. "
            f"Declare `schema: {CURRENT_SCHEMA_VERSION}` or {MIGRATE_HINT}"
        )
    if isinstance(version, bool) or not isinstance(version, int) or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema {version!r} is not supported; this srtctl loads schema {CURRENT_SCHEMA_VERSION} only. "
            f"For an older recipe, {MIGRATE_HINT}"
        )
    legacy = legacy_keys_present(config)
    if legacy:
        raise ValueError(
            f"recipe uses the pre-2.0 (v1) layout: {', '.join(legacy)}. These keys no longer load; {MIGRATE_HINT}"
        )


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

    Raises:
        ValueError: The recipe is pre-2.0 (no ``schema: 2`` or a v1-only key);
            see :func:`require_current_schema`.
    """
    require_current_schema(user_config)

    # Deep copy to avoid mutating original
    config = copy.deepcopy(user_config)

    # A declared mooncake-master service sets ``engine.mooncake_kv_store`` before
    # anything else looks at the engine.
    from srtctl.services.normalize import expand_services

    expand_services(config)

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

    # Resolve the cluster GPU exporter once, before container aliases. Keep
    # recipe overrides and the existing default_exporters opt-out authoritative.
    if "default_gpu_exporter" in cluster_config:
        tachometer = config.setdefault("observability", {}).setdefault("tachometer", {})
        tachometer.setdefault("default_gpu_exporter", copy.deepcopy(cluster_config["default_gpu_exporter"]))

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
            try:
                schema.load(resolve_config_with_defaults(config_dict, cluster_config))
            except Exception as e:  # noqa: BLE001 - a v1 layout or a bad value is a finding, not a crash
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


def _present_roles(cfg: dict) -> dict[str, dict]:
    """The ``roles:`` entries of a recipe dict that are mappings, by role name."""
    roles = cfg.get("roles")
    if not isinstance(roles, dict):
        return {}
    return {role: spec for role, spec in roles.items() if isinstance(spec, dict)}


def _engine_mapping(cfg: dict, role: str | None = None) -> dict | None:
    """The engine mapping ``role`` runs (``roles.<role>.engine``, else the top-level ``engine``), as a dict.

    A string shorthand (``engine: trtllm``) reads as ``{type: trtllm}``. None when no engine
    is declared (the default engine) or when the value is not a mapping (left for schema
    validation).
    """
    spec = _present_roles(cfg).get(role) if role is not None else None
    engine = spec.get("engine") if spec is not None and spec.get("engine") is not None else cfg.get("engine")
    if engine is None:
        return None
    if isinstance(engine, str):
        return {"type": engine}
    return engine if isinstance(engine, dict) else None


def _engine_type(engine: dict | None) -> str:
    return str(engine.get("type", "sglang")) if engine is not None else "sglang"


def _setdefault_trtllm_role_args(
    cfg: dict,
    defaults: dict,
    skip_section: Callable[[dict], bool] | None = None,
) -> dict[str, dict]:
    """``setdefault`` ``defaults`` into ``roles.<role>.args`` of every role that runs TRT-LLM.

    ``args`` is created when absent or null, so a role with no engine arguments gets the
    defaults too; a value that is not a mapping is left alone so schema validation reports
    it. ``skip_section`` lets a caller leave a role's args untouched based on their
    contents. Explicit recipe values are never clobbered. Returns the args touched, keyed
    by role, so callers can report explicit opt-outs.
    """
    touched: dict[str, dict] = {}
    for role, spec in _present_roles(cfg).items():
        if _engine_type(_engine_mapping(cfg, role)) != "trtllm":
            continue
        args = spec.get("args")
        if args is None:
            args = spec["args"] = {}
        elif not isinstance(args, dict):
            continue
        if skip_section is not None and skip_section(args):
            continue
        for key, value in defaults.items():
            args.setdefault(key, value)
        touched[role] = args
    return touched


def expand_observability(cfg: dict) -> dict:
    """Expand ``observability.enabled`` into the individual launch flags.

    See :class:`~srtctl.core.schema.ObservabilityConfig` for the capture settings.
    Mutates ``cfg`` in place and returns it.

    Defaults preserve explicit recipe values. Observability leaves publication
    settings unchanged; the legacy combined flag requires an explicit true.

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

    # --- traces leg: SPAN_CLOSED lines on every worker role and the frontend ----
    for spec in _present_roles(cfg).values():
        _setdefault_nested(spec, "env", ANALYTICS_SPAN_ENV)

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

    # --- metrics leg: engine metrics on the worker /metrics surface ----------
    # Publication uses the engine's metrics-only default. The legacy combined
    # flag is enabled only by an explicit recipe setting, not observability.
    # Every TRT-LLM role's args get the iteration-level gauges the capture reads;
    # a role with no args at all gets them too. expand_trtllm_engine_defaults runs
    # after this and must find True already in place.
    sections = _setdefault_trtllm_role_args(cfg, ANALYTICS_ENGINE_CONFIG)
    opted_out = [
        f"roles.{role}.args.{key}"
        for role, args in sections.items()
        for key in ANALYTICS_ENGINE_CONFIG
        if args.get(key) is False
    ]
    if opted_out:
        # Also reached by a saved or locked recipe: the load step bakes the
        # resolved engine keys in, so a later observability.enabled: true
        # meets an explicit false rather than an omission.
        logger.warning(
            "observability.enabled but %s is false — the iteration-level "
            "trtllm_kv_cache_* gauges (enable_iter_perf_stats) and per-request histograms "
            "(return_perf_metrics) need true; remove the explicit false to get them back.",
            ", ".join(opted_out),
        )

    logger.info(
        "observability.enabled: expanded span-event env (every role and the frontend), and per-iteration engine stats"
    )
    return cfg


def expand_trtllm_serve_defaults(cfg: dict) -> dict:
    """Bake the trtllm-serve worker-metrics default into the TRT-LLM roles' args.

    trtllm-serve registers a worker's Prometheus route (``/prometheus/metrics``)
    only when the engine runs with ``return_perf_metrics: true`` (TensorRT-LLM
    ``serve/openai_server.py``, ``register_routes``); TensorRT-LLM's own default
    is ``false``. Tachometer scrapes that route on every run, so without this
    default every trtllm-serve worker endpoint answers HTTP 404 and the capture
    silently has no worker-level data.

    Applies to every ``frontend.type: trtllm_serve`` recipe, to each role that
    runs TRT-LLM, independent of ``observability.enabled``. ``roles.<role>.args``
    is created when absent, so a role with no engine arguments gets the default
    too. Every write is a ``setdefault``: an explicit ``return_perf_metrics:
    false`` in the recipe wins, but is reported loudly. Mutates ``cfg`` in place
    and returns it.
    """
    from srtctl.core.schema import TRTLLM_SERVE_ENGINE_DEFAULTS

    frontend = cfg.get("frontend")
    if not isinstance(frontend, dict) or frontend.get("type") != "trtllm_serve":
        return cfg

    sections = _setdefault_trtllm_role_args(cfg, TRTLLM_SERVE_ENGINE_DEFAULTS)
    opted_out = [f"roles.{role}" for role, args in sections.items() if args.get("return_perf_metrics") is False]
    if opted_out:
        logger.warning(
            "frontend.type: trtllm_serve with return_perf_metrics: false on %s — those "
            "trtllm-serve workers will NOT mount /prometheus/metrics (HTTP 404), so the "
            "Tachometer backend_* endpoints and the per-request Prometheus histograms "
            "will be empty for them. Remove the line to keep the default.",
            ", ".join(opted_out),
        )
    return cfg


def expand_trtllm_engine_defaults(cfg: dict) -> dict:
    """Keep TensorRT-LLM's per-iteration statistics off unless a recipe asks for them.

    Applies ``TRTLLM_ENGINE_DEFAULTS`` (``enable_iter_perf_stats: false``) to the
    args of every role that runs TRT-LLM, under both the ``dynamo`` and the
    ``trtllm_serve`` frontend and independent of ``observability.enabled``.

    Why an explicit ``false`` when TensorRT-LLM's own default is already
    ``false``: ``dynamo.trtllm`` builds the engine arguments with
    ``enable_iter_perf_stats`` derived from ``--publish-metrics``
    (``components/src/dynamo/trtllm/workers/llm_worker.py``), and
    ``engine.publish_metrics`` passes that flag by default, so every Dynamo
    worker would otherwise collect KV-cache statistics and CUDA-event step timing
    on every executor loop. The engine YAML is merged over those derived
    arguments and wins on conflicts (TensorRT-LLM
    ``update_llm_args_with_extra_dict``), so the explicit key is what turns the
    statistics off. The request-level ``trtllm_*`` Prometheus series (request
    latency, TTFT, TPOT, queue / prefill / decode time, token counters) do not
    depend on it: they come from the per-request perf metrics, which
    ``--publish-metrics`` sets on the Dynamo path and ``return_perf_metrics: true``
    sets for trtllm-serve. What the default drops is the iteration-level
    ``trtllm_*`` gauges (``trtllm_kv_cache_*``, running / waiting requests,
    iteration latency) and, on Dynamo, the ``dynamo_component_kvstats_*`` gauges,
    the router worker-load sample and the Planner's forward-pass metrics. No
    benchmark client reads them. Captured iteration-level gauges can be absent
    or remain at their seeded values on a default run.
    Roles whose args select the legacy ``tensorrt`` backend are skipped: its
    ``LlmArgs`` rejects the key on containers before the backend's removal and
    always collected the statistics anyway.

    Every write is a ``setdefault``: an explicit ``enable_iter_perf_stats: true``
    in the recipe wins, and so does :func:`expand_observability`, which runs
    first and needs the iteration-level gauges for its capture. ``args`` is
    created for a role that has none. Mutates ``cfg`` in place and returns it.
    """
    from srtctl.core.schema import TRTLLM_ENGINE_DEFAULTS

    _setdefault_trtllm_role_args(
        cfg,
        TRTLLM_ENGINE_DEFAULTS,
        skip_section=lambda args: str(args.get("backend", "pytorch")).lower() in ("tensorrt", "trt"),
    )
    return cfg


def expand_engine_config_defaults(resolved_config: dict) -> dict:
    """Run the engine-config expansions that turn a resolved recipe into what the job runs.

    Order matters: :func:`expand_observability` first, so its ``True`` for the
    iteration statistics is in place before :func:`expand_trtllm_engine_defaults`
    setdefaults ``False``. Kept out of :func:`resolve_config_with_defaults` so
    tools that only inspect a recipe (validation, the MCP spec tools) keep
    seeing the recipe's own keys; every entry point that builds the ``SrtConfig``
    a job runs under, or shows in ``srtctl dry-run``, calls this so the two
    agree. Each expansion looks at every role's engine, so per-role engines are
    covered. Mutates and returns ``resolved_config``.
    """
    expand_observability(resolved_config)
    expand_trtllm_serve_defaults(resolved_config)
    expand_trtllm_engine_defaults(resolved_config)
    return resolved_config


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
    # launch flags and bake in the TRT-LLM engine-config defaults. Done on the
    # raw dict (before schema.load) so every downstream consumer -- worker
    # command builder, engine YAML writer, frontend env -- sees the expanded
    # values with no extra plumbing.
    expand_engine_config_defaults(resolved_config)

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
