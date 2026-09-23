#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter sweep generation for YAML configs.

This module generates multiple job configs from a sweep configuration by
expanding all combinations of sweep parameters.
"""

import copy
import itertools
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _parallel_size(config: dict[str, Any], name: str) -> int:
    """Return a positive parallel size from kebab- or snake-case config."""
    value = config.get(name, config.get(name.replace("-", "_"), 1))
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vLLM {name} must be a positive integer; got {value!r}") from exc
    if size < 1:
        raise ValueError(f"vLLM {name} must be a positive integer; got {value!r}")
    return size


def _vllm_parallelism_fits_allocation(config: dict[str, Any]) -> bool:
    """Return whether a vLLM variant fits every active worker allocation.

    TP-only vLLM workers may intentionally use fewer GPUs than a reserved node.
    Distributed DP workers, however, require their complete DP*TP*PP*PCP world
    to match the endpoint allocation exactly.
    """
    backend = config.get("backend", {})
    if backend.get("type") != "vllm":
        return True

    from srtctl.core.schema import ResourceConfig

    resources = ResourceConfig.Schema().load(config.get("resources", {}))
    vllm_config = backend.get("vllm_config") or {}
    frontend_type = (config.get("frontend") or {}).get("type", "dynamo")
    modes = (
        ("prefill", resources.num_prefill, resources.gpus_per_prefill),
        ("decode", resources.num_decode, resources.gpus_per_decode),
        ("aggregated", resources.num_agg, resources.gpus_per_agg),
    )

    for mode, worker_count, allocated_gpus in modes:
        if worker_count < 1:
            continue
        mode_config = vllm_config.get(mode) or {}
        dp_size = _parallel_size(mode_config, "data-parallel-size")
        required_gpus = (
            dp_size
            * _parallel_size(mode_config, "tensor-parallel-size")
            * _parallel_size(mode_config, "pipeline-parallel-size")
            * _parallel_size(mode_config, "prefill-context-parallel-size")
        )
        if required_gpus > allocated_gpus:
            return False
        if frontend_type != "vllm" and dp_size > 1 and required_gpus != allocated_gpus:
            return False

    return True


def expand_template(template: Any, values: dict[str, Any]) -> Any:
    """Recursively expand template strings with values.

    Args:
        template: Template object (dict, list, str, or other)
        values: Dictionary of parameter values to substitute

    Returns:
        Expanded template with {param} placeholders replaced
    """
    if isinstance(template, dict):
        return {k: expand_template(v, values) for k, v in template.items()}
    elif isinstance(template, list):
        return [expand_template(item, values) for item in template]
    elif isinstance(template, str):
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            # Handle list values specially - convert to comma-separated string or keep as list
            if isinstance(value, list):
                # For YAML lists, we want to keep them as lists, not convert to string
                if placeholder in result and result == placeholder:
                    # If the entire string is just the placeholder, replace with the list
                    return value
                else:
                    # If it's embedded in a string, convert to comma-separated
                    result = result.replace(placeholder, ",".join(str(v) for v in value))
            else:
                result = result.replace(placeholder, str(value))
        return result
    else:
        return template


def generate_sweep_configs(sweep_config: dict) -> list[tuple[dict, dict]]:
    """Generate all job configs from a sweep configuration.

    Args:
        sweep_config: Config dict with 'sweep' section defining parameters

    Returns:
        List of (expanded_config, param_values) tuples
    """
    if "sweep" not in sweep_config:
        raise ValueError("Sweep config must have 'sweep' section")

    # Apply cluster defaults before sweep expansion
    from srtctl.core.config import load_cluster_config, resolve_config_with_defaults

    cluster_config = load_cluster_config()
    sweep_config = resolve_config_with_defaults(sweep_config, cluster_config)

    # Extract sweep parameters
    sweep_params = sweep_config["sweep"]

    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values_list = [sweep_params[name] for name in param_names]

    configs = []
    filtered_count = 0
    for values in itertools.product(*param_values_list):
        # Create parameter dict for this combination
        params = dict(zip(param_names, values, strict=False))

        # Create a copy of the config without the sweep section
        config = copy.deepcopy(sweep_config)
        del config["sweep"]

        # Expand all template placeholders
        config = expand_template(config, params)

        # A Cartesian vLLM topology sweep can request a larger parallel world
        # than the GPUs allocated to a worker. Drop those variants before full
        # schema validation so one impossible combination does not abort the
        # entire sweep (notably for vllm-router, which validates DP eagerly).
        if not _vllm_parallelism_fits_allocation(config):
            filtered_count += 1
            continue

        # Generate a unique name for this config
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        config["name"] = f"{sweep_config['name']}_{param_str}"

        # Validate and serialize back to dict
        from srtctl.core.schema import SrtConfig

        schema = SrtConfig.Schema()
        validated = schema.load(config)
        config = schema.dump(validated)

        configs.append((config, params))

    if filtered_count:
        logger.warning(
            "Filtered %d vLLM sweep combination(s) incompatible with worker GPU allocations",
            filtered_count,
        )
    if filtered_count and not configs:
        raise ValueError("Sweep has no runnable configurations after filtering vLLM parallelism")

    return configs
