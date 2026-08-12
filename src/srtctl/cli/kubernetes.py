# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI helpers for Kubernetes rendering and deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from srtctl.core.config import (
    generate_override_configs,
    load_cluster_config,
    load_config,
    resolve_config_with_defaults,
)
from srtctl.core.schema import SrtConfig
from srtctl.kubernetes import apply_kubernetes, delete_kubernetes, render_kubernetes_manifests


def load_kubernetes_configs(config_path: Path, selector: str | None = None) -> list[SrtConfig]:
    """Load one recipe or its selected override variants."""
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: top-level YAML must be a mapping")
    if "base" not in raw:
        if selector is not None:
            raise ValueError("a variant selector requires an override-format YAML")
        return [load_config(config_path)]

    cluster_config = load_cluster_config()
    schema = SrtConfig.Schema()
    configs: list[SrtConfig] = []
    for _suffix, values in generate_override_configs(raw, selector=selector):
        resolved = resolve_config_with_defaults(values, cluster_config)
        config = schema.load(resolved)
        if not isinstance(config, SrtConfig):
            raise TypeError("SrtConfig schema returned an unexpected value")
        configs.append(config)
    return configs


def dump_configs(configs: list[SrtConfig]) -> str:
    manifests = [manifest for config in configs for manifest in render_kubernetes_manifests(config)]
    return yaml.safe_dump_all(manifests, sort_keys=False, explicit_start=True)


def run_kubernetes_command(args: Any, *, config_path: Path, selector: str | None) -> None:
    configs = load_kubernetes_configs(config_path, selector=selector)
    action = args.k8s_command
    executable = getattr(args, "kubectl", "kubectl")
    if action == "generate":
        rendered = dump_configs(configs)
        output = getattr(args, "manifest_output", None)
        if output is None:
            print(rendered, end="")
        else:
            output.write_text(rendered)
            print(f"Wrote: {output}")
        return
    if action == "apply":
        for config in configs:
            output = apply_kubernetes(
                config,
                executable=executable,
                wait=not getattr(args, "no_wait", False),
                timeout_seconds=getattr(args, "timeout", None),
            )
            if output:
                print(output, end="" if output.endswith("\n") else "\n")
        return
    if action == "delete":
        for config in configs:
            output = delete_kubernetes(config, executable=executable)
            if output:
                print(output, end="" if output.endswith("\n") else "\n")
        return
    raise ValueError(f"unknown Kubernetes command: {action}")
