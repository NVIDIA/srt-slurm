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
from srtctl.kubernetes import (
    apply_kubernetes,
    delete_kubernetes,
    get_kubernetes_status,
    render_kubernetes_manifests,
    run_kubernetes,
    stream_kubernetes_logs,
)


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
    if action == "run":
        output_root = getattr(args, "output_dir", None)
        for config in configs:
            output_dir = output_root
            if output_dir is not None and len(configs) > 1:
                output_dir = output_dir / (config.kubernetes.name or config.name)
            result = run_kubernetes(
                config,
                executable=executable,
                readiness_timeout_seconds=getattr(args, "timeout", None),
                benchmark_timeout_seconds=getattr(args, "benchmark_timeout", None),
                output_dir=output_dir,
                keep_resources=getattr(args, "keep_resources", False),
                stream_logs=not getattr(args, "no_follow", False),
                on_update=print,
            )
            print(yaml.safe_dump(result, sort_keys=False), end="")
        return
    if action == "status":
        statuses = [get_kubernetes_status(config, executable=executable) for config in configs]
        value: Any = statuses[0] if len(statuses) == 1 else statuses
        print(yaml.safe_dump(value, sort_keys=False), end="")
        return
    if action == "logs":
        if len(configs) != 1:
            raise ValueError("k8s logs requires exactly one selected configuration")
        return_code = stream_kubernetes_logs(
            configs[0],
            executable=executable,
            follow=getattr(args, "follow", False),
            component=getattr(args, "component", None),
            tail=getattr(args, "tail", 200),
        )
        if return_code != 0:
            raise RuntimeError(f"kubectl logs exited with status {return_code}")
        return
    if action == "delete":
        for config in configs:
            output = delete_kubernetes(config, executable=executable)
            if output:
                print(output, end="" if output.endswith("\n") else "\n")
        return
    raise ValueError(f"unknown Kubernetes command: {action}")
