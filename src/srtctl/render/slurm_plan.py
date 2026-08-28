# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile a portable Slurm serving lifecycle from a resolved recipe."""

from __future__ import annotations

import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from srtctl.backends.sglang import SGLangProtocol
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import get_srtslurm_setting
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import SrunSpec
from srtctl.frontends.dynamo import DynamoFrontend
from srtctl.ports import (
    ETCD_CLIENT_PORT,
    FRONTEND_PUBLIC_PORT,
    MOONCAKE_HTTP_METADATA_PORT,
    MOONCAKE_MASTER_PORT,
    MOONCAKE_METRICS_PORT,
    NATS_PORT,
)

_SECRET_NAME = re.compile(r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)(?:_|$)")
_PLACEHOLDER = re.compile(r"__SRTCTL_(?:NODE|IP)_\d+__|__SRTCTL_(?:JOB_ID|OUTPUT_ROOT|SOURCE_DIR)__")


@dataclass(frozen=True)
class LaunchStep:
    """One background service plus readiness gates that follow its launch."""

    label: str
    stage: str
    spec: SrunSpec
    waits: tuple[tuple[str, int, int, str], ...] = ()


def _is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name.upper()))


def _allocation_node_count(config: SrtConfig) -> int:
    count = config.total_nodes
    dedicated_roles = sum(
        (
            config.infra.etcd_nats_dedicated_node,
            config.frontend.dedicated_node,
            config.benchmark.client_dedicated_node,
        )
    )
    if dedicated_roles:
        count += 1 if config.benchmark.colocate_with_frontend else dedicated_roles
    return count


def _symbolic_topology(config: SrtConfig) -> tuple[Nodes, dict[str, str], int]:
    required_nodes = _allocation_node_count(config)
    nodelist = [f"__SRTCTL_NODE_{index}__" for index in range(required_nodes)]
    nodes = Nodes.from_nodelist(
        nodelist,
        frontend_dedicated_node=config.frontend.dedicated_node,
        client_dedicated_node=config.benchmark.client_dedicated_node,
        etcd_nats_dedicated_node=config.infra.etcd_nats_dedicated_node,
        colocate_dedicated_nodes=config.benchmark.colocate_with_frontend,
    )
    ips = {node: f"__SRTCTL_IP_{index}__" for index, node in enumerate(nodelist)}
    return nodes, ips, required_nodes


def _redact_command(argv: list[str], spec: SrunSpec) -> tuple[list[str], set[str]]:
    """Remove secret values while preserving the environment contract."""
    redacted = list(argv)
    required: set[str] = set()

    if spec.srun_export_env:
        for index, arg in enumerate(redacted):
            if not arg.startswith("--export=ALL,"):
                continue
            exports: list[str] = []
            for item in arg.removeprefix("--export=ALL,").split(","):
                name, separator, _value = item.partition("=")
                if separator and _is_secret_name(name):
                    exports.append(name)
                    required.add(name)
                else:
                    exports.append(item)
            redacted[index] = "--export=ALL," + ",".join(exports)

    if spec.env_to_set:
        bash_index = next(
            (index for index in range(len(redacted) - 1) if redacted[index : index + 2] == ["bash", "-c"]),
            None,
        )
        if bash_index is not None:
            bash_command = redacted[bash_index + 2]
            for name, value in spec.env_to_set.items():
                if not _is_secret_name(name):
                    continue
                bash_command = bash_command.replace(
                    f"export {name}={shlex.quote(value)}",
                    f'export {name}="${{{name}:?Set {name} before running this script}}"',
                )
                required.add(name)
            redacted[bash_index + 2] = bash_command

    return redacted, required


def _replace_static_paths(value: str, *, source_dir: Path) -> str:
    source = str(source_dir.resolve())
    return value.replace(source, "__SRTCTL_SOURCE_DIR__").replace("/__SRTCTL_OUTPUT_ROOT__", "__SRTCTL_OUTPUT_ROOT__")


def _shell_word(value: str, *, source_dir: Path) -> str:
    """Quote one argv item while allowing only generated placeholders to expand."""
    value = _replace_static_paths(value, source_dir=source_dir)
    pieces: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER.finditer(value):
        if match.start() > cursor:
            pieces.append(shlex.quote(value[cursor : match.start()]))
        token = match.group(0)
        if token == "__SRTCTL_JOB_ID__":
            expression = "${SLURM_JOB_ID}"
        elif token == "__SRTCTL_OUTPUT_ROOT__":
            expression = "${SRTCTL_OUTPUT_ROOT}"
        elif token == "__SRTCTL_SOURCE_DIR__":
            expression = "${SRTCTL_SOURCE_DIR}"
        elif token.startswith("__SRTCTL_NODE_"):
            index = token.removeprefix("__SRTCTL_NODE_").removesuffix("__")
            expression = f"${{SRTCTL_NODES[{index}]}}"
        else:
            index = token.removeprefix("__SRTCTL_IP_").removesuffix("__")
            expression = f"${{SRTCTL_NODE_IPS[{index}]}}"
        pieces.append(f'"{expression}"')
        cursor = match.end()
    if cursor < len(value):
        pieces.append(shlex.quote(value[cursor:]))
    return "".join(pieces) or "''"


def _render_command(argv: list[str], *, source_dir: Path) -> str:
    return " \\\n  ".join(_shell_word(arg, source_dir=source_dir) for arg in argv)


def _validate_config(config: SrtConfig) -> None:
    if config.resources.het_jobs:
        raise ValueError("render-launch currently supports homogeneous Slurm allocations only")
    if not isinstance(config.backend, SGLangProtocol):
        raise NotImplementedError("render-launch currently supports backend.type: sglang only")
    if config.frontend.type != "dynamo":
        raise NotImplementedError("render-launch currently supports frontend.type: dynamo only")
    if config.frontend.enable_multiple_frontends:
        raise ValueError("render-launch currently requires frontend.enable_multiple_frontends: false")
    if config.backend.get_srun_config().launch_per_endpoint:
        raise ValueError("render-launch does not yet support MPI-style endpoint launches")
    if config.model.stage_dir:
        raise ValueError("render-launch does not yet support model.stage_dir")


def _build_steps(
    config: SrtConfig,
    *,
    source_dir: Path,
    output_base: Path,
) -> tuple[list[LaunchStep], int, str, str]:
    nodes, node_ips, required_nodes = _symbolic_topology(config)

    def resolve_node_ip(node: str, _network_interface: str | None = None) -> str:
        return node_ips[node]

    runtime = RuntimeContext.from_config(
        config,
        "__SRTCTL_JOB_ID__",
        log_dir_base=Path("/__SRTCTL_OUTPUT_ROOT__"),
        nodes=nodes,
        node_ip_resolver=lambda node: resolve_node_ip(node),
        validate_paths=False,
        create_log_dir=False,
        prefer_output_env=False,
    )
    orchestrator = SweepOrchestrator(config, runtime)
    steps: list[LaunchStep] = []

    infra = orchestrator.build_head_infrastructure_srun()
    steps.append(
        LaunchStep(
            label="infra",
            stage="infrastructure",
            spec=infra,
            waits=(
                (runtime.nodes.infra, NATS_PORT, 300, "NATS"),
                (runtime.nodes.infra, ETCD_CLIENT_PORT, 300, "etcd"),
            ),
        )
    )

    mooncake = orchestrator.build_mooncake_master_srun()
    if mooncake is not None:
        steps.append(
            LaunchStep(
                label="mooncake-master",
                stage="infrastructure",
                spec=mooncake,
                waits=(
                    (runtime.nodes.infra, MOONCAKE_MASTER_PORT, 120, "Mooncake RPC"),
                    (runtime.nodes.infra, MOONCAKE_HTTP_METADATA_PORT, 120, "Mooncake metadata"),
                    (runtime.nodes.infra, MOONCAKE_METRICS_PORT, 120, "Mooncake metrics"),
                ),
            )
        )

    grouped: dict[tuple[str, int], list] = defaultdict(list)
    for process in orchestrator.backend_processes:
        grouped[(process.endpoint_mode, process.endpoint_index)].append(process)
    for process in orchestrator.backend_processes:
        endpoint_processes = grouped[(process.endpoint_mode, process.endpoint_index)]
        spec = orchestrator.build_worker_srun(
            process,
            endpoint_processes,
            resolve_node_ip=resolve_node_ip,
            include_fingerprint=False,
        )
        steps.append(
            LaunchStep(
                label=f"{process.endpoint_mode}-{process.endpoint_index}-rank-{process.node_rank}",
                stage="workers",
                spec=spec,
            )
        )

    topology = orchestrator._compute_frontend_topology()
    if topology.uses_nginx:
        raise ValueError("render-launch does not yet support nginx/multiple frontends")
    frontend = DynamoFrontend()
    frontend_specs = frontend.build_frontend_sruns(topology=topology, runtime=runtime, config=config)
    for index, spec in enumerate(frontend_specs):
        frontend_node = (spec.nodelist or (runtime.nodes.head,))[0]
        steps.append(
            LaunchStep(
                label=f"frontend-{index}",
                stage="frontend",
                spec=spec,
                waits=((frontend_node, FRONTEND_PUBLIC_PORT, 300, "Dynamo frontend"),),
            )
        )

    frontend_node = topology.frontend_nodes[0]
    return steps, required_nodes, node_ips[frontend_node], str(output_base.resolve())


def render_slurm_launch_script(config: SrtConfig, *, source_dir: Path, output_base: Path) -> str:
    """Render a standalone serving launcher for a future Slurm allocation."""
    _validate_config(config)
    steps, required_nodes, frontend_ip, output_default = _build_steps(
        config,
        source_dir=source_dir,
        output_base=output_base,
    )

    rendered_steps: list[tuple[LaunchStep, str]] = []
    required_secrets: set[str] = set()
    for step in steps:
        argv, secrets = _redact_command(step.spec.argv(job_id=None), step.spec)
        required_secrets.update(secrets)
        rendered_steps.append((step, _render_command(argv, source_dir=source_dir)))

    network_interface = str(get_srtslurm_setting("network_interface", "eth0"))
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by `srtctl render-launch`; allocation identities are resolved when this script runs.",
        "set -Eeuo pipefail",
        "",
        ': "${SLURM_JOB_ID:?Run this script inside an active Slurm allocation}"',
        'SRTCTL_RAW_NODELIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"',
        ': "${SRTCTL_RAW_NODELIST:?SLURM_JOB_NODELIST or SLURM_NODELIST must be set}"',
        f"SRTCTL_REQUIRED_NODES={required_nodes}",
        'SRTCTL_SOURCE_DIR="${SRTCTL_SOURCE_DIR:-}"',
        f'if [[ -z "${{SRTCTL_SOURCE_DIR}}" ]]; then SRTCTL_SOURCE_DIR={shlex.quote(str(source_dir.resolve()))}; fi',
        'SRTCTL_OUTPUT_ROOT="${SRTCTL_OUTPUT_ROOT:-}"',
        f'if [[ -z "${{SRTCTL_OUTPUT_ROOT}}" ]]; then SRTCTL_OUTPUT_ROOT={shlex.quote(output_default)}; fi',
        'SRTCTL_NETWORK_INTERFACE="${SRTCTL_NETWORK_INTERFACE:-}"',
        f'if [[ -z "${{SRTCTL_NETWORK_INTERFACE}}" ]]; then SRTCTL_NETWORK_INTERFACE={shlex.quote(network_interface)}; fi',
        'SRTCTL_LOG_DIR="${SRTCTL_OUTPUT_ROOT}/${SLURM_JOB_ID}/logs"',
        'mkdir -p "${SRTCTL_LOG_DIR}"',
        "",
    ]
    for name in sorted(required_secrets):
        lines.append(f': "${{{name}:?Set {name} before running this script}}"')
    if required_secrets:
        lines.append("")

    lines.extend(
        [
            "declare -a SRTCTL_NODES=()",
            "while IFS= read -r node; do",
            '  [[ -n "${node}" ]] || continue',
            '  if (( ${#SRTCTL_NODES[@]} < SRTCTL_REQUIRED_NODES )); then SRTCTL_NODES+=("${node}"); fi',
            'done < <(scontrol show hostnames "${SRTCTL_RAW_NODELIST}")',
            "if (( ${#SRTCTL_NODES[@]} < SRTCTL_REQUIRED_NODES )); then",
            '  echo "Need ${SRTCTL_REQUIRED_NODES} nodes; allocation has ${#SRTCTL_NODES[@]}" >&2',
            "  exit 2",
            "fi",
            "declare -a SRTCTL_NODE_IPS=()",
            'for node in "${SRTCTL_NODES[@]}"; do',
            '  ip=$(srun --overlap --nodes 1 --ntasks 1 --nodelist "${node}" bash -c \'',
            '    interface="$1"',
            '    if command -v ip >/dev/null 2>&1 && ip -o -4 addr show dev "${interface}" >/dev/null 2>&1; then',
            '      ip -o -4 addr show dev "${interface}" | while read -r _ _ _ cidr _; do printf "%s\\n" "${cidr%/*}"; break; done',
            "    else",
            '      read -r first _ <<< "$(hostname -I)"; printf "%s\\n" "${first}"',
            "    fi",
            '  \' _ "${SRTCTL_NETWORK_INTERFACE}")',
            '  [[ -n "${ip}" ]] || { echo "Could not resolve IP for ${node}" >&2; exit 2; }',
            '  SRTCTL_NODE_IPS+=("${ip}")',
            "done",
            "",
            "declare -a SRTCTL_PIDS=()",
            "declare -a SRTCTL_LABELS=()",
            "cleanup() {",
            "  local rc=$?",
            "  trap - EXIT INT TERM",
            "  set +e",
            '  for pid in "${SRTCTL_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done',
            '  for pid in "${SRTCTL_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done',
            '  exit "${rc}"',
            "}",
            "trap cleanup EXIT INT TERM",
            "",
            "launch() {",
            "  local label=$1",
            "  shift",
            '  echo "Starting ${label}"',
            '  "$@" &',
            '  SRTCTL_PIDS+=("$!")',
            '  SRTCTL_LABELS+=("${label}")',
            "}",
            "",
            "wait_for_port() {",
            '  python3 - "$1" "$2" "$3" "$4" <<\'PY\'',
            "import socket",
            "import sys",
            "import time",
            "host, port, timeout, label = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]",
            "deadline = time.monotonic() + timeout",
            "while time.monotonic() < deadline:",
            "    try:",
            "        with socket.create_connection((host, port), timeout=2):",
            '            print(f"{label} is ready at {host}:{port}")',
            "            raise SystemExit(0)",
            "    except OSError:",
            "        time.sleep(1)",
            'raise SystemExit(f"Timed out waiting for {label} at {host}:{port}")',
            "PY",
            "}",
            "",
            "wait_for_dynamo() {",
            '  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<\'PY\'',
            "import json",
            "import sys",
            "import time",
            "import urllib.request",
            "host, port = sys.argv[1], int(sys.argv[2])",
            "expected_prefill, expected_decode = int(sys.argv[3]), int(sys.argv[4])",
            "timeout, interval = float(sys.argv[5]), float(sys.argv[6])",
            "deadline = time.monotonic() + timeout",
            "while time.monotonic() < deadline:",
            "    try:",
            '        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as response:',
            '            instances = json.load(response).get("instances", [])',
            '        prefills = sum(i.get("endpoint") == "generate" and i.get("component") == "prefill" for i in instances)',
            '        decodes = sum(i.get("endpoint") == "generate" and i.get("component") in {"decode", "tensorrt_llm", "backend"} for i in instances)',
            "        if prefills >= expected_prefill and decodes >= expected_decode:",
            '            print(f"Dynamo workers are ready ({prefills} prefill, {decodes} decode)")',
            "            raise SystemExit(0)",
            "    except Exception:",
            "        pass",
            "    time.sleep(interval)",
            'raise SystemExit(f"Timed out waiting for Dynamo workers at {host}:{port}")',
            "PY",
            "}",
            "",
        ]
    )

    current_stage = ""
    for step, command in rendered_steps:
        if step.stage != current_stage:
            current_stage = step.stage
            lines.extend((f"# {current_stage}",))
        lines.extend((f"launch {shlex.quote(step.label)} \\", f"  {command}"))
        for host, port, timeout, label in step.waits:
            lines.append(
                "wait_for_port "
                f"{_shell_word(host.replace('__SRTCTL_NODE_', '__SRTCTL_IP_'), source_dir=source_dir)} "
                f"{port} {timeout} {shlex.quote(label)}"
            )
        lines.append("")

    expected_prefill = 0 if config.resources.num_agg else config.resources.num_prefill
    expected_decode = config.resources.num_agg or config.resources.num_decode
    health_interval = max(1, int(config.health_check.interval_seconds))
    health_timeout = max(1, int(config.health_check.max_attempts) * health_interval)
    frontend_expression = _shell_word(frontend_ip, source_dir=source_dir)
    frontend_inline = frontend_expression[1:-1] if frontend_expression.startswith('"') else frontend_expression
    lines.extend(
        [
            f"wait_for_dynamo {frontend_expression} {FRONTEND_PUBLIC_PORT} {expected_prefill} {expected_decode} {health_timeout} {health_interval}",
            "",
            f'echo "Dynamo serving stack is ready at http://{frontend_inline}:{FRONTEND_PUBLIC_PORT}"',
            'echo "Logs: ${SRTCTL_LOG_DIR}"',
            'echo "Press Ctrl-C to stop all launched services."',
            "",
            "while true; do",
            '  for index in "${!SRTCTL_PIDS[@]}"; do',
            '    pid="${SRTCTL_PIDS[$index]}"',
            '    if ! kill -0 "${pid}" 2>/dev/null; then',
            "      set +e",
            '      wait "${pid}"',
            "      rc=$?",
            "      set -e",
            '      echo "${SRTCTL_LABELS[$index]} exited with status ${rc}" >&2',
            '      exit "${rc}"',
            "    fi",
            "  done",
            "  sleep 2",
            "done",
            "",
        ]
    )
    return "\n".join(lines)
