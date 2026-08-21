# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from srtctl.core.config import expand_observability
from srtctl.core.schema import SrtConfig
from srtctl.render.lifecycle import build_local_lifecycle_render_context, render_local_lifecycle


def _config(
    *,
    frontend_type: str = "sglang",
    frontend_env: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
    dynamo_hash: str | None = None,
    cargo_patches: list[str] | None = None,
    dynamo: dict[str, object] | None = None,
    tachometer: dict[str, object] | None = None,
    setup_script: str | None = None,
    profiling_type: str | None = None,
    mooncake: dict[str, object] | None = None,
) -> SrtConfig:
    raw: dict[str, object] = {
        "name": "direct-render",
        "model": {
            "path": "hf:fake/mock-model",
            "container": "unused-on-direct-host",
            "precision": "fp8",
        },
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 8,
            "gpus_per_agg": 1,
        },
        "backend": {
            "type": "sglang",
            "sglang_config": {
                "aggregated": {
                    "served-model-name": "fake/mock-model",
                    "tp": 1,
                    "enable-metrics": True,
                }
            },
        },
        "frontend": {
            "type": frontend_type,
            "enable_multiple_frontends": False,
            "args": {"policy": "cache_aware"} if frontend_type == "sglang" else {"router-mode": "kv"},
            "env": frontend_env,
        },
        "environment": environment or {},
        "benchmark": {"type": "custom", "command": "aiperf profile --ui none"},
        "observability": {
            "enabled": True,
            "tachometer": tachometer or {"enabled": True},
        },
    }
    if dynamo is not None:
        raw["dynamo"] = dynamo
    elif dynamo_hash:
        raw["dynamo"] = {"hash": dynamo_hash}
        if cargo_patches:
            raw["dynamo"]["cargo_patches"] = cargo_patches
    if setup_script:
        raw["setup_script"] = setup_script
    if profiling_type:
        raw["profiling"] = {
            "type": profiling_type,
            "aggregated": {"start_step": 0, "stop_step": 1},
        }
    if mooncake is not None:
        raw["backend"]["mooncake_kv_store"] = mooncake
    expand_observability(raw)
    return SrtConfig.Schema().load(yaml.safe_load(yaml.safe_dump(raw)))


def _plan(context) -> dict[str, object]:
    return json.loads(context.direct_lifecycle_plan)


def _assert_valid_direct_script(script: str) -> None:
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    assert (
        'exec "${SRTCTL_PYTHON}" "${SRTCTL_SOURCE}/src/srtctl/render/direct_lifecycle.py" --plan "${DIRECT_PLAN}"'
        in script
    )
    assert "direct-lifecycle-plan.json" in script
    assert "srt_launch" not in script
    assert "srt_stop_group" not in script
    for forbidden in ("#SBATCH", "SLURM_", "scontrol", "srun", "do_sweep", "run_benchmark"):
        assert forbidden not in script


def test_local_lifecycle_renders_eight_tp1_workers_with_separate_logs(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)
    plan = _plan(context)

    assert len(context.worker_processes) == 8
    assert {worker.log_name for worker in context.worker_processes} == {f"worker-{index}.log" for index in range(8)}
    assert all(
        f"CUDA_VISIBLE_DEVICES={index}" in worker.command for index, worker in enumerate(context.worker_processes)
    )
    assert "-m sglang_router.launch_router" in context.router_command
    assert "--policy cache_aware" in context.router_command
    assert 'name = "router"' in str(plan["tachometer_config"])
    assert [worker["log_name"] for worker in plan["worker_processes"]] == [f"worker-{index}.log" for index in range(8)]
    assert plan["router_command"] == context.router_command
    _assert_valid_direct_script(script)


def test_local_dynamo_plan_contains_owned_infrastructure_and_observability(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo"),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)
    plan = _plan(context)

    assert context.needs_dynamo_infra
    assert "-m dynamo.sglang" in context.worker_processes[0].command
    assert "--host 0.0.0.0" in context.worker_processes[0].command
    assert "-m dynamo.frontend" in context.router_command
    assert "--model-name fake/mock-model" in context.router_command
    assert "--model-path fake/mock-model" in context.router_command
    assert "DYN_SYSTEM_PORT=7500" in context.worker_processes[0].command
    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace' in context.router_command
    assert all(
        f"--nccl-port {17_500 + index}" in worker.command for index, worker in enumerate(context.worker_processes)
    )
    assert plan["needs_dynamo_infra"] is True
    assert plan["etcd_client_port"] == 2379
    assert plan["nats_port"] == 4222
    assert plan["tachometer_enabled"] is True
    assert plan["ruter_enabled"] is True
    assert plan["benchmark_command"] == "aiperf profile --ui none"
    _assert_valid_direct_script(script)


def test_local_lifecycle_expands_artifact_dir_in_frontend_environment(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(
            frontend_type="dynamo",
            frontend_env={"DYN_REQUEST_TRACE_FILE_PATH": "{artifact_dir}/dynamo-request-trace.jsonl"},
        ),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    plan = _plan(context)

    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace.jsonl' in context.router_command
    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace.jsonl' in str(plan["router_command"])


def test_local_dynamo_plan_accepts_isolated_infra_ports(tmp_path) -> None:
    config = _config(
        frontend_type="dynamo",
        environment={
            "SRTCTL_ETCD_PORT": "22379",
            "SRTCTL_ETCD_PEER_PORT": "22380",
            "SRTCTL_NATS_PORT": "24222",
        },
    )
    context = build_local_lifecycle_render_context(
        config,
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    plan = _plan(context)

    assert "ETCD_ENDPOINTS=http://127.0.0.1:22379" in context.router_command
    assert "NATS_SERVER=nats://127.0.0.1:24222" in context.router_command
    assert plan["etcd_client_port"] == 22379
    assert plan["etcd_peer_port"] == 22380
    assert plan["nats_port"] == 24222


def test_local_dynamo_plan_caches_a_hash_pinned_source_build(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo", dynamo_hash="a6261680a974ca7c74dcf49592a7376d7de99380"),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    plan = _plan(context)

    assert context.dynamo_source_hash == "a6261680a974ca7c74dcf49592a7376d7de99380"
    assert plan["dynamo_source_hash"] == context.dynamo_source_hash
    assert plan["dynamo_source_cache_key"] == context.dynamo_source_cache_key
    assert plan["dynamo_package_version"] is None
    assert plan["dynamo_top_of_tree"] is False


def test_local_lifecycle_can_run_inside_sglang_container(tmp_path) -> None:
    source = tmp_path / "sglang"
    source.mkdir()
    context = build_local_lifecycle_render_context(
        _config(
            frontend_type="dynamo",
            dynamo_hash="a6261680a974ca7c74dcf49592a7376d7de99380",
            environment={
                "SRTCTL_LOCAL_CONTAINER_IMAGE": "lmsysorg/sglang:dev",
                "SRTCTL_SGLANG_SOURCE": str(source),
            },
        ),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert context.local_container_image == "lmsysorg/sglang:dev"
    assert context.sglang_source == str(source)
    assert "SRTCTL_LOCAL_CONTAINERIZED" in script
    assert "--network host" in script
    assert "--ipc host" in script
    assert "--gpus all" in script
    assert 'SRTCTL_MODEL_MOUNT_PATH="${SRTCTL_RENDERED_MODEL_PATH}"' in script
    assert '"$(basename "$(dirname "${SRTCTL_MODEL_MOUNT_PATH}")")" == "snapshots"' in script
    assert 'SRTCTL_MODEL_MOUNT_PATH="$(dirname "$(dirname "${SRTCTL_MODEL_MOUNT_PATH}")")"' in script
    assert 'SRTCTL_CONTAINER_NAME="srtctl-lifecycle-$$"' in script
    assert '--detach\n        --name "${SRTCTL_CONTAINER_NAME}"' in script
    assert '--label "${SRTCTL_CONTAINER_LABEL}"' in script
    assert 'docker ps -aq --filter "label=${SRTCTL_CONTAINER_LABEL}"' in script
    assert 'docker run "${SRT_CONTAINER_ARGS[@]}" >/dev/null' in script
    assert (
        'docker exec "${SRT_CONTAINER_EXEC_ARGS[@]}" "${SRTCTL_CONTAINER_NAME}" bash /run/srtctl-lifecycle.sh' in script
    )
    assert 'docker rm -f "${SRTCTL_CONTAINER_NAME}"' in script
    assert "SRTCTL_OUTPUT_DIR=${OUTPUT_DIR}" in script
    assert "SRTCTL_SGLANG_RUNTIME_DIR=${SRTCTL_SGLANG_RUNTIME_DIR}" in script
    assert 'sudo -n chown -R "${owner}" "${path}"' in script
    assert "--user" not in script
    assert "SRTCTL_HOST_CARGO_HOME" not in script
    assert 'mkdir -p "${OUTPUT_BASE}"' in script
    assert 'if [[ "${mode}" == "readonly" ]]; then' in script
    assert 'mount+=",readonly"' in script
    _assert_valid_direct_script(script)


def test_local_dynamo_plan_uses_slurm_cache_key_and_patches(tmp_path) -> None:
    patch = 'dynamo-tokenizers = { git = "https://github.com/ai-dynamo/frontend-crates", branch = "trace" }'
    context = build_local_lifecycle_render_context(
        _config(
            frontend_type="dynamo",
            dynamo_hash="a6261680a974ca7c74dcf49592a7376d7de99380",
            cargo_patches=[patch],
        ),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    plan = _plan(context)

    assert context.dynamo_source_cache_key == "a6261680a974ca7c74dcf49592a7376d7de99380-patch-52bdcd85"
    assert context.dynamo_cargo_patch_commands
    assert plan["dynamo_source_cache_key"] == context.dynamo_source_cache_key
    assert plan["dynamo_cargo_patch_commands"] == list(context.dynamo_cargo_patch_commands)


def test_local_dynamo_plan_supports_top_of_tree_and_setup_script(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo", dynamo={"top_of_tree": True}, setup_script="install-nixl.sh"),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    plan = _plan(context)

    assert context.dynamo_top_of_tree
    assert context.setup_script == "install-nixl.sh"
    assert plan["dynamo_top_of_tree"] is True
    assert plan["setup_script"] == "install-nixl.sh"


def test_local_dynamo_lifecycle_starts_mooncake_and_injects_worker_environment(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(
            frontend_type="dynamo",
            environment={"SRTCTL_LOCAL_CONTAINER_IMAGE": "lmsysorg/sglang:dev"},
            mooncake={
                "container": "nvcr.io/nvidia/mooncake:latest",
                "env": {"MOONCAKE_PROTOCOL": "rdma"},
                "master_extra_args": ["--custom-flag=true"],
            },
        ),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)
    plan = _plan(context)

    assert context.mooncake_master_command is not None
    assert "--custom-flag=true" in context.mooncake_master_command
    assert context.mooncake_container == "nvcr.io/nvidia/mooncake:latest"
    assert "MOONCAKE_MASTER=127.0.0.1:8700" in context.worker_processes[0].command
    assert "MOONCAKE_TE_META_DATA_SERVER=http://127.0.0.1:8701/metadata" in context.worker_processes[0].command
    assert "MOONCAKE_PROTOCOL=rdma" in context.worker_processes[0].command
    assert plan["mooncake_master_command"] == list(context.mooncake_master_command)
    assert 'SRTCTL_MOONCAKE_CONTAINER_NAME="${SRTCTL_CONTAINER_NAME}-mooncake"' in script
    assert "nvcr.io/nvidia/mooncake:latest" in script
    assert 'docker logs -f "${SRTCTL_MOONCAKE_CONTAINER_NAME}"' in script
    _assert_valid_direct_script(script)


@pytest.mark.parametrize(
    ("dynamo", "version"),
    [
        ({"version": "0.8.0"}, "0.8.0"),
        ({"wheel": "1.2.0.dev20260426"}, "1.2.0.dev20260426"),
    ],
)
def test_local_dynamo_plan_stages_exact_package_wheels(tmp_path, dynamo, version: str) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo", dynamo=dynamo),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    plan = _plan(context)

    assert context.dynamo_source_hash is None
    assert context.dynamo_package_version == version
    assert plan["dynamo_package_version"] == version
    assert plan["dynamo_source_hash"] is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(frontend_type="dynamo", profiling_type="nsys"), "profiling"),
        (_config(frontend_type="dynamo", tachometer={"enabled": True, "storage_subdir": "custom"}), "storage_subdir"),
    ],
)
def test_local_lifecycle_rejects_slurm_only_yaml_features(tmp_path, config: SrtConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_local_lifecycle_render_context(
            config,
            source_dir=tmp_path / "srt-slurm",
            output_base=tmp_path / "outputs",
        )
