# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

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
) -> SrtConfig:
    raw = {
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
            "tachometer": {"enabled": True},
        },
    }
    if dynamo_hash:
        raw["dynamo"] = {"hash": dynamo_hash}
        if cargo_patches:
            raw["dynamo"]["cargo_patches"] = cargo_patches
    expand_observability(raw)
    return SrtConfig.Schema().load(yaml.safe_load(yaml.safe_dump(raw)))


def test_local_lifecycle_renders_eight_tp1_workers_with_separate_logs(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert len(context.worker_processes) == 8
    assert {worker.log_name for worker in context.worker_processes} == {f"worker-{index}.log" for index in range(8)}
    assert all(
        f"CUDA_VISIBLE_DEVICES={index}" in worker.command for index, worker in enumerate(context.worker_processes)
    )
    assert "-m sglang_router.launch_router" in context.router_command
    assert "--policy cache_aware" in context.router_command
    assert 'name = "router"' in context.tachometer_config
    assert script.count('"${LOG_DIR}/worker-') == 8
    assert '"${LOG_DIR}/router.log"' in script
    assert '"${LOG_DIR}/tachometer.log"' in script
    assert "setsid" in script
    assert "kill -TERM 0" not in script
    for forbidden in ("#SBATCH", "SLURM_", "scontrol", "srun", "do_sweep", "run_benchmark"):
        assert forbidden not in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_dynamo_lifecycle_starts_owned_infrastructure(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo"),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert context.needs_dynamo_infra
    assert "-m dynamo.sglang" in context.worker_processes[0].command
    assert "-m dynamo.frontend" in context.router_command
    assert "--model-name fake/mock-model" in context.router_command
    assert "--model-path fake/mock-model" in context.router_command
    assert 'srt_launch "nats"' in script
    assert 'srt_launch "etcd"' in script
    assert "DYN_SYSTEM_PORT=7500" in context.worker_processes[0].command
    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace' in context.router_command
    assert all(
        f"--nccl-port {17_500 + index}" in worker.command for index, worker in enumerate(context.worker_processes)
    )
    assert 'srt_wait_http_ready "http://127.0.0.1:6100/health"' not in script
    assert "srt_wait_router_ready" in script
    assert 'TACHOMETER_STORAGE="${ARTIFACT_DIR}/tachometer/raw/scrape"' in script
    assert '"${SRTCTL_TACHOMETER_DEFAULT}" == "tachometer-scraper"' in script
    assert 'SRTCTL_TACHOMETER_DEFAULT="${SRTCTL_SOURCE}/bin/tachometer-scraper"' in script
    assert (
        'export AIPERF_DATASET_MMAP_BASE_PATH="${AIPERF_DATASET_MMAP_BASE_PATH:-${ARTIFACT_DIR}/aiperf-mmap}"' in script
    )
    assert 'mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}" "${AIPERF_DATASET_MMAP_BASE_PATH}"' in script
    assert context.ruter_enabled
    assert 'SRTCTL_RUTER_PYTHON="${SRTCTL_RUTER_PYTHON:-${SRTCTL_SOURCE}/.venv/bin/python}"' in script
    assert (
        "Observability enabled: Dynamo request tracing is on (DYN_REQUEST_TRACE=1; request-end gzip JSONL: ${ARTIFACT_DIR}/dynamo-request-trace.*.jsonl.gz)"
        in script
    )
    assert '"${SRTCTL_RUTER_PYTHON}" -m srtctl.ruter init "${OUTPUT_DIR}" --output "${LOG_DIR}/.ruter"' in script
    assert '"max_tokens": 16' in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_lifecycle_expands_artifact_dir_in_frontend_environment(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(
            frontend_type="dynamo",
            frontend_env={"DYN_REQUEST_TRACE_FILE_PATH": "{artifact_dir}/dynamo-request-trace.jsonl"},
        ),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace.jsonl' in context.router_command
    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace.jsonl' in script
    assert "export OUTPUT_DIR LOG_DIR ARTIFACT_DIR" in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_dynamo_lifecycle_accepts_isolated_infra_ports(tmp_path) -> None:
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
    script = render_local_lifecycle(context)

    assert "ETCD_ENDPOINTS=http://127.0.0.1:22379" in context.router_command
    assert "NATS_SERVER=nats://127.0.0.1:24222" in context.router_command
    assert '"${SRTCTL_NATS_BINARY}" -js -a "127.0.0.1" -p 24222' in script
    assert "http://127.0.0.1:22379/health" in script
    assert '--listen-peer-urls "http://127.0.0.1:22380"' in script


def test_local_dynamo_lifecycle_caches_a_hash_pinned_source_build(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo", dynamo_hash="a6261680a974ca7c74dcf49592a7376d7de99380"),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert context.dynamo_source_hash == "a6261680a974ca7c74dcf49592a7376d7de99380"
    assert "srt_install_dynamo_from_source_cache" in script
    assert "SRTCTL_DYNAMO_SOURCE_HASH=a6261680a974ca7c74dcf49592a7376d7de99380" in script
    assert 'cache_root="${SRTCTL_DYNAMO_CACHE_ROOT:-${SRTCTL_SOURCE}/configs/dynamo-wheels}"' in script
    assert "flock -x 201" in script
    assert 'maturin build --release --out "${cache}"' in script
    assert '"${SRTCTL_PYTHON}" -m ensurepip --upgrade' in script
    assert 'pip install --quiet --force-reinstall --no-deps "${wheel}"' in script
    assert 'pip install --quiet --editable "${source_dir}/dynamo"' in script
    assert "requirements.sglang.txt" not in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


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
    assert '--detach\n        --name "srtctl-lifecycle-$$"' in script
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
    assert 'if [[ ! -f "${runtime_dir}/.complete" ]]; then' in script
    assert 'touch "${runtime_dir}/.complete"' in script
    assert 'runtime_dir="${SRTCTL_SGLANG_RUNTIME_DIR:-${runtime_root}/sglang-${revision}}"' in script
    assert 'git -c safe.directory="${SRTCTL_SGLANG_SOURCE}"' in script
    assert "Installing source-pinned Rust ${rust_toolchain}" in script
    assert 'rustup toolchain install "${rust_toolchain}" --profile minimal' in script
    assert 'export RUSTUP_TOOLCHAIN="${rust_toolchain}"' in script
    assert (
        'SRTCTL_DYNAMO_CACHE_ROOT="${SRTCTL_DYNAMO_CACHE_ROOT:-${OUTPUT_BASE}/.srtctl-cache/dynamo-wheels}"' in script
    )
    assert "srt_install_sglang_from_source" in script
    assert 'pip install --quiet --editable "${source_copy}/python"' in script
    assert 'pip install --quiet --force-reinstall --no-deps "${wheel}"' in script
    assert 'if [[ "${SRTCTL_LOCAL_CONTAINERIZED:-}" == "1" ]]; then' in script
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libclang-dev protobuf-compiler" in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_dynamo_lifecycle_uses_slurm_cache_key_and_patches(tmp_path) -> None:
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
    script = render_local_lifecycle(context)

    assert context.dynamo_source_cache_key == "a6261680a974ca7c74dcf49592a7376d7de99380-patch-52bdcd85"
    assert context.dynamo_cargo_patch_commands
    assert f"SRTCTL_DYNAMO_SOURCE_CACHE_KEY={context.dynamo_source_cache_key}" in script
    assert "find . -name Cargo.toml -exec sed -i -E" in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
