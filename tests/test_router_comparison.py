# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the three router-comparison arms.

Arm 1 routes two aggregate SGLang workers through the experimental Rust
``sgl-router``; arm 2 replaces it with a Dynamo frontend over native SGLang
sidecars; arm 3 adds an external worker-selection policy catalog.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import SGLangProtocol, SGLangServerConfig
from srtctl.core.config import expand_observability
from srtctl.core.schema import (
    POLICY_CATALOG_CONFIG_NAME,
    DynamoConfig,
    DynamoPolicyCatalogConfig,
    SrtConfig,
    dynamo_source_cache_key,
    policy_catalog_shell_commands,
)
from srtctl.core.topology import Process
from srtctl.frontends import get_frontend
from srtctl.frontends.dynamo import DynamoFrontend
from srtctl.frontends.sgl_router import SGLRouterFrontend
from srtctl.frontends.static_router import RouterWorker
from srtctl.render.direct_plan import build_direct_plan_context, render_direct_container_shim
from srtctl.render.direct_stages.common import apply_dependency_override

CATALOG = DynamoPolicyCatalogConfig(
    package="sgl-router-cache-aware-dynamo-policy",
    git="https://github.com/ishandhanani/router-sandbox",
    rev="411b4648e7fd50c776f0c787f1abb2592eeef41b",
    crate_subdir="crates/sgl-router-cache-aware",
)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


def _raw_config(
    *,
    frontend: dict[str, object],
    dynamo: dict[str, object],
    observability_enabled: bool = True,
) -> dict[str, object]:
    raw: dict[str, object] = {
        "name": "router-arm",
        "model": {"path": "/models/Qwen3-32B-FP8", "container": "lmsysorg/sglang:dev", "precision": "fp8"},
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 2,
            "gpus_per_agg": 4,
        },
        "environment": {
            "SRTCTL_LOCAL_CONTAINER_IMAGE": "lmsysorg/sglang:dev",
            "SRTCTL_SGLANG_SOURCE": "/src/sglang",
        },
        "frontend": frontend,
        "dynamo": dynamo,
        "backend": {
            "type": "sglang",
            "kv_events_config": {"aggregated": True},
            "sglang_config": {
                "aggregated": {
                    "served-model-name": "Qwen/Qwen3-32B-FP8",
                    "tensor-parallel-size": 4,
                    "page-size": 64,
                    "enable-metrics": True,
                }
            },
        },
        "benchmark": {"type": "custom", "command": "aiperf profile --ui none"},
        "observability": {"enabled": observability_enabled, "tachometer": {"enabled": True}},
    }
    expand_observability(raw)
    return raw


def _config(**kwargs: object) -> SrtConfig:
    raw = _raw_config(**kwargs)  # type: ignore[arg-type]
    return SrtConfig.Schema().load(yaml.safe_load(yaml.safe_dump(raw)))


def _arm1_config(**frontend_extra: object) -> SrtConfig:
    return _config(
        frontend={
            "type": "sgl-router",
            "enable_multiple_frontends": False,
            "args": {"policy": "cache_aware_zmq"},
            **frontend_extra,
        },
        dynamo={"install": False},
    )


def _arm2_config() -> SrtConfig:
    return _config(
        frontend={"type": "dynamo", "enable_multiple_frontends": False, "args": {"router-mode": "kv"}},
        dynamo={"hash": "4dca1626b2708383514fcd65ac816ba7429c05a4", "event_plane": "zmq", "sidecar": True},
    )


def _arm3_config() -> SrtConfig:
    return _config(
        frontend={"type": "dynamo", "enable_multiple_frontends": False, "args": {"router-mode": "kv"}},
        dynamo={
            "hash": "4dca1626b2708383514fcd65ac816ba7429c05a4",
            "event_plane": "zmq",
            "sidecar": True,
            "policy_catalog": {
                "git": CATALOG.git,
                "rev": CATALOG.rev,
                "crate_subdir": CATALOG.crate_subdir,
                "package": CATALOG.package,
                "config": "worker-selection.yaml",
            },
        },
    )


def _direct_plan(config: SrtConfig, tmp_path: Path) -> dict[str, object]:
    context = build_direct_plan_context(config, source_dir=tmp_path / "srt-slurm", output_base=tmp_path / "outputs")
    script = render_direct_container_shim(context)
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    return json.loads(context.direct_plan_json)


# ---------------------------------------------------------------------------
# Arm 1: raw experimental router
# ---------------------------------------------------------------------------


def test_experimental_router_is_registered_and_status_gated() -> None:
    frontend = get_frontend("sgl-router")
    assert isinstance(frontend, SGLRouterFrontend)
    # `/readyz` answers with a bare status code, so readiness must not parse JSON.
    assert frontend.health_endpoint == "/readyz"
    assert frontend.health_status_only is True


def test_experimental_router_command_advertises_every_aggregate_worker() -> None:
    frontend = SGLRouterFrontend()
    command = frontend.build_router_command(
        [RouterWorker("agg", "http://10.0.0.1:6100"), RouterWorker("agg", "http://10.0.0.1:6132")],
        "0.0.0.0",
        8000,
    )

    assert command[0] == "sgl-router"
    assert command[1:4] == ["--worker-urls", "http://10.0.0.1:6100", "http://10.0.0.1:6132"]
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8000"]
    # Static discovery has no prefill/decode split.
    assert "--pd-disaggregation" not in command


def test_experimental_router_rejects_disaggregated_workers() -> None:
    with pytest.raises(ValueError, match="aggregate workers only"):
        SGLRouterFrontend().build_router_command(
            [RouterWorker("prefill", "http://10.0.0.1:6100"), RouterWorker("decode", "http://10.0.0.2:6100")],
            "0.0.0.0",
            8000,
        )


def test_experimental_router_gates_on_every_worker_not_just_the_first() -> None:
    """`/readyz` flips on one registration, so the barrier lives on the workers."""
    frontend = SGLRouterFrontend()
    processes = [
        SimpleNamespace(endpoint_mode="agg", node="node0", http_port=6100, bootstrap_port=None),
        SimpleNamespace(endpoint_mode="agg", node="node0", http_port=6132, bootstrap_port=None),
    ]
    with patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"):
        urls = frontend.get_backend_health_urls(None, processes, None)
    assert urls == ["http://10.0.0.1:6100/health", "http://10.0.0.1:6132/health"]


def test_experimental_router_supplies_required_model_identity() -> None:
    config = _arm1_config()
    managed = SGLRouterFrontend().get_managed_frontend_args(config, config.backend, [])
    # A bare model directory is rejected by the router's tokenizer loader.
    assert managed == ["--model-id", "Qwen/Qwen3-32B-FP8", "--tokenizer-path", "/model/tokenizer.json"]


def test_experimental_router_yields_to_explicit_model_identity() -> None:
    config = _arm1_config(args={"policy": "cache_aware_zmq", "model-id": "custom", "tokenizer-path": "/t.json"})
    assert SGLRouterFrontend().get_managed_frontend_args(config, config.backend, []) == []


def test_experimental_router_builds_from_source_and_uses_that_binary() -> None:
    config = _arm1_config(source="/src/sglang/experimental/sgl-router")
    frontend = SGLRouterFrontend()
    assert frontend.build_bash_preamble(config) == (
        "cargo build --release --manifest-path /src/sglang/experimental/sgl-router/Cargo.toml"
    )

    runtime = MagicMock()
    runtime.log_dir = Path("/tmp")
    runtime.network_interface = None
    runtime.container_image = "lmsysorg/sglang:dev"
    runtime.container_mounts = []
    runtime.environment = {}
    runtime.nodes.het_group_for.return_value = None
    topology = SimpleNamespace(frontend_nodes=["node0"], frontend_port=8000)
    processes = [SimpleNamespace(endpoint_mode="agg", node="node0", http_port=6100, bootstrap_port=None)]

    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch.object(SGLRouterFrontend, "start_process", return_value=MagicMock()) as start,
    ):
        frontend.start_frontends(topology, runtime, config, config.backend, processes)

    command = start.call_args.kwargs["command"]
    assert command[0] == "/src/sglang/experimental/sgl-router/target/release/sgl-router"
    assert "cache_aware_zmq" in command


def test_experimental_router_workers_launch_sglang_not_dynamo() -> None:
    backend = SGLangProtocol(
        kv_events_config={"aggregated": True},
        sglang_config=SGLangServerConfig(aggregated={"tensor-parallel-size": 4}),
    )
    process = Process(
        node="node0",
        gpu_indices=frozenset(range(4)),
        sys_port=7500,
        http_port=6100,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=0,
        kv_events_port=5200,
    )
    runtime = MagicMock()
    runtime.model_path = Path("/models/Qwen3-32B-FP8")
    runtime.is_hf_model = False
    runtime.dynamo = DynamoConfig(install=False)

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(process, [process], runtime, frontend_type="sgl-router")

    assert command[:3] == ["python3", "-m", "sglang.launch_server"]
    assert "--request-plane" not in command
    kv_events = json.loads(command[command.index("--kv-events-config") + 1])
    assert kv_events["endpoint"] == "tcp://*:5200"


def test_experimental_router_rejects_prefill_decode_topology() -> None:
    raw = _raw_config(
        frontend={"type": "sgl-router", "enable_multiple_frontends": False},
        dynamo={"install": False},
    )
    raw["resources"] = {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    }
    with pytest.raises(ValidationError, match="aggregate worker pool"):
        SrtConfig.Schema().load(yaml.safe_load(yaml.safe_dump(raw)))


def test_direct_arm1_runs_raw_sglang_behind_the_router(tmp_path: Path) -> None:
    plan = _direct_plan(_arm1_config(source="/src/sglang/experimental/sgl-router"), tmp_path)

    assert plan["frontend_kind"] == "sgl-router"
    assert plan["dynamo_enabled"] is False
    assert plan["router_health_path"] == "/readyz"
    assert plan["dynamo_source_cache_key"] is None
    assert plan["sgl_router_source_subdir"] == "experimental/sgl-router"
    assert len(plan["worker_processes"]) == 2

    router = str(plan["router_command"])
    assert router.startswith('"${SRTCTL_SGL_ROUTER_BIN}"')
    assert "--policy cache_aware_zmq" in router
    assert "--tokenizer-path /models/Qwen3-32B-FP8/tokenizer.json" in router
    assert "--worker-urls http://127.0.0.1:6100 http://127.0.0.1:6132" in router

    worker = str(plan["worker_processes"][0]["command"])
    assert "-m sglang.launch_server" in worker
    assert "dynamo" not in worker
    assert "ETCD_ENDPOINTS" not in worker
    assert '"endpoint":"tcp://*:5200"' in worker
    # Every worker is gated directly, since /readyz needs only one registration.
    assert plan["worker_health_urls"] == ["http://127.0.0.1:6100/health", "http://127.0.0.1:6132/health"]
    # Raw SGLang publishes Prometheus on its serving port, not a Dynamo system port.
    assert 'url = "http://127.0.0.1:6100/metrics"' in str(plan["tachometer_config"])


def test_direct_arm1_rejects_an_opaque_prebuilt_router(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="drop frontend.binary"):
        build_direct_plan_context(
            _arm1_config(binary="/opt/bin/sgl-router"),
            source_dir=tmp_path / "srt-slurm",
            output_base=tmp_path / "outputs",
        )


# ---------------------------------------------------------------------------
# Arm 2: native sidecars on the direct host
# ---------------------------------------------------------------------------


def test_direct_arm2_couples_engine_and_sidecar_lifecycles(tmp_path: Path) -> None:
    plan = _direct_plan(_arm2_config(), tmp_path)

    assert plan["frontend_kind"] == "dynamo"
    assert plan["dynamo_enabled"] is True
    worker = str(plan["worker_processes"][0]["command"])

    # The engine is the native server; the sidecar is the Dynamo registration.
    assert "-m sglang.launch_server" in worker
    assert "-m dynamo.sglang.sidecar --grpc-endpoint 127.0.0.1:50051" in worker
    assert "--grpc-port 50051" in worker
    assert "-m dynamo.sglang " not in worker

    # The sidecar only starts once the engine's gRPC port binds ...
    assert "/dev/tcp/127.0.0.1/50051" in worker
    # ... and either process exiting takes the other down.
    assert 'wait -n "${ENGINE_PID}" "${SIDECAR_PID}"' in worker
    assert "trap cleanup EXIT INT TERM" in worker

    # Co-located workers must not collide on the sidecar gRPC port.
    second = str(plan["worker_processes"][1]["command"])
    assert "--grpc-port 50052" in second
    assert "-m dynamo.sglang.sidecar --grpc-endpoint 127.0.0.1:50052" in second

    # KV events and Dynamo discovery survive the sidecar rewrite.
    assert '"endpoint":"tcp://*:5200"' in worker
    assert "ETCD_ENDPOINTS=http://127.0.0.1:2379" in worker
    assert "DYN_EVENT_PLANE=zmq" in worker


def test_direct_without_sidecar_keeps_the_legacy_dynamo_worker(tmp_path: Path) -> None:
    config = _config(
        frontend={"type": "dynamo", "enable_multiple_frontends": False, "args": {"router-mode": "kv"}},
        dynamo={"hash": "4dca1626b2708383514fcd65ac816ba7429c05a4"},
    )
    worker = str(_direct_plan(config, tmp_path)["worker_processes"][0]["command"])
    assert "-m dynamo.sglang " in worker
    assert "dynamo.sglang.sidecar" not in worker
    assert "--request-plane tcp" in worker


# ---------------------------------------------------------------------------
# Arm 3: external worker-selection policy catalog
# ---------------------------------------------------------------------------


def test_policy_catalog_requires_a_pinned_dynamo_revision() -> None:
    with pytest.raises(ValueError, match="pinned source build"):
        DynamoConfig(top_of_tree=True, policy_catalog=CATALOG)


def test_policy_catalog_requires_the_dynamo_frontend() -> None:
    raw = _raw_config(
        frontend={"type": "sgl-router", "enable_multiple_frontends": False},
        dynamo={
            "hash": "4dca1626b2708383514fcd65ac816ba7429c05a4",
            "policy_catalog": {"git": CATALOG.git, "rev": CATALOG.rev, "package": CATALOG.package},
        },
    )
    with pytest.raises(ValidationError, match="requires frontend.type: dynamo"):
        SrtConfig.Schema().load(yaml.safe_load(yaml.safe_dump(raw)))


def test_policy_catalog_needs_exactly_one_immutable_source() -> None:
    with pytest.raises(ValueError, match="exactly one of git or path"):
        DynamoPolicyCatalogConfig(package="p", git="https://example.com/x", path="/x")
    with pytest.raises(ValueError, match="immutable rev"):
        DynamoPolicyCatalogConfig(package="p", git="https://example.com/x")


def test_policy_catalog_partitions_the_dynamo_build_cache() -> None:
    dynamo_hash = "4dca1626b2708383514fcd65ac816ba7429c05a4"
    plain = dynamo_source_cache_key(dynamo_hash)
    linked = dynamo_source_cache_key(dynamo_hash, None, CATALOG)
    assert plain == dynamo_hash
    assert linked.startswith(f"{dynamo_hash}-policy-")
    assert linked != plain


def test_policy_catalog_build_pins_kv_router_to_the_same_checkout() -> None:
    commands = policy_catalog_shell_commands(CATALOG, repo_var="PWD", workdir_var="DYN_BUILD_DIR")
    joined = "\n".join(commands)

    # The catalog source is materialized at the requested immutable revision ...
    assert f'git -C "$DYN_BUILD_DIR/policy-catalog" fetch --depth 1 origin {CATALOG.rev}' in joined
    # ... its dynamo-kv-router is repointed at the checkout being built, so the
    # plugin's router types unify with the ones the bindings compile against ...
    assert 'dynamo-kv-router = { path = \\"$DYN_BUILD_DIR/dynamo/lib/kv-router\\"' in joined
    assert 'features = [\\"standalone-selection\\"]' in joined
    # ... and it replaces Dynamo's own catalog dependency alias.
    assert f'dynamo-worker-selection-policy-catalog = {{ package = \\"{CATALOG.package}\\"' in joined
    assert '"$PWD"/lib/bindings/python/Cargo.toml' in joined


def test_policy_catalog_install_enables_the_custom_policy_feature() -> None:
    install = DynamoConfig(
        hash="4dca1626b2708383514fcd65ac816ba7429c05a4", policy_catalog=CATALOG
    ).get_install_commands()
    assert "maturin build --release --features custom-policy -o /tmp" in install
    assert f"/{POLICY_CATALOG_CONFIG_NAME}" in install


def test_dependency_override_replaces_only_the_named_declaration() -> None:
    original = (
        "[dependencies]\n"
        'dynamo-kv-router = { git = "https://github.com/ai-dynamo/dynamo.git", branch = "main" }\n'
        'serde = "1"\n'
    )
    patched = apply_dependency_override(
        original, "dynamo-kv-router", 'dynamo-kv-router = { path = "/d/lib/kv-router" }'
    )
    assert 'dynamo-kv-router = { path = "/d/lib/kv-router" }' in patched
    assert "github.com/ai-dynamo/dynamo.git" not in patched
    assert 'serde = "1"' in patched


def test_dynamo_frontend_passes_the_published_policy_config() -> None:
    config = _arm3_config()
    published = config.dynamo.policy_config_path
    assert published is not None
    assert published.endswith(f"{config.dynamo.source_cache_key}/{POLICY_CATALOG_CONFIG_NAME}")
    assert DynamoFrontend().get_managed_frontend_args(config) == ["--router-policy-config", published]


def test_explicit_router_policy_config_wins_over_the_derived_path() -> None:
    config = _config(
        frontend={
            "type": "dynamo",
            "enable_multiple_frontends": False,
            "args": {"router-mode": "kv", "router-policy-config": "/configs/mine.yaml"},
        },
        dynamo={
            "hash": "4dca1626b2708383514fcd65ac816ba7429c05a4",
            "policy_catalog": {"git": CATALOG.git, "rev": CATALOG.rev, "package": CATALOG.package},
        },
    )
    assert DynamoFrontend().get_managed_frontend_args(config) == []


def test_direct_arm3_plumbs_the_catalog_and_its_policy_config(tmp_path: Path) -> None:
    plan = _direct_plan(_arm3_config(), tmp_path)

    catalog = plan["dynamo_policy_catalog"]
    assert catalog is not None
    assert catalog["package"] == CATALOG.package
    assert catalog["rev"] == CATALOG.rev
    assert catalog["features"] == ["custom-policy"]
    assert plan["dynamo_policy_config_name"] == POLICY_CATALOG_CONFIG_NAME
    # The published path is only known after the build, so the router command
    # defers to the runtime export rather than baking a render-time guess.
    assert '--router-policy-config "${SRTCTL_POLICY_CONFIG}"' in str(plan["router_command"])
    assert str(plan["dynamo_source_cache_key"]).startswith("4dca1626b2708383514fcd65ac816ba7429c05a4-policy-")


def test_arm2_and_arm3_differ_only_in_worker_selection(tmp_path: Path) -> None:
    arm2 = _direct_plan(_arm2_config(), tmp_path)
    arm3 = _direct_plan(_arm3_config(), tmp_path)

    assert [worker["command"] for worker in arm2["worker_processes"]] == [
        worker["command"] for worker in arm3["worker_processes"]
    ]
    assert arm2["dynamo_policy_catalog"] is None
    assert arm3["dynamo_policy_catalog"] is not None
    assert "--router-policy-config" not in str(arm2["router_command"])
