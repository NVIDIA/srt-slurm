# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic dispatch tests; a test frontend makes no claims about KV interoperability."""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangProtocol, VLLMProtocol
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import NodePortAllocator
from srtctl.frontends.base import _FRONTENDS
from srtctl.frontends.sglang import SGLangRouterFrontend
from srtctl.frontends.vllm_router import VLLMRouterFrontend
from srtctl.ports import HTTP_PORTS


@pytest.fixture(autouse=True)
def test_frontend(monkeypatch):
    class RoleTestFrontend(SGLangRouterFrontend):
        type = "role-test-router"
        required_backend = None

    monkeypatch.setitem(_FRONTENDS, "role-test-router", RoleTestFrontend)


def recipe():
    return {
        "schema": 2,
        "name": "independent-engines",
        "model": {"path": "/models/test", "container": "default-image", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8, "het_jobs": False},
        "roles": {
            "prefill": {
                "engine": "vllm",
                "nodes": 1,
                "workers": 1,
                "gpus": 2,
                "args": {"tensor-parallel-size": 2},
                "env": {"PREFILL_ONLY": "1"},
                "container": "prefill-image",
            },
            "decode": {"engine": "sglang", "nodes": 1, "workers": 1, "gpus": 2, "args": {"tp-size": 2}},
        },
        "frontend": {"type": "role-test-router", "enable_multiple_frontends": False},
    }


def load(data):
    return SrtConfig.Schema().load(copy.deepcopy(data))


def test_independent_engine_arguments_environments_and_images():
    config = load(recipe())
    assert isinstance(config.backend_for_role("prefill"), VLLMProtocol)
    assert isinstance(config.backend_for_role("decode"), SGLangProtocol)
    assert config.backend_for_role("prefill").get_config_for_mode("prefill") == {"tensor-parallel-size": 2}
    assert config.backend_for_role("decode").get_config_for_mode("decode") == {"tp-size": 2}
    assert config.backend_for_role("prefill").prefill_environment == {"PREFILL_ONLY": "1"}
    assert config.backend_for_role("decode").prefill_environment == {}
    assert config.worker_container_for_role("prefill") == "prefill-image"
    assert config.worker_container_for_role("decode") == "default-image"


@pytest.mark.parametrize("shared_engine", ["sglang", "vllm"])
def test_shared_and_role_engines_are_rejected_at_preflight(shared_engine):
    from srtctl.core.validation import preflight_config_variants

    data = recipe()
    data["engine"] = shared_engine
    # Preflight reports a recipe the loader rejects as a finding on that variant, not as a crash.
    (result,) = preflight_config_variants(data, cluster_config=None)
    assert not result.ok
    assert [issue.code for issue in result.errors] == ["recipe-rejected"]
    assert "cannot be combined with a top-level engine" in result.errors[0].message


def test_explicit_role_engines_do_not_inherit_sibling_options():
    data = recipe()
    data["roles"]["prefill"]["engine"] = "sglang"
    data["roles"]["decode"]["engine"] = {"type": "sglang", "gpu_type": "h100"}
    config = load(data)
    assert config.backend_for_role("decode").gpu_type == "h100"
    assert config.backend_for_role("prefill").gpu_type == SGLangProtocol().gpu_type
    assert config.backend_for_role("prefill").get_config_for_mode("prefill") == {"tensor-parallel-size": 2}
    assert config.backend_for_role("decode").get_config_for_mode("decode") == {"tp-size": 2}


@pytest.mark.parametrize("missing_role", ["prefill", "decode"])
def test_no_default_requires_an_engine_on_every_role(missing_role):
    data = recipe()
    data["roles"]["decode"]["engine"] = "sglang"
    data["roles"][missing_role].pop("engine")
    with pytest.raises(ValidationError, match=rf"roles\.{missing_role}\.engine must name a type"):
        load(data)


def test_roles_without_any_engine_share_the_default_backend():
    data = recipe()
    for spec in data["roles"].values():
        spec.pop("engine")
        spec.pop("container", None)
    config = load(data)
    assert not config.has_role_backends
    assert config.backend_for_role("prefill") is config.backend_for_role("decode")


def test_role_mapping_and_containers_survive_a_schema_round_trip():
    config = load(recipe())
    assert set(config.role_backends) == {"prefill", "decode"}
    dumped = SrtConfig.Schema().dump(config)
    reloaded = SrtConfig.Schema().load(dumped)
    assert reloaded.role_backends == config.role_backends
    assert reloaded.role_containers == config.role_containers


def test_shared_placement_does_not_allocate_both_roles_on_node_zero():
    config = load(recipe())
    endpoints = config.allocate_worker_endpoints(["node0", "node1"])
    assert [(ep.mode, ep.nodes) for ep in endpoints] == [("prefill", ("node0",)), ("decode", ("node1",))]


def test_colocation_shares_allocator_and_allocates_distinct_ports():
    data = recipe()
    data["roles"]["decode"]["nodes"] = "colocate"
    config = load(data)
    endpoints = config.allocate_worker_endpoints(["node0"])
    assert not endpoints[0].gpu_indices.intersection(endpoints[1].gpu_indices)
    allocator = NodePortAllocator()
    reserved_port = allocator.next(HTTP_PORTS, "node0")
    processes = config.worker_processes(endpoints, allocator)
    assert len(processes) == 2
    assert {p.node for p in processes} == {"node0"}
    assert reserved_port not in {p.http_port for p in processes}
    assert len({p.http_port for p in processes}) == len(processes)
    assert len({p.sys_port for p in processes}) == len(processes)


def test_homogeneous_colocation_still_separates_worker_gpus():
    data = recipe()
    data["roles"]["prefill"]["engine"] = "sglang"
    data["roles"]["decode"]["nodes"] = "colocate"
    config = load(data)
    assert config.worker_container_for_role("prefill") == "prefill-image"
    endpoints = config.allocate_worker_endpoints(["node0"])
    processes = config.worker_processes(endpoints)
    assert [(p.endpoint_mode, p.node, p.gpu_indices) for p in processes] == [
        ("prefill", "node0", frozenset({0, 1})),
        ("decode", "node0", frozenset({2, 3})),
    ]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("frontend", {"type": "dynamo"}, "Dynamo"),
        ("dynamo", {"sidecar": True}, "sidecars"),
        ("profiling", {"type": "nsys"}, "profiling"),
        ("observability", {"enabled": True}, "profiling"),
    ],
)
def test_unsupported_job_wide_features_fail_early(field, value, message):
    data = recipe()
    data[field] = value
    with pytest.raises(ValidationError, match=message):
        load(data)


def test_heterogeneous_slurm_job_is_rejected():
    data = recipe()
    data["resources"]["het_jobs"] = True
    with pytest.raises(ValidationError, match="het_jobs"):
        load(data)


def test_required_frontend_engine_checks_overridden_role():
    data = recipe()
    data["frontend"]["type"] = "sglang-router"
    with pytest.raises(ValidationError, match="prefill=vllm"):
        load(data)


def test_role_local_implicit_service_is_not_silently_ignored():
    data = recipe()
    data["roles"]["prefill"]["engine"] = {"type": "vllm", "mooncake_kv_store": {}}
    with pytest.raises(ValidationError, match="Mooncake"):
        load(data)


def test_role_local_discovery_connector_is_not_silently_ignored():
    data = recipe()
    data["roles"]["prefill"]["engine"] = {"type": "vllm", "connector": "moriio"}
    with pytest.raises(ValidationError, match="discovery connectors"):
        load(data)


@pytest.mark.parametrize("grpc_role", ["prefill", "decode"])
def test_role_local_sglang_grpc_is_rejected_before_router_uses_wrong_protocol(grpc_role):
    data = recipe()
    data["roles"]["prefill"]["engine"] = "sglang"
    data["frontend"]["type"] = "sglang-router"
    data["roles"][grpc_role]["args"]["grpc-mode"] = True
    with pytest.raises(ValidationError, match="SGLang gRPC"):
        load(data)


def test_served_model_name_comes_from_the_selected_decode_engine():
    data = recipe()
    data["roles"]["decode"]["engine"] = "vllm"
    data["roles"]["decode"]["args"] = {"served-model-name": "decode-model"}
    assert load(data).served_model_name == "decode-model"


def test_multinode_trt_packing_is_explicitly_rejected():
    data = recipe()
    data["roles"]["prefill"].update(engine="trtllm", gpus=16, nodes=2)
    with pytest.raises(ValidationError, match="multi-node TRT-LLM"):
        load(data)


@pytest.mark.parametrize("role_engines", [False, True])
@pytest.mark.parametrize(
    "prefill_gpus,prefill_engine,message",
    [
        (1, "vllm", "prefill parallelism requires.*2 GPUs"),
        (2, {"type": "vllm", "dp_launch_mode": "per_gpu"}, "requires backend.dp_launch_mode: per_node"),
        (2, "vllm", "different expansion factors"),
    ],
)
def test_vllm_router_validates_prefill_for_shared_and_role_engines(role_engines, prefill_gpus, prefill_engine, message):
    data = recipe()
    data["frontend"]["type"] = "vllm-router"
    data["roles"]["prefill"].update(engine=prefill_engine, gpus=prefill_gpus, args={"data-parallel-size": 2})
    data["roles"]["decode"].update(engine="vllm", gpus=1, args={"data-parallel-size": 1})
    if not role_engines:
        data["engine"] = data["roles"]["prefill"].pop("engine")
        data["roles"]["decode"].pop("engine")

    with pytest.raises(ValidationError, match=message):
        load(data)


def test_vllm_router_role_engines_expand_and_count_each_roles_dp_ranks():
    data = recipe()
    data["frontend"]["type"] = "vllm-router"
    data["roles"]["prefill"]["args"] = {"data-parallel-size": 2}
    data["roles"]["decode"].update(engine="vllm", gpus=4, args={"data-parallel-size": 2, "tensor-parallel-size": 2})
    config = load(data)
    processes = config.worker_processes(config.allocate_worker_endpoints(["node0", "node1"]))
    frontend = VLLMRouterFrontend()

    args = frontend.get_managed_frontend_args(config, config.backend, processes)
    assert args[args.index("--intra-node-data-parallel-size") + 1] == "2"
    assert frontend.health_expectations(config, processes)[:2] == (2, 2)


def test_vllm_role_workers_launch_with_separate_images_args_and_ports(tmp_path):
    data = recipe()
    data["frontend"]["type"] = "vllm-router"
    data["roles"]["decode"].update(
        engine="vllm", nodes="colocate", gpus=4, args={"tensor-parallel-size": 4}, container="decode-image"
    )
    config = load(data)
    runtime = RuntimeContext(
        job_id="7",
        run_name=config.name,
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/test"),
        container_image=Path(config.model.container),
        container_mounts={tmp_path: Path("/logs")},
        gpus_per_node=8,
        network_interface=None,
    )
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.1"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as srun,
    ):
        SweepOrchestrator(config, runtime).start_all_workers()

    prefill, decode = [call.kwargs for call in srun.call_args_list]
    assert [worker["container_image"] for worker in (prefill, decode)] == ["prefill-image", "decode-image"]
    commands = [worker["command"] for worker in (prefill, decode)]
    assert all(command[:3] == ["vllm", "serve", "/model"] for command in commands)
    assert [command[command.index("--tensor-parallel-size") + 1] for command in commands] == ["2", "4"]
    assert len({command[command.index("--port") + 1] for command in commands}) == 2
    assert prefill["env_to_set"]["PREFILL_ONLY"] == "1"
    assert "PREFILL_ONLY" not in decode["env_to_set"]
