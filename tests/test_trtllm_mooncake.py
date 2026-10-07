# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM Mooncake pool launch contract, without Slurm or GPUs."""

from pathlib import Path

import pytest
import yaml

from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

TEST_RECIPE = """
benchmark:
  concurrencies: 4x8
  isl: 128
  osl: 128
  type: sa-bench
engine:
  served_model_name: Qwen/Qwen3-0.6B
  type: trtllm
frontend:
  enable_multiple_frontends: false
  type: trtllm_serve
model:
  container: mock-trtllm.sqsh
  path: hf:fake/mock-model
  precision: bf16
name: trtllm-mooncake-test
resources:
  gpu_type: h100
  gpus_per_node: 8
roles:
  decode:
    args:
      backend: pytorch
      cache_transceiver_config:
        backend: NIXL
      max_batch_size: 64
      max_num_tokens: 4096
      max_seq_len: 4096
      tensor_parallel_size: 1
      trust_remote_code: true
    gpus: 1
    nodes: 1
    workers: 1
  prefill:
    args:
      backend: pytorch
      cache_transceiver_config:
        backend: NIXL
      kv_cache_config:
        disk_cache_size: 0
        host_cache_size: 0
        use_kv_cache_manager_v2: true
      kv_connector_config:
        connector: mooncake-store
        mooncake_store:
          global_segment_size: 4GiB
          master_server_address: file:///logs/mooncake_master.addr
          protocol: rdma
      max_batch_size: 8
      max_num_tokens: 8192
      max_seq_len: 4096
      tensor_parallel_size: 1
      trust_remote_code: true
    gpus: 1
    nodes: 1
    workers: 1
schema: 2
services:
- name: mooncake-master
  options:
    eviction_ratio: 0.05
    master_timeout_s: 900
    store_role: both
  start: with_workers
  type: mooncake-master
- name: mooncake-donor
  options:
    protocol: rdma
    size: 4GiB
  placement:
    node: decode
  readiness:
    file:
      path: mooncake/donor-{node}.ready
    timeout_seconds: 900
  start: with_workers
  type: mooncake-donor
slurm:
  time_limit: 00:30:00
"""


def pool_recipe() -> dict:
    return yaml.safe_load(TEST_RECIPE)


@pytest.mark.parametrize("donor_nodes", [1, 2])
def test_pool_launches_workers_before_waiting_for_master_and_donors(tmp_path: Path, monkeypatch, donor_nodes) -> None:
    from srtctl.cli.mixins.service_stage import ServiceStageMixin

    calls: list[dict] = []
    probes: list[str] = []
    original_wait = ServiceStageMixin._wait_service_ready

    def check_order(self, proc, service, ctx):
        if service.type in ("mooncake-master", "mooncake-donor"):
            workers = [call for call in calls if call.get("step_name", "").startswith(("prefill_", "decode_"))]
            assert len(workers) == 1 + donor_nodes
            assert sum(call.get("step_name", "").startswith("service_mooncake-donor") for call in calls) == donor_nodes
            assert not any(call.get("step_name") == "trtllm_serve_orchestrator" for call in calls)
            probes.append(service.type)
        return original_wait(self, proc, service, ctx)

    monkeypatch.setattr(ServiceStageMixin, "_wait_service_ready", check_order)
    output = tmp_path / "outputs" / "42042"
    recipe = pool_recipe()
    recipe["resources"]["spread_workers"] = True
    recipe["roles"]["decode"].update(nodes=donor_nodes, workers=donor_nodes)
    recipe["model"]["path"] = "hf:fake/mock-model"
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe))
    exit_code = run_mock_sweep(
        config_path=recipe_path,
        output_dir=output,
        job_id="42042",
        options=MockOptions(
            nodelist=tuple(f"mock-node-{i:02d}" for i in range(1, donor_nodes + 2)),
            on_srun=calls.append,
        ),
    )

    assert exit_code == 0
    assert probes == ["mooncake-master"] + ["mooncake-donor"] * donor_nodes
    master = next(i for i, call in enumerate(calls) if call.get("step_name") == "service_mooncake-master")
    donor = next(i for i, call in enumerate(calls) if call.get("step_name", "").startswith("service_mooncake-donor"))
    workers = [i for i, call in enumerate(calls) if call.get("step_name", "").startswith(("prefill_", "decode_"))]
    assert master < donor < min(workers)
    assert calls[master]["command"][:4] == ["trtllm-serve", "mooncake_master", "--rpc_port", "8700"]
    assert calls[master]["command"][-2:] == ["--address_file", "/logs/mooncake_master.addr"]
    assert calls[master]["srun_options"]["mpi"] == "none"
    assert calls[master]["env_to_set"]["TRTLLM_MOONCAKE_MASTER_TIMEOUT"] == "900"
    assert "--master_server_address" in calls[donor]["command"]
    assert calls[donor]["command"][:2] == ["trtllm-serve", "mooncake_donor"]
    assert "file:///logs/mooncake_master.addr" in calls[donor]["command"]
    assert calls[donor]["srun_options"]["mpi"] == "none"
    assert calls[donor]["env_to_set"]["TLLM_LOG_LEVEL"] == "INFO"
    assert calls[donor]["env_to_set"]["TRTLLM_MOONCAKE_MASTER_TIMEOUT"] == "900"
    assert calls[donor]["nodelist"] == ["mock-node-02"]
    prefill = next(call for call in calls if call.get("step_name", "").startswith("prefill_"))
    assert prefill["env_to_set"]["TRTLLM_MOONCAKE_STORE_ROLE"] == "both"
    assert prefill["env_to_set"]["TRTLLM_MOONCAKE_MASTER_TIMEOUT"] == "900"
    assert "TRTLLM_MOONCAKE_RUN_DIR" in prefill["bash_preamble"]
    assert "/logs/mooncake/prefill-0" in prefill["bash_preamble"]


@pytest.mark.parametrize("failed_service", ["mooncake-master", "mooncake-donor"])
def test_concurrent_pool_readiness_failure_stops_run(tmp_path: Path, monkeypatch, failed_service) -> None:
    from srtctl.cli.mixins.service_stage import ServiceStageMixin
    from srtctl.core.processes import ProcessRegistry

    calls: list[dict] = []
    terminated: list[str] = []
    original_wait = ServiceStageMixin._wait_service_ready
    original_cleanup = ProcessRegistry.cleanup

    def fail_probe(self, proc, service, ctx):
        if service.type == failed_service:
            raise RuntimeError("simulated pool readiness failure")
        return original_wait(self, proc, service, ctx)

    def record_cleanup(self):
        terminated.extend(self.get_all_processes())
        return original_cleanup(self)

    monkeypatch.setattr(ServiceStageMixin, "_wait_service_ready", fail_probe)
    monkeypatch.setattr(ProcessRegistry, "cleanup", record_cleanup)
    recipe = pool_recipe()
    recipe["model"]["path"] = "hf:fake/mock-model"
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    result = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "outputs",
        job_id="42043",
        options=MockOptions(nodelist=("mock-node-01", "mock-node-02"), on_srun=calls.append),
    )
    assert result != 0
    assert not any(call.get("step_name") == "trtllm_serve_orchestrator" for call in calls)
    for prefix in ("service_mooncake-master", "service_mooncake-donor", "prefill_", "decode_"):
        assert any(name.startswith(prefix) for name in terminated), terminated


def test_pool_rejects_missing_prefill_connector(tmp_path: Path) -> None:
    recipe = pool_recipe()
    del recipe["roles"]["prefill"]["args"]["kv_connector_config"]
    path = tmp_path / "missing-connector.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(Exception, match="kv_connector_config.connector: mooncake-store"):
        SrtConfig.from_yaml(path)


@pytest.mark.parametrize("explicit_probe", [False, True])
def test_donor_file_readiness_without_info_logs_clears_stale_marker(
    tmp_path: Path, monkeypatch, explicit_probe
) -> None:
    from srtctl.cli.mixins.service_stage import ServiceStageMixin
    from srtctl.core.readiness import run_probe
    from srtctl.services import FileProbe

    output = tmp_path / "outputs"
    marker = output / "logs/mooncake/donor-mock-node-02.ready"
    marker.parent.mkdir(parents=True)
    marker.write_text("stale previous donor\n")
    checked: list[str] = []
    original_wait = ServiceStageMixin._wait_ready

    def on_srun(call):
        if call.get("step_name", "").startswith("service_mooncake-donor"):
            assert not marker.exists(), "stale marker must be removed before srun"
            assert call["env_to_set"]["TLLM_LOG_LEVEL"] == "ERROR"
            marker.write_text("10.0.0.2 4294967296\n")

    def check_probe(proc, service, readiness):
        if service.type == "mooncake-donor":
            assert isinstance(readiness.probe, FileProbe)
            assert readiness.probe.path == "mooncake/donor-mock-node-02.ready"
            assert run_probe(readiness.probe, host=proc.node, log_file=proc.log_file)
            checked.append(service.name)
            return
        original_wait(proc, service, readiness)

    monkeypatch.setattr(ServiceStageMixin, "_wait_ready", staticmethod(check_probe))
    recipe = pool_recipe()
    recipe["model"]["path"] = "hf:fake/mock-model"
    recipe["services"][1]["env"] = {"TLLM_LOG_LEVEL": "ERROR"}
    if not explicit_probe:
        del recipe["services"][1]["readiness"]
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    result = run_mock_sweep(
        config_path=path,
        output_dir=output,
        job_id="42044",
        options=MockOptions(nodelist=("mock-node-01", "mock-node-02"), on_srun=on_srun),
    )
    assert result == 0
    assert checked == ["mooncake-donor"]


def test_pool_rejects_donor_protocol_mismatch(tmp_path: Path) -> None:
    recipe = pool_recipe()
    recipe["services"][1]["options"]["protocol"] = "tcp"
    path = tmp_path / "mismatched-protocol.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(Exception, match="options.protocol must match prefill mooncake_store.protocol"):
        SrtConfig.from_yaml(path)


def test_pool_rejects_missing_connector_preset(tmp_path: Path) -> None:
    recipe = pool_recipe()
    del recipe["roles"]["prefill"]["args"]["kv_connector_config"]["connector"]
    path = tmp_path / "missing-preset.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(Exception, match="connector: mooncake-store"):
        SrtConfig.from_yaml(path)


@pytest.mark.parametrize("frontend_type", ["dynamo", "trtllm_serve"])
def test_pool_provisioning_entrypoint(tmp_path: Path, frontend_type: str) -> None:
    """Only Dynamo pool clients need the serve.py provisioning adapter."""
    recipe = pool_recipe()
    recipe["frontend"]["type"] = frontend_type
    recipe["model"]["path"] = "hf:fake/mock-model"
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    calls: list[dict] = []
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "outputs" / "42042",
            job_id="42042",
            options=MockOptions(on_srun=calls.append),
        )
        == 0
    )
    prefill = next(call for call in calls if call.get("step_name", "").startswith("prefill_"))
    decode = next(call for call in calls if call.get("step_name", "").startswith("decode_"))
    if frontend_type == "dynamo":
        cmd = prefill["command"]
        launcher = cmd.index("trtllm-llmapi-launch")
        assert cmd[launcher + 1 : launcher + 3] == ["python3", "-c"]
        assert "maybe_provision_pool" in cmd[launcher + 3]
        assert cmd[launcher + 4] == "/logs/trtllm_config_prefill.yaml"
        assert cmd[cmd.index("--extra-engine-args") + 1] == "/logs/trtllm_config_prefill.yaml"
        assert "TRTLLM_MOONCAKE_RUN_DIR" in prefill["bash_preamble"]
        assert "dynamo.trtllm" in decode["command"]
    else:
        assert "trtllm-serve" in prefill["command"]
    assert not any("maybe_provision_pool" in arg for arg in decode["command"])


@pytest.mark.parametrize("worker_fails", [False, True])
def test_dynamo_pool_context_covers_worker_lifetime(tmp_path: Path, monkeypatch, worker_fails: bool) -> None:
    """Execute the generated adapter with tekit/Dynamo seams, without GPU imports."""
    import os
    import runpy
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    from srtctl.backends.trtllm import _DYNAMO_MOONCAKE_ENTRYPOINT

    path = tmp_path / "engine.yaml"
    connector = {"connector": "mooncake-store", "mooncake_store": {"global_segment_size": "240GiB"}}
    path.write_text(yaml.safe_dump({"kv_connector_config": connector}))
    events = []

    @contextmanager
    def provision(config):
        assert vars(config) == connector
        events.append("provision")
        monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(tmp_path / "mooncake.json"))
        try:
            yield
        finally:
            events.append("cleanup")

    def run_module(name, *, run_name, alter_sys):
        assert (name, run_name, alter_sys) == ("dynamo.trtllm", "__main__", True)
        assert sys.argv[1:] == ["--extra-engine-args", str(path), "--disaggregation-mode", "prefill"]
        assert os.environ["MOONCAKE_CONFIG_PATH"] == str(tmp_path / "mooncake.json")
        events.append("worker")
        if worker_fails:
            raise RuntimeError("worker failed")

    monkeypatch.setitem(
        sys.modules, "tensorrt_llm.llmapi.llm_args", SimpleNamespace(KvCacheConnectorConfig=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
        SimpleNamespace(maybe_provision_pool=provision),
    )
    monkeypatch.setattr(runpy, "run_module", run_module)
    monkeypatch.setattr(
        sys, "argv", ["-c", str(path), "--extra-engine-args", str(path), "--disaggregation-mode", "prefill"]
    )
    if worker_fails:
        with pytest.raises(RuntimeError, match="worker failed"):
            exec(_DYNAMO_MOONCAKE_ENTRYPOINT, {})
    else:
        exec(_DYNAMO_MOONCAKE_ENTRYPOINT, {})
    assert events == ["provision", "worker", "cleanup"]
