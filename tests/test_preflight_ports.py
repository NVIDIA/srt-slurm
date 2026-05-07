# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SweepOrchestrator.preflight_check_ports.

We can't actually srun in unit tests; instead we patch start_srun_process to
return a fake Popen whose communicate() yields known stdout/returncode, then
verify the orchestrator's parsing/abort logic.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends.sglang import SGLangProtocol, SGLangServerConfig
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import (
    FrontendConfig,
    ResourceConfig,
    SrtConfig,
)


def _make_runtime(job_id: str = "144826") -> RuntimeContext:
    return RuntimeContext(
        job_id=job_id,
        run_name="test-run",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path("/models/test"),
        container_image=Path("/path/to/container.sqsh"),
        gpus_per_node=8,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def _make_config_with_dp_attention() -> SrtConfig:
    """SGLang prefill recipe with enable-dp-attention=true (the failing case)."""
    return SrtConfig(
        name="test-dp",
        model={"path": "test-model", "container": "test.sqsh", "precision": "fp16"},
        resources=ResourceConfig(
            gpu_type="a100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=1,
            prefill_workers=1,
            decode_workers=1,
        ),
        frontend=FrontendConfig(type="dynamo"),
        backend=SGLangProtocol(
            sglang_config=SGLangServerConfig(
                prefill={"enable-dp-attention": True, "data-parallel-size": 4},
                decode={},
            ),
        ),
    )


def _make_config_no_dp_attention() -> SrtConfig:
    """SGLang recipe without DP attention — preflight should be a no-op."""
    return SrtConfig(
        name="test-no-dp",
        model={"path": "test-model", "container": "test.sqsh", "precision": "fp16"},
        resources=ResourceConfig(
            gpu_type="a100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=1,
            prefill_workers=1,
            decode_workers=1,
        ),
        frontend=FrontendConfig(type="dynamo"),
        backend=SGLangProtocol(
            sglang_config=SGLangServerConfig(prefill={}, decode={}),
        ),
    )


def _fake_popen(stdout: bytes, returncode: int) -> MagicMock:
    proc = MagicMock()
    proc.communicate.return_value = (stdout, b"")
    proc.returncode = returncode
    return proc


class TestPreflightCheckPorts:
    def test_skips_when_no_dp_attention_ports(self, caplog):
        config = _make_config_no_dp_attention()
        runtime = _make_runtime()
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        with (
            patch("srtctl.cli.do_sweep.start_srun_process") as mock_srun,
            caplog.at_level("DEBUG", logger="srtctl.cli.do_sweep"),
        ):
            orchestrator.preflight_check_ports()
        # No srun should have been spawned because no process has DP-attn ports.
        mock_srun.assert_not_called()

    def test_skips_when_backend_does_not_implement_method(self):
        """Backends without dp_attention_tcp_ports (TRTLLM/vLLM/Mocker) skip cleanly.

        The orchestrator uses getattr(backend, "dp_attention_tcp_ports", None)
        so unverified backends fall through without crashing.
        """
        config = _make_config_with_dp_attention()
        runtime = _make_runtime()
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        # Replace the orchestrator's `backend` property with one that lacks
        # the optional method, simulating TRTLLM/vLLM/Mocker.
        class _BareBackend:
            type = "test-bare"
            # Intentionally no dp_attention_tcp_ports attribute.

        with (
            patch.object(SweepOrchestrator, "backend", new=_BareBackend()),
            patch("srtctl.cli.do_sweep.start_srun_process") as mock_srun,
        ):
            orchestrator.preflight_check_ports()  # must return cleanly

        mock_srun.assert_not_called()
        # Verified live behavior of the on-disk non-SGLang backends: none of
        # them expose this method either.
        from srtctl.backends.mocker import MockerProtocol
        from srtctl.backends.trtllm import TRTLLMProtocol
        from srtctl.backends.vllm import VLLMProtocol

        for cls in (TRTLLMProtocol, VLLMProtocol, MockerProtocol):
            assert not hasattr(cls, "dp_attention_tcp_ports"), (
                f"{cls.__name__} must not stub dp_attention_tcp_ports until SGLang's "
                "DP-attention TCP-port behavior is verified for it"
            )

    def test_all_ports_free_logs_info_and_returns(self, caplog):
        config = _make_config_with_dp_attention()
        runtime = _make_runtime()
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        # Helper exits 0 with one PORT_OK line per checked port.
        ports_per_call: list[list[int]] = []

        def fake_srun(*, command, nodelist, **kw):
            # The 3rd positional arg of `command` after "python3" + script path is "--ports"
            assert command[2] == "--ports"
            ports = [int(p) for p in command[3:]]
            ports_per_call.append(ports)
            stdout = "\n".join(f"PORT_OK    127.0.0.1:{p}" for p in ports).encode()
            return _fake_popen(stdout, 0)

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", side_effect=fake_srun),
            caplog.at_level("INFO", logger="srtctl.cli.do_sweep"),
        ):
            orchestrator.preflight_check_ports()

        # One srun per prefill leader.
        prefill_leaders = [p for p in orchestrator.backend_processes if p.is_leader and p.endpoint_mode == "prefill"]
        assert len(ports_per_call) == len(prefill_leaders)

        # Each call must include all six SGLang DP-attention TCP ports.
        for ports, leader in zip(ports_per_call, prefill_leaders, strict=False):
            base = leader.http_port
            assert ports == [base, base + 233, base + 234, base + 235, base + 236, base + 237]

        # Per-job jitter is applied: --port should NOT be 30000 for job 144826.
        for leader in prefill_leaders:
            assert leader.http_port != 30000
            # rpc_port (the failing one) should not be 30236 anymore.
            assert leader.http_port + 236 != 30236

        # Confirm the user-facing success log line is present.
        assert any("preflight: ports" in r.message and "free on" in r.message for r in caplog.records)

    def test_busy_port_aborts_with_diagnostic(self, caplog):
        config = _make_config_with_dp_attention()
        runtime = _make_runtime()
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        # Simulate the offending listener: a python3 process owned by `jullin`
        # bound to the would-be rpc_port (matches the actual failing pattern).
        leaders = [p for p in orchestrator.backend_processes if p.is_leader and p.endpoint_mode == "prefill"]
        assert leaders, "test setup expects at least one prefill leader"
        rpc_port = leaders[0].http_port + 236
        busy_stdout = (
            f"PORT_OK    127.0.0.1:{leaders[0].http_port}\n"
            f"PORT_BUSY  127.0.0.1:{rpc_port}  pid=12345 uid=1234 user=jullin "
            f"name=python3 cmdline='python3 -m dynamo.sglang' state=R\n"
        ).encode()

        def fake_srun(**_):
            return _fake_popen(busy_stdout, 1)

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", side_effect=fake_srun),
            caplog.at_level("WARNING", logger="srtctl.cli.do_sweep"),
            pytest.raises(RuntimeError, match="port collision"),
        ):
            orchestrator.preflight_check_ports()

        # The PORT_BUSY diagnostic line must surface in the log so the user
        # can see pid/user/cmdline of the offending process.
        log_text = "\n".join(r.message for r in caplog.records)
        assert "PORT_BUSY" in log_text
        assert "pid=12345" in log_text
        assert "user=jullin" in log_text
        assert "dynamo.sglang" in log_text

    def test_helper_failure_with_no_output_still_aborts(self):
        """A helper crash (non-zero rc, empty stdout) must still fail-fast."""
        config = _make_config_with_dp_attention()
        runtime = _make_runtime()
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        def fake_srun(**_):
            return _fake_popen(b"", 2)

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", side_effect=fake_srun),
            pytest.raises(RuntimeError, match="rc=2"),
        ):
            orchestrator.preflight_check_ports()

    def test_runs_outside_pyxis(self):
        """Preflight srun must not request a container image (pyxis is slow)."""
        config = _make_config_with_dp_attention()
        runtime = _make_runtime()
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)

        captured_kwargs: list[dict] = []

        def fake_srun(*, command, nodelist, **kw):
            captured_kwargs.append({"command": command, "nodelist": nodelist, **kw})
            # Pretend everything is free.
            ports = [int(p) for p in command[3:]]
            stdout = "\n".join(f"PORT_OK    127.0.0.1:{p}" for p in ports).encode()
            return _fake_popen(stdout, 0)

        with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=fake_srun):
            orchestrator.preflight_check_ports()

        assert captured_kwargs, "preflight should have invoked srun at least once"
        for kw in captured_kwargs:
            assert kw["container_image"] is None, "preflight must not pyxis-mount"
            assert kw["container_mounts"] is None
            assert kw["use_bash_wrapper"] is False
            # 1 minute time budget per the plan.
            assert kw["srun_options"] == {"time": "1:00"}
