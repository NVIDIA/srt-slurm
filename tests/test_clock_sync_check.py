"""Tests for the pre-server NTP clock-sync gate that guards DCGM power runs.

Sample timestamps (orchestrator host) and window boundaries (benchmark client
host) are compared by raw float, so the orchestrator probes every allocation
node's bare host for a synchronised clock before spending GPU time.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import (
    BenchmarkConfig,
    CpuPowerExporterConfig,
    ModelConfig,
    PlacementConfig,
    ResourceConfig,
    RoleConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)


def _config(*, telemetry: TelemetryConfig | None = None) -> SrtConfig:
    return SrtConfig(
        name="clock-sync-test",
        model=ModelConfig(path="/models/test", container="test.sqsh", precision="fp8"),
        resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4),
        roles={"prefill": RoleConfig(nodes=1), "decode": RoleConfig(nodes=1)},
        benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], placement=PlacementConfig(node="last_decode")),
        telemetry=telemetry or TelemetryConfig(),
    )


def _dcgm_power(**overrides) -> TelemetryConfig:
    fields: dict = {
        "enabled": True,
        "dcgm_exporter": TelemetryExporterConfig(container_image="dcgm-exporter", port=9401),
    }
    fields.update(overrides)
    return TelemetryConfig(**fields)


def _runtime(tmp_path: Path, *, bench: str = "node0") -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head="node0", bench=bench, infra="node0", worker=("node1", "node2")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/test"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def _proc(returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.wait.return_value = returncode
    return proc


class TestSchema:
    def test_defaults_to_enabled(self):
        assert TelemetryConfig().clock_sync_check is True

    def test_can_be_disabled(self):
        assert _dcgm_power(clock_sync_check=False).clock_sync_check is False


class TestGate:
    def test_probes_every_node_on_the_bare_host(self, tmp_path):
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power()), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_proc(0)) as srun:
            orch._check_clock_sync()

        assert [c.kwargs["nodelist"] for c in srun.call_args_list] == [["node0"], ["node1"], ["node2"]]
        for call in srun.call_args_list:
            assert call.kwargs["container_image"] is None, "the time daemon lives on the host, not in the container"
            assert call.kwargs["command"][:2] == ["bash", "-c"]
            assert "timedatectl" in call.kwargs["command"][2]

    def test_includes_a_dedicated_bench_node_once(self, tmp_path):
        """A client node outside the worker set still feeds window timestamps and must be probed."""
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power()), runtime=_runtime(tmp_path, bench="node9"))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_proc(0)) as srun:
            orch._check_clock_sync()

        probed = [c.kwargs["nodelist"][0] for c in srun.call_args_list]
        assert probed == ["node0", "node9", "node1", "node2"]
        assert len(probed) == len(set(probed))

    def test_required_run_aborts_when_a_node_is_unsynced(self, tmp_path):
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power(required=True)), runtime=_runtime(tmp_path))
        procs = iter([_proc(0), _proc(1), _proc(0)])

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", side_effect=lambda **_: next(procs)),
            pytest.raises(RuntimeError, match="clock_sync_check: node1"),
        ):
            orch._check_clock_sync()

    def test_best_effort_run_only_warns(self, tmp_path, caplog):
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power(required=False)), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_proc(1)), caplog.at_level("WARNING"):
            orch._check_clock_sync()  # must not raise

        assert "telemetry.required is false" in caplog.text

    def test_a_hung_probe_counts_as_a_failure(self, tmp_path):
        hung = MagicMock()
        hung.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="bash", timeout=30),
            0,
        ]  # timed-out wait, then post-kill wait
        procs = iter([hung, _proc(0), _proc(0)])
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power(required=True)), runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", side_effect=lambda **_: next(procs)),
            pytest.raises(RuntimeError, match="clock_sync_check: node0"),
        ):
            orch._check_clock_sync()

        hung.kill.assert_called_once()


class TestSkipped:
    @pytest.mark.parametrize(
        "telemetry",
        [
            TelemetryConfig(),  # disabled
            _dcgm_power(clock_sync_check=False),
            TelemetryConfig(
                enabled=True, dcgm_exporter=None, cpu_power_exporter=CpuPowerExporterConfig(port=9405)
            ),  # CPU-power-only: no DCGM windows to align
        ],
        ids=["telemetry-disabled", "opted-out", "no-dcgm-exporter"],
    )
    def test_no_srun_when_not_applicable(self, tmp_path, telemetry):
        orch = SweepOrchestrator(config=_config(telemetry=telemetry), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process") as srun:
            orch._check_clock_sync()

        srun.assert_not_called()

    def test_eval_only_skips_the_gate(self, tmp_path, monkeypatch):
        """EVAL_ONLY runs have no benchmark, hence no windows to align."""
        monkeypatch.setenv("EVAL_ONLY", "true")
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power(required=True)), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process") as srun:
            orch._check_clock_sync()

        srun.assert_not_called()
