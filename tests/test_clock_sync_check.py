"""Tests for the pre-server NTP clock-sync gate that guards DCGM power runs.

Sample timestamps (orchestrator host) and window boundaries (benchmark client
host) are compared by raw float, so the orchestrator probes every allocation
node's bare host for a synchronised clock before spending GPU time.
"""

import re
import subprocess
import sys
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
            script = call.kwargs["command"][2]
            # Every probe branch must leave evidence in the .out, not just an exit code.
            for daemon in ("timedatectl", "chronyc", "ntpq"):
                assert daemon in script
                assert f'echo "$(hostname): {daemon}' in script

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
        assert orch._clock_sync_failures == ["node0", "node1", "node2"]

    def test_passing_probe_leaves_no_failures(self, tmp_path):
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power(required=False)), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_proc(0)):
            orch._check_clock_sync()

        assert orch._clock_sync_failures == []

    def test_each_passing_node_logs_which_daemon_vouched(self, tmp_path, caplog):
        """The per-node .out carries the probe's evidence; the sweep log should name it, not just say OK."""
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power()), runtime=_runtime(tmp_path))

        def srun(*, nodelist, output, **_):
            # Emulate what the real probe prints for each branch.
            node = nodelist[0]
            evidence = {
                "node0": f"{node}: timedatectl NTPSynchronized=yes\n",
                "node1": f"{node}: chronyc Leap status Normal\nSystem time     : 0.000012 seconds fast of NTP time\n",
                "node2": "",  # a probe that passed silently still gets a bare OK line
            }[node]
            Path(output).write_text(evidence)
            return _proc(0)

        with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=srun), caplog.at_level("INFO"):
            orch._check_clock_sync()

        assert "clock_sync_check: node0 OK (timedatectl NTPSynchronized=yes)" in caplog.text
        assert "clock_sync_check: node1 OK (chronyc Leap status Normal)" in caplog.text
        assert "clock_sync_check: node2 OK\n" in caplog.text or "clock_sync_check: node2 OK" in caplog.text
        assert "all 3 node(s) report NTP-synchronised clocks" in caplog.text

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("node7: timedatectl NTPSynchronized=yes\n", " (timedatectl NTPSynchronized=yes)"),
            ("node7: chronyc Leap status Normal\nLast offset : +0.000003 seconds\n", " (chronyc Leap status Normal)"),
            ("no-hostname-prefix\n", " (no-hostname-prefix)"),
            ("", ""),
        ],
    )
    def test_evidence_is_the_first_line_without_the_hostname_prefix(self, tmp_path, content, expected):
        log = tmp_path / "clock_sync_node7.out"
        log.write_text(content)
        assert SweepOrchestrator._clock_sync_evidence(log) == expected

    def test_missing_evidence_file_is_tolerated(self, tmp_path):
        assert SweepOrchestrator._clock_sync_evidence(tmp_path / "absent.out") == ""

    def test_best_effort_failures_reach_the_power_session(self, tmp_path):
        """The warning alone is not enough: the manifest must carry the unverified-clock reason."""
        orch = SweepOrchestrator(config=_config(telemetry=_dcgm_power(required=False)), runtime=_runtime(tmp_path))
        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_proc(1)):
            orch._check_clock_sync()

        session = MagicMock()
        with (
            patch("srtctl.cli.mixins.telemetry_stage.PowerTelemetrySession", return_value=session),
            patch("srtctl.cli.mixins.telemetry_stage.build_expected_devices", return_value=[]),
            patch.object(type(orch), "backend_processes", new_callable=lambda: property(lambda self: [])),
            patch.object(orch, "_telemetry_nodes", return_value=["node1", "node2"]),
            patch.object(orch, "_start_exporter_container", return_value=[]),
        ):
            orch.start_power_telemetry(MagicMock())

        session.record_clock_sync_failures.assert_called_once_with(["node0", "node1", "node2"])

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


class TestProbeScript:
    """Run CLOCK_SYNC_SCRIPT under real bash with fake probe tools on PATH.

    The script is a quoted one-liner built from Python string pieces; only
    executing it proves the quoting, here-strings, and branch order are right.
    """

    CHRONY_NORMAL = (
        "Reference ID    : 0A000001 (10.0.0.1)\nStratum         : 3\n"
        "System time     : 0.000012345 seconds fast of NTP time\nLast offset     : +0.000003210 seconds\n"
        "Leap status     : Normal\n"
    )
    NTPQ_PEER = (
        "     remote           refid      st t when poll reach   delay   offset  jitter\n"
        "*10.0.0.1        .GPS.            1 u   12   64  377    0.123    0.004   0.010\n"
    )

    @staticmethod
    def _run(tmp_path: Path, tools: dict[str, str]) -> subprocess.CompletedProcess[str]:
        bindir = tmp_path / "bin"
        bindir.mkdir()
        # The kernel branch would otherwise consult the test host's real clock state.
        tools = {"python3": "echo 'fake python3: no adjtimex here' >&2; exit 1\n", **tools}
        for name, body in tools.items():
            exe = bindir / name
            exe.write_text("#!/usr/bin/env bash\n" + body)
            exe.chmod(0o755)
        env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)}
        return subprocess.run(
            ["bash", "-c", SweepOrchestrator.CLOCK_SYNC_SCRIPT], capture_output=True, text=True, env=env, check=False
        )

    def test_timedatectl_wins_first(self, tmp_path):
        r = self._run(tmp_path, {"timedatectl": "echo yes\n", "chronyc": "exit 1\n", "ntpq": "exit 1\n"})
        assert r.returncode == 0
        assert r.stdout.splitlines()[0].endswith(": timedatectl NTPSynchronized=yes")

    def test_chronyc_evidence_includes_the_offset(self, tmp_path):
        r = self._run(
            tmp_path,
            {"timedatectl": "echo no\n", "chronyc": f"cat <<'EOF'\n{self.CHRONY_NORMAL}EOF\n", "ntpq": "exit 1\n"},
        )
        assert r.returncode == 0
        lines = r.stdout.splitlines()
        assert lines[0].endswith(": chronyc Leap status Normal")
        assert any(line.startswith("System time") for line in lines)
        assert any(line.startswith("Last offset") for line in lines)

    def test_ntpq_selected_peer_is_the_last_resort(self, tmp_path):
        r = self._run(
            tmp_path,
            {
                "timedatectl": "exit 1\n",
                "chronyc": "echo '506 Cannot talk to daemon' >&2; exit 1\n",
                "ntpq": f"cat <<'EOF'\n{self.NTPQ_PEER}EOF\n",
            },
        )
        assert r.returncode == 0
        lines = r.stdout.splitlines()
        assert lines[0].endswith(": ntpq has a selected peer")
        assert lines[1].startswith("*10.0.0.1")

    def test_kernel_flag_vouches_when_no_daemon_answers(self, tmp_path):
        """Hosts without systemd/D-Bus or a chrony/ntp CLI (container-rooted Slurm nodes) still prove sync."""
        r = self._run(
            tmp_path,
            {
                "timedatectl": "echo 'Failed to connect to bus: Host is down' >&2; exit 1\n",
                "chronyc": "exit 127\n",
                "ntpq": "exit 127\n",
                "python3": "echo 'maxerror 4508us status 0x2001'\n",
            },
        )
        assert r.returncode == 0
        assert r.stdout.splitlines() == [
            f"{r.stdout.split(':', 1)[0]}: kernel adjtimex STA_UNSYNC clear (maxerror 4508us status 0x2001)"
        ]

    def test_failure_report_keeps_every_probe_stderr(self, tmp_path):
        """clock_sync_<node>.out must say why nothing vouched, not just that nothing did."""
        r = self._run(
            tmp_path,
            {
                "timedatectl": "echo 'Failed to connect to bus: Host is down' >&2; exit 1\n",
                "ntpq": "exit 1\n",
                "python3": "echo 'kernel clock unsynchronised: state 5 status 0x41 maxerror 16000000us' >&2; exit 1\n",
            },
        )
        assert r.returncode == 1
        assert r.stdout == ""
        assert "timedatectl: Failed to connect to bus: Host is down" in r.stderr
        assert re.search(r"^chronyc: .*chronyc: command not found", r.stderr, re.MULTILINE)
        assert "adjtimex: kernel clock unsynchronised: state 5 status 0x41 maxerror 16000000us" in r.stderr

    @pytest.mark.parametrize(
        "tools",
        [
            {"timedatectl": "exit 127\n", "chronyc": "exit 127\n", "ntpq": "exit 127\n"},
            {"timedatectl": "echo no\n", "chronyc": "echo 'Leap status     : Not synchronised'\n", "ntpq": "exit 1\n"},
            {"timedatectl": "echo no\n", "chronyc": "exit 1\n", "ntpq": "printf '+10.0.0.2 x 2 u 1 64 377 0 0 0\\n'\n"},
        ],
        ids=["no-tools", "chrony-unsynced", "ntpq-no-selected-peer"],
    )
    def test_no_proof_exits_nonzero_with_no_stdout(self, tmp_path, tools):
        r = self._run(tmp_path, tools)
        assert r.returncode == 1
        assert r.stdout == ""
        assert "not NTP-synchronised" in r.stderr


class TestAdjtimexProbe:
    """The python one-liner the kernel branch runs on the bare host."""

    def test_embeds_in_the_bash_single_quotes(self):
        probe = SweepOrchestrator.CLOCK_SYNC_ADJTIMEX_PROBE
        assert "'" not in probe
        compile(probe, "<probe>", "exec")
        assert f"python3 -c '{probe}'" in SweepOrchestrator.CLOCK_SYNC_SCRIPT

    @pytest.mark.skipif(sys.platform != "linux", reason="adjtimex(2) is Linux-only")
    def test_reads_the_live_kernel_state(self):
        """Whatever this host's clock state is, the struct layout must yield a sane, parseable answer."""
        r = subprocess.run(
            [sys.executable, "-c", SweepOrchestrator.CLOCK_SYNC_ADJTIMEX_PROBE],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            assert re.fullmatch(r"maxerror \d+us status 0x[0-9a-f]+\n", r.stdout)
        else:
            assert r.returncode == 1
            assert re.fullmatch(
                r"kernel clock unsynchronised: state [0-5] status 0x[0-9a-f]+ maxerror \d+us\n", r.stderr
            ), r.stderr


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
