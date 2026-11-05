# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA NIC traffic capture: the sampler on fake sysfs trees, its config, launch and head-node merge."""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.mixins.postprocess_stage import _has_power_samples
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core import nic_traffic
from srtctl.core.nic_traffic import SAMPLES_HEADER, discover_ports
from srtctl.core.nic_traffic_session import NicTrafficSession, NicTrafficSessionSettings
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep


def _rdma_device(
    sysfs: Path, name: str, *, driver: str, netdev: str, counters: dict[str, int], port: str = "1"
) -> Path:
    """``/sys/class/infiniband/<name>`` with a driver symlink, a netdev and port counter files."""
    driver_dir = sysfs / "bus" / "pci" / "drivers" / driver
    driver_dir.mkdir(parents=True, exist_ok=True)
    device_dir = sysfs / "class" / "infiniband" / name
    (device_dir / "device" / "net" / netdev).mkdir(parents=True)
    (device_dir / "device" / "driver").symlink_to(driver_dir)
    port_dir = device_dir / "ports" / port
    for relative, value in counters.items():
        path = port_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{value}\n")
    return port_dir


def test_ionic_reads_unicast_hw_counters_as_bytes(tmp_path: Path) -> None:
    _rdma_device(
        tmp_path,
        "rdma0",
        driver="ionic",
        netdev="enp1s0",
        counters={
            "hw_counters/tx_rdma_ucast_bytes": 490_015_000_000,
            "hw_counters/rx_rdma_ucast_bytes": 7,
            "hw_counters/tx_rdma_mcast_bytes": 111,
            "hw_counters/tx_rdma_retx_bytes": 222,
            "hw_counters/tx_rdma_ccl_cts_bytes": 333,
        },
    )

    ports, unsupported = discover_ports(tmp_path)

    assert unsupported == []
    [port] = ports
    assert (port.device, port.port, port.netdev, port.driver) == ("rdma0", "1", "enp1s0", "ionic")
    assert port.read_bytes() == (490_015_000_000, 7)
    assert port.describe()["counter_source"] == "hw_counters/rdma_ucast_bytes"


def test_standard_driver_reads_port_data_in_four_octet_units(tmp_path: Path) -> None:
    _rdma_device(
        tmp_path,
        "mlx5_0",
        driver="mlx5_core",
        netdev="ib0",
        counters={"counters/port_xmit_data": 250, "counters/port_rcv_data": 100},
    )

    [port], unsupported = discover_ports(tmp_path)

    assert unsupported == []
    assert port.driver == "mlx5_core"
    assert port.read_bytes() == (1000, 400)
    assert port.describe()["bytes_per_unit"] == 4


def test_port_without_its_drivers_counters_is_unsupported_not_sampled(tmp_path: Path) -> None:
    _rdma_device(
        tmp_path,
        "bnxt_re0",
        driver="bnxt_en",
        netdev="eth0",
        counters={"hw_counters/tx_bytes": 5, "hw_counters/rx_bytes": 5},
    )
    _rdma_device(
        tmp_path,
        "mlx5_1",
        driver="mlx5_core",
        netdev="ib1",
        counters={"counters/port_xmit_data": 1, "counters/port_rcv_data": 1},
    )

    ports, unsupported = discover_ports(tmp_path)

    assert [port.device for port in ports] == ["mlx5_1"]
    [entry] = unsupported
    assert (entry.device, entry.port, entry.driver) == ("bnxt_re0", "1", "bnxt_en")
    assert "counters/port_xmit_data" in entry.reason


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def _run_standalone(sysfs: Path, out: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    # -I -S: no site-packages, so srtctl is not importable and the file must run on the stdlib alone.
    return subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(nic_traffic.__file__)),
            "--output-dir",
            str(out),
            "--interval-seconds",
            "0.05",
            "--sysfs-root",
            str(sysfs),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_standalone_file_samples_raw_counters_until_sigterm(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    port_dir = _rdma_device(
        sysfs,
        "rdma0",
        driver="ionic",
        netdev="enp1s0",
        counters={"hw_counters/tx_rdma_ucast_bytes": 100, "hw_counters/rx_rdma_ucast_bytes": 200},
    )
    out = tmp_path / "out"
    proc = _run_standalone(sysfs, out, {"SLURMD_NODENAME": "node-a"})
    try:
        _wait_for(out / "node-a.ready.json")
        (port_dir / "hw_counters/tx_rdma_ucast_bytes").write_text("900\n")
        deadline = time.monotonic() + 10
        while "900" not in (out / "node-a.csv").read_text():
            assert time.monotonic() < deadline, "the sampler never re-read the counter"
            time.sleep(0.02)
    finally:
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, stderr
    with (out / "node-a.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == SAMPLES_HEADER
    assert rows[1][2:] == ["node-a", "rdma0", "1", "enp1s0", "ionic", "100", "200"]
    assert rows[-1][2:] == ["node-a", "rdma0", "1", "enp1s0", "ionic", "900", "200"]
    timestamps = [float(row[0]) for row in rows[1:]]
    assert timestamps == sorted(timestamps)
    assert [int(row[1]) for row in rows[1:]] == list(range(len(rows) - 1))
    metadata = json.loads((out / "node-a.metadata.json").read_text())
    assert metadata["sample_row_count"] == len(rows) - 1
    assert [device["device"] for device in metadata["devices"]] == ["rdma0"]


class _SlowCounterClock:
    """Stands in for the sampler's ``time`` module; every counter read advances it by 0.1 s."""

    def __init__(self, passes: int) -> None:
        self.now = 1000.0
        self.passes_left = passes
        self.signal_handlers = {}

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, _seconds: float) -> None:
        self.passes_left -= 1
        if self.passes_left == 0:
            self.signal_handlers[signal.SIGTERM](signal.SIGTERM, None)


def test_each_row_is_stamped_at_its_own_port_reads_and_shares_the_pass_seq(tmp_path: Path, monkeypatch) -> None:
    sysfs = tmp_path / "sys"
    for name in ("rdma0", "rdma1"):
        _rdma_device(
            sysfs,
            name,
            driver="ionic",
            netdev=f"net-{name}",
            counters={"hw_counters/tx_rdma_ucast_bytes": 0, "hw_counters/rx_rdma_ucast_bytes": 0},
        )
    clock = _SlowCounterClock(passes=2)
    read_counter = nic_traffic._read_counter

    def slow_read(path: Path) -> int:
        clock.now += 0.1
        return read_counter(path)

    monkeypatch.setattr(nic_traffic, "time", clock)
    monkeypatch.setattr(nic_traffic, "_read_counter", slow_read)
    monkeypatch.setattr(nic_traffic.signal, "signal", clock.signal_handlers.__setitem__)
    monkeypatch.setenv("SLURMD_NODENAME", "node-a")
    out = tmp_path / "out"

    nic_traffic.collect(output_dir=out, ready_dir=out, interval_seconds=1.0, sysfs_root=sysfs)

    with (out / "node-a.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    assert [(row[1], row[3]) for row in rows] == [("0", "rdma0"), ("0", "rdma1"), ("1", "rdma0"), ("1", "rdma1")]
    # Each port's tx and rx reads take 0.2 s, so rdma1 is read 0.2 s after rdma0 in the same pass;
    # each row carries the midpoint of its own two reads.
    ready_at = json.loads((out / "node-a.ready.json").read_text())["ready_at_unix"]
    assert [round(float(row[0]) - ready_at, 6) for row in rows] == [0.1, 0.3, 0.5, 0.7]


def test_node_without_counters_writes_an_error_marker_and_exits_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _run_standalone(tmp_path / "sys", out, {"SLURMD_NODENAME": "node-b"})
    _, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, stderr
    assert "no RDMA port" in json.loads((out / "node-b.error.json").read_text())["error"]
    assert not (out / "node-b.csv").exists()
    assert not (out / "node-b.ready.json").exists()


def _recipe(tmp_path: Path, roles: dict, nic_traffic: dict | None = None) -> SrtConfig:
    recipe = {
        "schema": 2,
        "name": "nic",
        "model": {"path": "/models/m", "container": "image.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "roles": roles,
        "benchmark": {"type": "manual"},
    }
    if nic_traffic is not None:
        recipe["observability"] = {"nic_traffic": nic_traffic}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return SrtConfig.from_yaml(path)


SINGLE_NODE = {"agg": {"nodes": 1, "workers": 1}}
TWO_NODES = {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}}


@pytest.mark.parametrize(
    ("roles", "nic_traffic", "expected"),
    [
        (SINGLE_NODE, None, False),
        (TWO_NODES, None, True),
        (SINGLE_NODE, {"enabled": True}, True),
        (TWO_NODES, {"enabled": False}, False),
    ],
    ids=["single-node-default-off", "multi-node-default-on", "explicit-true", "explicit-false"],
)
def test_capture_defaults_on_only_for_multi_node_jobs(tmp_path: Path, roles, nic_traffic, expected) -> None:
    assert _recipe(tmp_path, roles, nic_traffic).nic_traffic_enabled is expected


@pytest.mark.parametrize(
    ("nic_traffic", "message"),
    [
        ({"storage_subdir": "power/nic"}, "overlaps telemetry.storage_subdir"),
        ({"storage_subdir": "cpu_power"}, "overlaps telemetry.cpu_power.storage_subdir"),
        ({"storage_subdir": "tachometer"}, "overlaps observability.tachometer.storage_subdir"),
        ({"storage_subdir": "../nic"}, "safe relative path"),
        ({"sample_interval_seconds": 0}, "finite and positive"),
        ({"sample_interval_seconds": 5}, "exceeds"),
    ],
)
def test_invalid_capture_settings_are_rejected_at_load(tmp_path: Path, nic_traffic: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _recipe(tmp_path, TWO_NODES, nic_traffic)


class _FakeProcess:
    def __init__(self, name: str, running: bool = True) -> None:
        self.name = name
        self.is_running = running
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.is_running = False


def _write_node_csv(nodes_dir: Path, hostname: str, rows: list[tuple[float, str, int, int]]) -> None:
    with (nodes_dir / f"{hostname}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SAMPLES_HEADER)
        for seq, (timestamp, device, tx, rx) in enumerate(rows):
            writer.writerow((repr(timestamp), seq, hostname, device, "1", "enp1s0", "ionic", tx, rx))


def _session(tmp_path: Path, nodes: tuple[str, ...]) -> NicTrafficSession:
    session = NicTrafficSession(
        NicTrafficSessionSettings(
            nic_dir=tmp_path / "nic_traffic", job_id="7", run_name="run", nodes=nodes, sample_interval_seconds=1.0
        )
    )
    session.initialize()
    return session


def test_finalize_merges_every_node_in_time_order(tmp_path: Path) -> None:
    session = _session(tmp_path, ("node-a", "node-b"))
    sampler = _FakeProcess("telemetry_nic_traffic")
    session.add_process(sampler)  # type: ignore[arg-type]
    _write_node_csv(session.samples_dir, "node-a", [(2.0, "rdma0", 20, 0), (3.0, "rdma0", 30, 0)])
    _write_node_csv(session.samples_dir, "node-b", [(2.5, "rdma1", 25, 0)])

    outcome = session.stop_and_finalize()

    assert sampler.terminated
    assert outcome.status == "complete"
    assert outcome.reason_codes == ()
    with session.traffic_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == SAMPLES_HEADER
    assert [(row[0], row[2]) for row in rows[1:]] == [("2.0", "node-a"), ("2.5", "node-b"), ("3.0", "node-a")]
    manifest = json.loads(session.manifest_path.read_text())
    assert manifest["observed_nodes"] == ["node-a", "node-b"]
    assert manifest["sample_row_count"] == 3


def test_manifest_names_missing_and_counterless_nodes(tmp_path: Path) -> None:
    session = _session(tmp_path, ("node-a", "node-b", "node-c"))
    session.add_process(_FakeProcess("telemetry_nic_traffic", running=False))  # type: ignore[arg-type]
    _write_node_csv(session.samples_dir, "node-a", [(1.0, "rdma0", 1, 1)])
    (session.samples_dir / "node-a.metadata.json").write_text(
        json.dumps({"devices": [{"device": "rdma0", "driver": "ionic"}], "unsupported": []})
    )
    (session.ready_dir / "node-b.error.json").write_text(json.dumps({"error": "no RDMA port"}))

    outcome = session.stop_and_finalize(interrupted=True)

    assert outcome.status == "incomplete"
    assert outcome.reason_codes == (
        "interrupted",
        "nic_counters_unavailable",
        "nic_node_samples_missing",
        "nic_sampler_exited",
    )
    manifest = json.loads(session.manifest_path.read_text())
    assert manifest["expected_nodes"] == ["node-a", "node-b", "node-c"]
    assert manifest["observed_nodes"] == ["node-a"]
    assert manifest["missing_nodes"] == ["node-b", "node-c"]
    assert manifest["node_errors"] == {"node-b": "no RDMA port"}
    assert manifest["nodes"]["node-a"]["devices"] == [{"device": "rdma0", "driver": "ionic"}]
    assert manifest["exited_processes"] == ["telemetry_nic_traffic"]


def test_capture_never_writes_files_power_consumers_claim(tmp_path: Path) -> None:
    session = _session(tmp_path, ("node-a",))
    _write_node_csv(session.samples_dir, "node-a", [(1.0, "rdma0", 1, 1)])
    session.stop_and_finalize()

    names = {path.name for path in tmp_path.rglob("*")}
    assert {"nic_traffic.csv", "nic_manifest.json"} <= names
    assert not names & {"samples.csv", "manifest.json"}
    assert not _has_power_samples(tmp_path)


class _Stage(TelemetryStageMixin):
    def __init__(self, tmp_path: Path, config: SrtConfig, *, het_groups: dict[str, int] | None = None) -> None:
        self.config = config
        self.runtime = MagicMock()
        self.runtime.log_dir = tmp_path
        self.runtime.job_id = "7"
        self.runtime.run_name = "run"
        self.runtime.srun_options = {}
        self.runtime.nodes.compute = ("node-a", "node-b")
        self.runtime.nodes.het = het_groups is not None
        self.runtime.nodes.het_group_for.side_effect = lambda node: (het_groups or {}).get(node)

    @property
    def backend_processes(self):
        return []


@patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
def test_multi_node_job_launches_one_bare_host_sampler_per_node(mock_srun, tmp_path: Path) -> None:
    stage = _Stage(tmp_path, _recipe(tmp_path, TWO_NODES))
    registry = ProcessRegistry(job_id="7")

    session = stage.start_nic_traffic(registry)

    assert session is not None
    kwargs = mock_srun.call_args.kwargs
    assert mock_srun.call_count == 1
    assert kwargs["nodelist"] == ["node-a", "node-b"]
    assert (kwargs["nodes"], kwargs["ntasks"]) == (2, 2)
    assert kwargs["use_bash_wrapper"] is False
    assert "container_image" not in kwargs
    assert kwargs["step_name"] == "telemetry_nic_traffic"
    assert kwargs["command"][1:3] == ["-m", "srtctl.core.nic_traffic"]
    assert kwargs["command"][kwargs["command"].index("--interval-seconds") + 1] == "1.0"
    [process] = registry.get_all_processes().values()
    assert process.critical is False
    assert process.step_name == "telemetry_nic_traffic"
    assert json.loads((tmp_path / "nic_traffic" / "nic_manifest.json").read_text())["expected_nodes"] == [
        "node-a",
        "node-b",
    ]


@patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
def test_het_job_launches_one_sampler_step_per_component(mock_srun, tmp_path: Path) -> None:
    stage = _Stage(tmp_path, _recipe(tmp_path, TWO_NODES), het_groups={"node-a": 0, "node-b": 1})

    stage.start_nic_traffic(ProcessRegistry(job_id="7"))

    launches = [
        (call.kwargs["het_group"], call.kwargs["nodelist"], call.kwargs["step_name"])
        for call in mock_srun.call_args_list
    ]
    assert launches == [(0, ["node-a"], "telemetry_nic_traffic_g0"), (1, ["node-b"], "telemetry_nic_traffic_g1")]


@patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
def test_disabled_capture_launches_nothing(mock_srun, tmp_path: Path) -> None:
    stage = _Stage(tmp_path, _recipe(tmp_path, TWO_NODES, {"enabled": False}))

    assert stage.start_nic_traffic(ProcessRegistry(job_id="7")) is None
    stage.finalize_nic_traffic()
    mock_srun.assert_not_called()
    assert not (tmp_path / "nic_traffic").exists()


@patch("srtctl.cli.mixins.telemetry_stage.start_srun_process", side_effect=OSError("srun: not found"))
def test_launch_failure_is_absorbed(_mock_srun, tmp_path: Path) -> None:
    stage = _Stage(tmp_path, _recipe(tmp_path, TWO_NODES))

    session = stage.start_nic_traffic(ProcessRegistry(job_id="7"))
    stage.finalize_nic_traffic()

    assert session is not None
    manifest = json.loads(session.manifest_path.read_text())
    assert manifest["missing_nodes"] == ["node-a", "node-b"]


def test_node_discovery_failure_is_absorbed(tmp_path: Path) -> None:
    stage = _Stage(tmp_path, _recipe(tmp_path, TWO_NODES))
    stage._telemetry_nodes = MagicMock(side_effect=ValueError("bad allocation"))  # type: ignore[method-assign]

    assert stage.start_nic_traffic(ProcessRegistry(job_id="7")) is None
    stage.finalize_nic_traffic()


def test_mock_sweep_starts_and_finalizes_the_capture(tmp_path: Path) -> None:
    recipe = {
        "schema": 2,
        "name": "nic-mock",
        "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "roles": TWO_NODES,
        "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
        "observability": {"tachometer": {"enabled": False}},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    output_dir = tmp_path / "output"
    launches: list[dict] = []

    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=output_dir,
        job_id="7",
        options=MockOptions(
            child_duration_s=0, phase_pause_s=0, nodelist=("node-01", "node-02"), on_srun=launches.append
        ),
    )

    assert exit_code == 0
    assert [call["step_name"] for call in launches if "srtctl.core.nic_traffic" in call["command"]] == [
        "telemetry_nic_traffic"
    ]
    manifest = json.loads((output_dir / "logs" / "nic_traffic" / "nic_manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert len(manifest["expected_nodes"]) == 2
    assert "nic_node_samples_missing" in manifest["reason_codes"]
