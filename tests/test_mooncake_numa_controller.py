# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise NUMA benchmark policy, evidence and local mailbox lifecycle."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

from tools import benchmark_mooncake_numa as benchmark


def inventory(host: str = "synthetic-node") -> dict:
    return {
        "host": host,
        "ip": "192.0.2.1",
        "infiniband": {
            nic: {"numa_node": numa, "ports": {"1": {"state": "4: ACTIVE", "link_layer": "InfiniBand"}}}
            for numa, nic in enumerate(("mlx5_0", "mlx5_6"))
        },
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("MC_NUMA_ROUNDS", "0"),
        ("MC_NUMA_ROUNDS", "11"),
        ("MC_NUMA_SECONDS", "nan"),
        ("MC_NUMA_SECONDS", "0"),
        ("MC_NUMA_WARMUP", "-1"),
        ("MC_NUMA_BATCH_SIZE", "0"),
        ("MC_NUMA_POOL_GIB", "0.001"),
        ("MC_NUMA_BUDGET", "-1"),
        ("MC_NUMA_BUDGET", "0"),
        ("MC_NUMA_BUDGET", "1321"),
    ],
)
def test_settings_reject_invalid_measurement_and_working_set_sizes(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setattr(benchmark.os, "environ", {name: value})
    with pytest.raises(ValueError):
        benchmark.Settings.from_env()


@pytest.mark.parametrize("cluster,allow_off,enabled", [("raplab", False, 12), ("hecate", True, 24)])
def test_plan_keeps_matched_pairs_and_explicit_affinity_off_policy(cluster: str, allow_off: bool, enabled: int) -> None:
    settings = benchmark.Settings(cluster=cluster, allow_cross_rail=allow_off)
    plan = benchmark.make_plan(settings)

    assert len(plan) == 24 and len({case["case_id"] for case in plan}) == 24
    assert sum(case["enabled"] for case in plan) == enabled
    for repeat in range(3):
        selected = [case for case in plan if case["round"] == repeat]
        assert {(case["donor_numa"], case["reader_numa"], case["affinity"]) for case in selected} == {
            (donor, reader, affinity) for donor in (0, 1) for reader in (0, 1) for affinity in (True, False)
        }
        assert all(case["enabled"] == (case["affinity"] or allow_off) for case in selected)
    assert plan == benchmark.make_plan(settings)


@pytest.mark.parametrize("fault", ["wrong_numa", "inactive", "ethernet", "override_wrong_group"])
def test_group_preflight_requires_common_active_numa_local_ib_ports(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    reader, donor = inventory(), inventory()
    monkeypatch.delenv("MC_NUMA_GROUP0", raising=False)
    monkeypatch.delenv("MC_NUMA_GROUP1", raising=False)
    assert benchmark.discover_groups(reader, donor) == {0: ["mlx5_0"], 1: ["mlx5_6"]}
    if fault == "wrong_numa":
        donor["infiniband"]["mlx5_0"]["numa_node"] = 1
    elif fault == "inactive":
        donor["infiniband"]["mlx5_0"]["ports"]["1"]["state"] = "2: INIT"
    elif fault == "ethernet":
        donor["infiniband"]["mlx5_0"]["ports"]["1"]["link_layer"] = "Ethernet"
    else:
        monkeypatch.setenv("MC_NUMA_GROUP0", "mlx5_6")
    with pytest.raises(ValueError, match="NUMA"):
        benchmark.discover_groups(reader, donor)


def counters(group0: int, group1: int) -> dict:
    return {
        "ib_counters": {
            f"{nic}/ports/1/counters/port_xmit_data": value for nic, value in (("mlx5_0", group0), ("mlx5_6", group1))
        }
    }


def test_nic_evidence_converts_four_byte_counter_units_and_verifies_group_share() -> None:
    result = benchmark.nic_evidence(counters(10, 20), counters(300010, 10020), {0: ["mlx5_0"], 1: ["mlx5_6"]}, 0)
    assert result["verified"]
    assert result["tx_bytes_by_nic"] == {"mlx5_0": 1_200_000, "mlx5_6": 40_000}
    assert result["expected_group_share"] == pytest.approx(30 / 31)
    assert "whole reader process" in result["scope"]


@pytest.mark.parametrize("fault", ["missing", "reset", "wrong_group", "no_traffic"])
def test_bad_or_missing_nic_evidence_is_never_verified(fault: str) -> None:
    before, after = counters(10, 20), counters(600010, 20)
    if fault == "missing":
        del after["ib_counters"]["mlx5_6/ports/1/counters/port_xmit_data"]
    elif fault == "reset":
        after = counters(1, 20)
    elif fault == "wrong_group":
        after = counters(10, 600020)
    else:
        after = counters(10, 20)
    assert not benchmark.nic_evidence(before, after, {0: ["mlx5_0"], 1: ["mlx5_6"]}, 0)["verified"]


@pytest.mark.parametrize(
    "message",
    [
        "W1019 05:57:18 rdma_transport.cpp] Worker: Process failed for slice 0xffff, retry_cnt:0",
        "W1019 05:57:19 rdma_transport.cpp] Worker: Cannot make connection for endpoint 172.29.110.5:16144",
        "E1019 05:57:20 rdma_transport.cpp] transport retry counter exceeded",
    ],
)
def test_native_transport_failures_invalidate_even_an_initial_retry(message: str) -> None:
    assert benchmark.transport_errors("ordinary setup message\n" + message + "\n") == [message]
    assert benchmark.transport_errors("Created RDMA context\ninitial retry_cnt:0\n") == []


def measured_rows() -> list[dict]:
    return [
        {
            **case,
            "status": "PASS",
            "goodput_gb_s": 10 if case["donor_numa"] == case["reader_numa"] else 5,
            "nic_evidence": {"verified": True},
            "placement_verified": True,
        }
        for case in benchmark.make_plan(benchmark.Settings(rounds=1, allow_cross_rail=True))
    ]


def test_matched_comparisons_preserve_direction_and_do_not_zero_fill() -> None:
    rows = measured_rows()
    ratios = benchmark.comparisons(rows)
    assert all(row["status"] == "VALID" for row in ratios)
    assert [row["bandwidth_ratio"] for row in ratios if row["comparison"].startswith("remote-memory")] == [0.5, 0.5]
    for mode in ("failed", "missing", "counter", "nonfinite", "placement", "missing_placement"):
        altered = [dict(row) for row in rows]
        index = next(
            i
            for i, row in enumerate(altered)
            if (row["donor_numa"], row["reader_numa"], row["affinity"]) == (1, 0, True)
        )
        if mode == "missing":
            altered.pop(index)
        elif mode == "failed":
            altered[index]["status"] = "FAIL"
        elif mode == "counter":
            altered[index]["nic_evidence"] = {"verified": False}
        elif mode == "placement":
            altered[index]["placement_verified"] = False
        elif mode == "missing_placement":
            del altered[index]["placement_verified"]
        else:
            altered[index]["goodput_gb_s"] = float("nan")
        invalid = [row for row in benchmark.comparisons(altered) if row["reader_numa"] == 0]
        assert len(invalid) == 2
        assert all(row["status"] == "INCONCLUSIVE" and row["bandwidth_ratio"] is None for row in invalid)


@pytest.mark.parametrize(
    "cluster,allow_off,problem",
    [
        ("raplab", False, ""),
        ("hecate", True, ""),
        ("hecate", True, "bad_read"),
        ("hecate", True, "wrong_group"),
        ("hecate", True, "transport"),
        ("hecate", True, "late_placement_failure"),
    ],
)
def test_full_controller_writes_results_and_reports_using_real_json_mailboxes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cluster: str, allow_off: bool, problem: str
) -> None:
    settings = benchmark.Settings(cluster=cluster, allow_cross_rail=allow_off, rounds=1)
    controller = benchmark.Controller(tmp_path, settings)
    benchmark.atomic_json(tmp_path / "reader-ready.json", {"status": "ready", "inventory": inventory("reader")})
    monkeypatch.setattr(benchmark, "collect_inventory", lambda *args: inventory("donor"))
    monkeypatch.setattr(controller, "start_master", lambda ip: "192.0.2.1:8700")
    original_request = controller.request
    reader_configs = []
    totals = [0, 0]
    verification_counts = {}

    def request(action, **fields):
        identifier, path = original_request(action, **fields)
        benchmark.atomic_json(path, {"status": "pong" if action == "ping" else "stopped"})
        return identifier, path

    class Donor:
        returncode = None

        def __init__(self, config):
            self.config = config

        def wait(self, timeout=None):
            benchmark.atomic_json(Path(self.config["output"]), {"status": "PASS"})
            self.returncode = 0
            return 0

    def launch(argv, log_path):
        config = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
        benchmark.atomic_json(Path(config["donor_ready"]), {"status": "PASS", "endpoint": "donor:1234"})
        child = Donor(config)
        controller.children.append(child)
        return child

    def execute(config, role, label):
        error = (
            "Worker: Process failed for slice 0xffff, retry_cnt:0\n"
            if problem == "transport" and label == "reader"
            else ""
        )
        Path(config["output"]).parent.joinpath(f"{label}.native.log").write_text(error)
        if role == "seed":
            return {"status": "PASS", "keys": ["key0", "key1"]}, {"returncode": 0}
        if config["trace"]:
            return {"status": "PASS", "trace": True, "count": 2}, {"returncode": 0}
        reader_configs.append(config)
        expected = config["reader_numa"] if config["affinity"] else config["donor_numa"]
        totals[1 - expected if problem == "wrong_group" else expected] += 600_000
        status = "FAIL" if problem == "bad_read" and config["reader_numa"] == 0 else "PASS"
        return {
            "status": status,
            "goodput_gb_s": 12,
            "latency_ms": {"p50": 1, "p95": 2, "p99": 3},
            "error": "synthetic bad read" if status == "FAIL" else "",
        }, {"returncode": int(status == "FAIL")}

    monkeypatch.setattr(controller, "request", request)
    monkeypatch.setattr(controller, "launch", launch)
    monkeypatch.setattr(controller, "execute_reader", execute)

    def donor_request(config, action, keys=None):
        if action == "verify":
            key = config["case_id"]
            verification_counts[key] = verification_counts.get(key, 0) + 1
            if problem == "late_placement_failure" and verification_counts[key] == 2:
                raise RuntimeError("synthetic final donor NUMA residency mismatch")
        return {"status": "PASS", "health": counters(*totals)}

    monkeypatch.setattr(controller, "donor_request", donor_request)
    monkeypatch.setattr(benchmark, "stop_process", lambda child: child.wait() == 0)

    rc = controller.run()

    rows = [json.loads(line) for line in (tmp_path / "logs/results.jsonl").read_text().splitlines()]
    assert len(rows) == (4 if problem == "late_placement_failure" or not allow_off else 8)
    if problem == "late_placement_failure":
        assert all(not row.get("placement_verified") for row in rows)
    else:
        assert all(row.get("placement_verified") for row in rows)
    assert len(list(csv.DictReader((tmp_path / "logs/summary.csv").open()))) == len(rows)
    assert (tmp_path / "logs/perf-analysis.md").is_file()
    assert json.loads((tmp_path / "logs/mailbox-check.json").read_text())["round_trip_s"] >= 0
    for numa in (0, 1):
        selected = [config for config in reader_configs if config["reader_numa"] == numa]
        assert {tuple(config["nics"]) for config in selected} == {("mlx5_0",) if numa == 0 else ("mlx5_6",)}
    ratios = json.loads((tmp_path / "logs/comparisons.json").read_text())
    if problem:
        assert any(row["status"] == "INCONCLUSIVE" and row["bandwidth_ratio"] is None for row in ratios)
    elif allow_off:
        assert all(row["status"] == "VALID" for row in ratios)
    if problem == "late_placement_failure":
        assert rc == 1 and all(row["status"] == "INCONCLUSIVE" for row in ratios)
    elif problem in ("bad_read", "transport"):
        assert rc == 1 and any(row["status"] == "FAIL" for row in rows)
    elif not problem:
        assert rc == 0 and all(row["status"] == "PASS" for row in rows)


@pytest.mark.parametrize("timeout", [False, True])
def test_agent_real_filesystem_protocol_reaps_child_before_completion_and_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: bool
) -> None:
    commands, replies = tmp_path / "commands", tmp_path / "replies"
    commands.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")
    marker = tmp_path / "child.txt"
    benchmark.atomic_json(commands / "000001.json", {"action": "ping"})
    benchmark.atomic_json(
        commands / "000002.json",
        {
            "action": "run",
            "config": str(config),
            "role": "read",
            "native_log": str(tmp_path / "native.log"),
            "timeout": 0.2 if timeout else 2,
        },
    )
    script = f"from pathlib import Path;import os,time;Path({str(marker)!r}).write_text(str(os.getpid()));" + (
        "time.sleep(30)" if timeout else ""
    )
    monkeypatch.setattr(benchmark, "workload_argv", lambda *args: [sys.executable, "-c", script])
    monkeypatch.setattr(benchmark, "collect_inventory", lambda *args: inventory())
    monkeypatch.setattr(benchmark.signal, "signal", lambda *args: None)
    real_popen, real_atomic = benchmark.subprocess.Popen, benchmark.atomic_json
    children = []

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def publish(path, data):
        if path == replies / "000002.json":
            assert children[0].poll() is not None
        result = real_atomic(path, data)
        if path == replies / "000002.json":
            real_atomic(commands / "000003.json", {"action": "shutdown"})
        return result

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "atomic_json", publish)

    assert benchmark.agent(tmp_path) == 0

    assert json.loads((replies / "000001.json").read_text())["status"] == "pong"
    completed = json.loads((replies / "000002.json").read_text())
    assert completed["status"] == "completed" and completed["timed_out"] == timeout
    assert not completed.get("cancelled", False)
    assert (completed["returncode"] != 0) == timeout
    assert json.loads((replies / "000003.json").read_text())["status"] == "stopped"
    assert marker.is_file() and int(marker.read_text()) == children[0].pid
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


def test_agent_shutdown_cancels_and_reaps_active_reader_before_both_acknowledgments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands, replies = tmp_path / "commands", tmp_path / "replies"
    commands.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")
    benchmark.atomic_json(
        commands / "000001.json",
        {
            "action": "run",
            "config": str(config),
            "role": "read",
            "native_log": str(tmp_path / "native.log"),
            "timeout": 30,
        },
    )
    benchmark.atomic_json(commands / "000002.json", {"action": "shutdown"})
    monkeypatch.setattr(benchmark, "workload_argv", lambda *args: [sys.executable, "-c", "import time;time.sleep(60)"])
    monkeypatch.setattr(benchmark, "collect_inventory", lambda *args: inventory())
    monkeypatch.setattr(benchmark.signal, "signal", lambda *args: None)
    real_popen, real_atomic = benchmark.subprocess.Popen, benchmark.atomic_json
    children, published = [], []

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def publish(path, data):
        if path.parent == replies:
            assert children and children[0].poll() is not None
            published.append((path.name, data["status"]))
        return real_atomic(path, data)

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "atomic_json", publish)

    assert benchmark.agent(tmp_path) == 0

    assert published == [("000001.json", "completed"), ("000002.json", "stopped")]
    completed = json.loads((replies / "000001.json").read_text())
    assert completed["cancelled"] and not completed["timed_out"]
    assert completed["returncode"] != 0
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


def test_unconfirmed_reader_shutdown_force_kills_only_live_owned_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = benchmark.Controller(tmp_path, benchmark.Settings(rounds=1))
    controller.donor_reader_safe = False
    waited, killed = [], []

    class Child:
        def __init__(self, pid, returncode):
            self.pid, self.returncode = pid, returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            waited.append((self.pid, timeout))
            if self.returncode is None:
                assert (self.pid, benchmark.signal.SIGKILL) in killed
                self.returncode = -benchmark.signal.SIGKILL
            return self.returncode

    reaped, live = Child(1001, 0), Child(2002, None)
    controller.children = [reaped, live]

    def preflight_failure(*args):
        raise RuntimeError("synthetic failure while the remote reader may still hold memory")

    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(benchmark, "collect_inventory", preflight_failure)
    monkeypatch.setattr(benchmark, "read_mailbox_json", lambda path: None)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(benchmark.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(
        benchmark, "stop_process", lambda child: pytest.fail("Unsafe cleanup must not gracefully unmount a donor")
    )

    assert controller.run() == 1

    assert killed == [(live.pid, benchmark.signal.SIGKILL)]
    assert waited == [(live.pid, 5), (reaped.pid, 5)]
    assert reaped.returncode == 0 and live.returncode == -benchmark.signal.SIGKILL
    assert not controller.donor_reader_safe
    assert "reader shutdown not confirmed" in (tmp_path / "logs/perf-analysis.md").read_text()
