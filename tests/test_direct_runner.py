# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from srtctl.render import direct_runner
from srtctl.render.direct_runner import DirectRunInterrupted, DirectRunner, _router_counts
from srtctl.render.direct_stages import (
    AuxiliaryServiceStageMixin,
    BenchmarkStageMixin,
    InfrastructureStageMixin,
    PostProcessStageMixin,
    RuntimeSetupStageMixin,
    ServingStageMixin,
    TelemetryStageMixin,
)


def _plan(tmp_path) -> dict[str, object]:
    return {
        "source_dir": str(tmp_path / "source"),
        "output_base": str(tmp_path / "output-base"),
        "sglang_source": str(tmp_path / "sglang-source"),
        "frontend_port": 8000,
        "global_environment": [],
        "benchmark_environment": [],
        "dynamo_source_cache_key": "dynamo-test",
        "tachometer_enabled": False,
        "ruter_enabled": False,
    }


def _runner(tmp_path, monkeypatch) -> DirectRunner:
    output = tmp_path / "output"
    logs = output / "logs"
    artifacts = output / "artifacts"
    logs.mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(output))
    monkeypatch.setenv("LOG_DIR", str(logs))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.setenv("SRTCTL_PYTHON", sys.executable)
    return DirectRunner(_plan(tmp_path))


def test_router_counts_accepts_dynamo_health_shape() -> None:
    assert _router_counts(
        {
            "instances": [
                {"endpoint": "generate", "component": "prefill"},
                {"endpoint": "generate", "component": "decode"},
                {"endpoint": "generate", "component": "backend"},
                {"endpoint": "metrics", "component": "prefill"},
            ]
        },
    ) == (1, 2)


def test_runner_composes_direct_only_stages() -> None:
    assert DirectRunner.__bases__ == (
        RuntimeSetupStageMixin,
        InfrastructureStageMixin,
        ServingStageMixin,
        TelemetryStageMixin,
        AuxiliaryServiceStageMixin,
        BenchmarkStageMixin,
        PostProcessStageMixin,
    )


def test_runner_executes_as_a_direct_file() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(direct_runner.__file__)), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Rendered direct execution plan" in result.stdout


def test_runner_tracks_subprocess_groups_and_separate_logs(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    worker = runner._launch_shell("worker-0", "worker-0.log", f"{sys.executable} -c 'print(\"worker ready\")'")

    assert worker.process.wait(timeout=10) == 0
    assert (runner.log_dir / "worker-0.log").read_text(encoding="utf-8") == "worker ready\n"
    assert (runner.log_dir / "worker-0.log.command").is_file()
    assert (runner.log_dir / "worker-0.log.pid").read_text(encoding="utf-8") == f"{worker.process.pid}\n"

    runner._cleanup()


def test_dynamo_cache_key_includes_current_python_and_host_shape(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    key = runner._dynamo_source_cache_key()

    assert key.startswith("dynamo-test-")
    assert sys.implementation.cache_tag in key
    assert len(key.rsplit("-", 1)[1]) == 12


def test_runner_converts_termination_to_cleanup_signal(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)

    with pytest.raises(DirectRunInterrupted) as error:
        runner._on_signal(signal.SIGTERM, None)

    assert error.value.signal_number == signal.SIGTERM


def test_runner_keeps_cleanup_best_effort_when_ruter_python_is_missing(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    runner.plan["ruter_enabled"] = True

    runner._normalize_ruter()

    assert "ruter.log" in {path.name for path in runner.log_dir.iterdir()}


def test_dynamo_source_archive_excludes_build_artifacts(tmp_path) -> None:
    source = tmp_path / "build" / "dynamo"
    (source / "src").mkdir(parents=True)
    (source / "target").mkdir()
    (source / ".git").mkdir()
    (source / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "target" / "ignored").write_text("ignored\n", encoding="utf-8")
    (source / ".git" / "ignored").write_text("ignored\n", encoding="utf-8")
    archive = tmp_path / "dynamo-src.tar.gz"

    DirectRunner._write_archive(tmp_path / "build", "dynamo", archive)

    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "dynamo/src/main.py" in names
    assert not any("target" in name or ".git" in name for name in names)


def _aux_service(**overrides) -> dict[str, object]:
    service = {
        "name": "router",
        "command": [sys.executable, "-c", "print('ok')"],
        "container_image": None,
        "env": {},
        "source": None,
        "build_command": None,
        "inherit_discovery_env": True,
    }
    service.update(overrides)
    return service


class TestAuxiliaryServiceStage:
    def test_no_services_is_a_noop(self, tmp_path, monkeypatch) -> None:
        runner = _runner(tmp_path, monkeypatch)

        runner._start_auxiliary_services()

        assert runner.auxiliary_services == {}
        assert runner.processes == []

    def test_launches_and_injects_discovery_env(self, tmp_path, monkeypatch) -> None:
        runner = _runner(tmp_path, monkeypatch)
        runner.plan["etcd_client_port"] = 2379
        runner.plan["nats_port"] = 4222
        marker = tmp_path / "marker.txt"
        script = (
            "import os\n"
            f"open({str(marker)!r}, 'w').write(os.environ.get('ETCD_ENDPOINTS', '') + '|' "
            "+ os.environ.get('NATS_SERVER', '') + '|' + os.environ.get('FOO', ''))\n"
            "import time\ntime.sleep(5)\n"
        )
        runner.plan["auxiliary_services"] = [_aux_service(command=[sys.executable, "-c", script], env={"FOO": "bar"})]

        runner._start_auxiliary_services()

        assert "router" in runner.auxiliary_services
        assert runner.auxiliary_services["router"] in runner.processes
        for _ in range(50):
            if marker.exists():
                break
            time.sleep(0.1)
        assert marker.read_text(encoding="utf-8") == "http://127.0.0.1:2379|nats://127.0.0.1:4222|bar"
        runner._cleanup()

    def test_inherit_discovery_env_false_omits_etcd_and_nats(self, tmp_path, monkeypatch) -> None:
        runner = _runner(tmp_path, monkeypatch)
        runner.plan["etcd_client_port"] = 2379
        runner.plan["nats_port"] = 4222
        marker = tmp_path / "marker.txt"
        script = (
            "import os\n"
            f"open({str(marker)!r}, 'w').write('present' if 'ETCD_ENDPOINTS' in os.environ else 'absent')\n"
            "import time\ntime.sleep(5)\n"
        )
        runner.plan["auxiliary_services"] = [
            _aux_service(command=[sys.executable, "-c", script], inherit_discovery_env=False)
        ]

        runner._start_auxiliary_services()

        for _ in range(50):
            if marker.exists():
                break
            time.sleep(0.1)
        assert marker.read_text(encoding="utf-8") == "absent"
        runner._cleanup()

    def test_service_exiting_at_startup_dies(self, tmp_path, monkeypatch) -> None:
        runner = _runner(tmp_path, monkeypatch)
        runner.plan["etcd_client_port"] = 2379
        runner.plan["nats_port"] = 4222
        runner.plan["auxiliary_services"] = [_aux_service(command=[sys.executable, "-c", "import sys; sys.exit(1)"])]

        with pytest.raises(RuntimeError, match="router"):
            runner._start_auxiliary_services()

        runner._cleanup()

    def test_launches_in_declared_order(self, tmp_path, monkeypatch) -> None:
        runner = _runner(tmp_path, monkeypatch)
        runner.plan["etcd_client_port"] = 2379
        runner.plan["nats_port"] = 4222
        marker = tmp_path / "order.txt"
        script = (
            f"import pathlib; p = pathlib.Path({str(marker)!r}); "
            "p.write_text((p.read_text() if p.exists() else '') + {name!r})\n"
            "import time\ntime.sleep(5)\n"
        )
        marker.write_text("", encoding="utf-8")
        runner.plan["auxiliary_services"] = [
            _aux_service(name="b", command=[sys.executable, "-c", script.format(name="b,")]),
            _aux_service(name="a", command=[sys.executable, "-c", script.format(name="a,")]),
            _aux_service(name="c", command=[sys.executable, "-c", script.format(name="c,")]),
        ]

        runner._start_auxiliary_services()

        assert list(runner.auxiliary_services.keys()) == ["b", "a", "c"]
        for _ in range(50):
            if marker.read_text(encoding="utf-8").count(",") == 3:
                break
            time.sleep(0.1)
        assert marker.read_text(encoding="utf-8") == "b,a,c,"
        runner._cleanup()
