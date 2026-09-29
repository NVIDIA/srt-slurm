# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the publisher protocol with real local subprocesses, without a service."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from marshmallow import ValidationError

from srtctl.core.publishers import publish_results
from srtctl.core.schema import ReportingConfig, ResultPublisherConfig


def dispatch(tmp_path, script, *, exit_code=0, timeout=5):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    publisher = ResultPublisherConfig("example", [sys.executable, "-c", script], timeout_seconds=timeout)
    publish_results([publisher], log_dir=logs, job_id="42", benchmark_type="custom", run_exit_code=exit_code)
    return json.loads((logs / "publishers/example.json").read_text())


def test_command_receives_original_paths_and_outcome(tmp_path):
    receipt = dispatch(
        tmp_path,
        "import json, pathlib, sys; r=json.load(sys.stdin); "
        "pathlib.Path('request.json').write_text(json.dumps(r)); "
        "print(json.dumps({'protocol_version':1,'status':'accepted','links':{'tracking':'https://example.org/jobs/42'}}))",
        exit_code=7,
    )
    request = json.loads((tmp_path / "request.json").read_text())
    assert request == receipt["request"]
    assert request == {
        "protocol_version": 1,
        "run_dir": str(tmp_path),
        "log_dir": str(tmp_path / "logs"),
        "job_id": "42",
        "benchmark_type": "custom",
        "run_exit_code": 7,
    }
    assert receipt["state"] == "accepted"
    assert receipt["response"]["links"] == {"tracking": "https://example.org/jobs/42"}


@pytest.mark.parametrize(
    "script,error",
    [
        ("raise SystemExit(9)", "CalledProcessError"),
        ("print('not JSON')", "JSONDecodeError"),
        ("print('x' * 70000)", "ValueError"),
        ("print('{\"protocol_version\":2,\"status\":\"accepted\"}')", "ValueError"),
        ("print('{\"protocol_version\":1,\"status\":\"success\"}')", "ValueError"),
        ("print('{\"protocol_version\":1,\"status\":\"accepted\",\"links\":{\"a\":\"file:///tmp/a\"}}')", "ValueError"),
        ("print('{\"protocol_version\":1,\"status\":\"accepted\",\"links\":{\"a\":\"https://u:p@example.org\"}}')", "ValueError"),
    ],
)
def test_failures_are_separate_from_benchmark(tmp_path, script, error):
    receipt = dispatch(tmp_path, script)
    assert receipt["state"] == "unknown"
    assert receipt["error_type"] == error
    assert receipt["request"]["run_exit_code"] == 0


def test_timeout_stops_child_processes(tmp_path):
    # A descendant tries to write after the parent times out. Killing only the
    # direct child would leave this operation running outside the time bound.
    script = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', "
        "\"import time,pathlib; time.sleep(2); pathlib.Path('orphan').touch()\"]); time.sleep(30)"
    )
    receipt = dispatch(tmp_path, script, timeout=1)
    assert receipt["error_type"] == "TimeoutExpired"
    import time

    time.sleep(1.5)
    assert not (tmp_path / "orphan").exists()


def test_absent_executable_does_not_prevent_next_publisher(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    publish_results(
        [
            ResultPublisherConfig("missing", [str(tmp_path / "missing")]),
            ResultPublisherConfig("next", [sys.executable, "-c", "print('{\"protocol_version\":1,\"status\":\"skipped\"}')"]),
        ],
        log_dir=logs,
        job_id="42",
        benchmark_type="custom",
        run_exit_code=0,
    )
    assert json.loads((logs / "publishers/missing.json").read_text())["error_type"] == "FileNotFoundError"
    assert json.loads((logs / "publishers/next.json").read_text())["state"] == "skipped"


def test_commands_are_argv_not_shell(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    injected = f"$(touch {tmp_path / 'injected'})"
    script = "import json,sys; assert sys.argv[1].startswith('$('); print(json.dumps({'protocol_version':1,'status':'skipped'}))"
    publish_results(
        [ResultPublisherConfig("example", [sys.executable, "-c", script, injected])],
        log_dir=logs, job_id="42", benchmark_type="custom", run_exit_code=0,
    )
    assert not (tmp_path / "injected").exists()
    assert json.loads((logs / "publishers/example.json").read_text())["state"] == "skipped"


def test_no_publisher_does_no_io(tmp_path):
    publish_results([], log_dir=tmp_path / "absent", job_id="42", benchmark_type="custom", run_exit_code=0)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [
    {"name": "../bad", "command": ["cmd"]},
    {"name": "a", "command": []},
    {"name": "a", "command": [""]},
    {"name": "a", "command": "cmd"},
    {"name": "a", "command": ["cmd"], "timeout_seconds": 0},
])
def test_invalid_configuration_rejected(value):
    with pytest.raises((ValueError, ValidationError)):
        ResultPublisherConfig.Schema().load(value)


def test_duplicate_receipt_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        ReportingConfig.Schema().load({"publishers": [{"name": "a", "command": ["cmd"]}] * 2})


def test_diagnostics_do_not_copy_command_output(tmp_path, caplog):
    receipt = dispatch(tmp_path, "import sys; print('secret-output'); print('secret-stderr',file=sys.stderr); sys.exit(1)")
    assert "secret" not in json.dumps(receipt)
    assert "secret" not in caplog.text


def test_hook_runs_after_artifacts_before_s3_even_on_failure(tmp_path):
    from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

    mixin = PostProcessStageMixin()
    mixin.runtime = SimpleNamespace(log_dir=tmp_path / "logs", job_id="42")
    publishers = [ResultPublisherConfig("example", ["publisher"])]
    mixin.config = SimpleNamespace(reporting=ReportingConfig(publishers=publishers), benchmark=SimpleNamespace(type="custom"))
    calls = Mock()
    for name in (
        "_copy_config_to_logs", "_generate_rollup", "_extract_benchmark_results", "_compare_against_previous_lock",
        "_normalize_ruter", "_build_perf_dashboard", "_build_power_energy_report", "_run_postprocess_container",
    ):
        setattr(mixin, name, getattr(calls, name))
    mixin._get_ai_analysis_config = Mock(return_value=None)
    with patch("srtctl.cli.mixins.postprocess_stage.write_lockfile"), patch(
        "srtctl.cli.mixins.postprocess_stage.publish_results", calls.publish_results
    ):
        mixin.run_postprocess(4)
    names = [call[0] for call in calls.mock_calls]
    assert names.index("_build_power_energy_report") < names.index("publish_results") < names.index("_run_postprocess_container")
    calls.publish_results.assert_called_once_with(
        publishers, log_dir=tmp_path / "logs", job_id="42", benchmark_type="custom", run_exit_code=4
    )
