# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone fingerprint script mounted into workers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from srtctl.runtime_scripts import fingerprint as runtime_fingerprint

ARM_CPUINFO = """\
processor       : 0
BogoMIPS        : 2000.00
Features        : fp asimd evtstrm
CPU implementer : 0x41
CPU architecture: 8
CPU variant     : 0x1
CPU part        : 0xd49
CPU revision    : 1
"""

ARM_CPU_MODEL = "ARM CPU implementer 0x41 part 0xd49 (architecture 8, variant 0x1, revision 1)"


def test_cpu_info_uses_arm_fallback(tmp_path, monkeypatch):
    cpuinfo_path = tmp_path / "cpuinfo"
    cpuinfo_path.write_text(ARM_CPUINFO)
    monkeypatch.setattr(runtime_fingerprint.os, "sched_getaffinity", lambda _pid: {0, 1, 3})
    monkeypatch.setattr(runtime_fingerprint.os, "cpu_count", lambda: 8)

    cpu = runtime_fingerprint.cpu_info(
        cpuinfo_path,
        {"SLURM_CPUS_ON_NODE": "3", "IGNORED": "value"},
    )

    assert cpu == {
        "model": ARM_CPU_MODEL,
        "logical_cpus": 8,
        "affinity_cpus": 3,
        "affinity_list": "0-1,3",
        "slurm": {"SLURM_CPUS_ON_NODE": "3"},
    }


def test_environment_variables_redacts_secrets():
    captured = runtime_fingerprint.environment_variables(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "HF_TOKEN": "secret-value",
            "UNRELATED": "ignored",
        }
    )

    assert captured == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HF_TOKEN": "***REDACTED***",
    }


def test_model_identity_reads_huggingface_metadata(tmp_path):
    revision = "a" * 40
    refs_path = tmp_path / ".huggingface" / "refs" / "main"
    refs_path.parent.mkdir(parents=True)
    refs_path.write_text(revision)
    (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": "org/model"}))

    assert runtime_fingerprint.model_identity(tmp_path) == {
        "hf_revision": revision,
        "model_id": "org/model",
    }


def test_model_identity_ignores_wrong_shaped_json(tmp_path):
    metadata_path = tmp_path / ".huggingface" / "download_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("null")
    (tmp_path / "config.json").write_text(json.dumps("_name_or_path"))

    assert runtime_fingerprint.model_identity(tmp_path) is None


def test_run_contains_unexpected_probe_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")

    monkeypatch.setattr(runtime_fingerprint.subprocess, "run", fail)

    assert runtime_fingerprint.run(["probe"]) is None


def test_capture_isolates_probe_failures(tmp_path, monkeypatch):
    def fail(_model_path):
        raise RuntimeError("broken model metadata")

    monkeypatch.setattr(runtime_fingerprint, "find_python", lambda: "python3")
    monkeypatch.setattr(runtime_fingerprint, "os_description", lambda: "Test OS")
    monkeypatch.setattr(runtime_fingerprint, "cpu_info", lambda _path: {"model": "Test CPU"})
    monkeypatch.setattr(runtime_fingerprint, "gpu_info", lambda: {"available": False})
    monkeypatch.setattr(runtime_fingerprint, "cuda_version", lambda: "unavailable")
    monkeypatch.setattr(runtime_fingerprint, "nccl_version", lambda _python: "unavailable")
    monkeypatch.setattr(runtime_fingerprint, "framework_versions", lambda _python: {})
    monkeypatch.setattr(runtime_fingerprint, "model_identity", fail)
    monkeypatch.setattr(runtime_fingerprint, "environment_variables", lambda: {})
    monkeypatch.setattr(runtime_fingerprint, "pip_packages", lambda _python: {})

    fingerprint = runtime_fingerprint.capture_fingerprint(tmp_path)

    assert fingerprint["model"] is None
    assert fingerprint["cpu"] == {"model": "Test CPU"}


def test_main_writes_capture_result(tmp_path, monkeypatch):
    output_path = tmp_path / "fingerprint.json"
    expected = {"hostname": "worker-0", "cpu": {"model": "test"}}
    monkeypatch.setattr(runtime_fingerprint, "capture_fingerprint", lambda _model_path: expected)

    runtime_fingerprint.main(
        [
            "--output",
            str(output_path),
            "--model-path",
            str(tmp_path / "model"),
        ]
    )

    assert json.loads(output_path.read_text()) == expected


def test_runtime_script_executes_directly(tmp_path):
    output_path = tmp_path / "fingerprint.json"
    script_path = Path(runtime_fingerprint.__file__)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--output",
            str(output_path),
            "--model-path",
            str(tmp_path / "missing-model"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    fingerprint = json.loads(output_path.read_text())
    assert list(fingerprint) == [
        "hostname",
        "timestamp",
        "arch",
        "os",
        "cpu",
        "gpu",
        "python_version",
        "cuda_version",
        "nccl_version",
        "frameworks",
        "model",
        "env",
        "pip_packages",
    ]
    assert fingerprint["hostname"]
