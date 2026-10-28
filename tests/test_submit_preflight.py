# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-time checks that catch nsys and Tachometer problems before the allocation."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from rich.console import Console
from test_dry_run import _make_config

from srtctl.cli import submit

OBSERVED = {"observability": {"enabled": True}, "frontend": {"type": "dynamo"}}


@pytest.fixture
def cluster(monkeypatch):
    """srtslurm.yaml settings seen by submit; SRTCTL_NSYS_BIN unset unless a test sets it."""
    settings: dict = {}
    monkeypatch.delenv("SRTCTL_NSYS_BIN", raising=False)
    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: settings.get(key, default))
    return settings


def executable(path: Path, body: bytes = b"#!/bin/sh\n") -> Path:
    path.write_bytes(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_nsys_choice_order(cluster, monkeypatch):
    assert submit.nsys_binary_choice(_make_config({"engine": "vllm"})) is None  # No nsys runs.
    config = _make_config({**OBSERVED, "engine": "vllm"})
    assert submit.nsys_binary_choice(config) == ("nsys", "image PATH")
    cluster["nsys_binary"] = "/shared/nsys"
    assert submit.nsys_binary_choice(config) == ("/shared/nsys", "srtslurm.yaml nsys_binary")
    monkeypatch.setenv("SRTCTL_NSYS_BIN", "/mine/nsys")
    assert submit.nsys_binary_choice(config) == ("/mine/nsys", "SRTCTL_NSYS_BIN")


def test_absolute_nsys_must_exist_and_execute(cluster, tmp_path):
    config = _make_config({**OBSERVED, "engine": "vllm"})
    cluster["nsys_binary"] = str(tmp_path / "missing" / "nsys")
    assert "not found on this host" in submit._nsys_problem(config, tmp_path)
    plain = tmp_path / "nsys"
    plain.write_text("")
    cluster["nsys_binary"] = str(plain)
    assert "not executable" in submit._nsys_problem(config, tmp_path)
    cluster["nsys_binary"] = str(executable(tmp_path / "nsys-ok"))
    assert submit._nsys_problem(config, tmp_path) is None


def test_nsys_built_for_another_arch_is_rejected(cluster, tmp_path, monkeypatch):
    config = _make_config({**OBSERVED, "engine": "vllm"})
    cluster["nsys_binary"] = str(executable(tmp_path / "nsys"))
    monkeypatch.setattr(submit, "arch_from_binary", lambda _path: "x86_64")
    monkeypatch.setattr(submit, "detect_target_arch", lambda _source: "aarch64")
    assert "built for x86_64, but the compute nodes are aarch64" in submit._nsys_problem(config, tmp_path)


@pytest.mark.parametrize(("engine", "warns"), [("vllm", True), ("sglang", False), ("trtllm", False)])
def test_image_nsys_warning_only_for_engines_whose_image_lacks_it(cluster, engine, warns):
    config = _make_config({**OBSERVED, "engine": engine})
    assert (submit.nsys_image_warning(config) is not None) == warns
    cluster["nsys_binary"] = "/shared/nsys"
    assert submit.nsys_image_warning(config) is None  # A configured nsys silences it.


def test_job_script_records_the_checked_nsys(cluster, tmp_path):
    cluster["nsys_binary"] = "/shared/nsys"
    config = _make_config({**OBSERVED, "engine": "vllm"})
    script = submit.generate_minimal_sbatch_script(config, tmp_path / "config.yaml")
    assert "export SRTCTL_NSYS_BIN=/shared/nsys" in script
    cluster.clear()
    assert "SRTCTL_NSYS_BIN" not in submit.generate_minimal_sbatch_script(config, tmp_path / "config.yaml")


def test_dry_run_names_nsys_source_and_warning(cluster, capsys, monkeypatch):
    monkeypatch.setattr(submit, "console", Console(width=200))
    submit.show_config_details(_make_config({**OBSERVED, "engine": "vllm"}))
    output = capsys.readouterr().out
    assert "nsys (image PATH)" in output and "nsys warning" in output


def test_old_tachometer_scraper_is_detected(tmp_path):
    (tmp_path / "bin").mkdir()
    scraper = executable(tmp_path / "bin" / "tachometer-scraper", b"\x7fELF old build")
    assert "timestamp_ns" in submit._tachometer_scraper_problem(tmp_path)
    executable(scraper, b"\x7fELF ... timestamp_ns ...")
    assert submit._tachometer_scraper_problem(tmp_path) is None


def setup_tree(root: Path, collector: bool = True) -> None:
    """What make setup installs; the OTel collector is left out on request."""
    names = ["configs/nats-server", "configs/etcd", "configs/process-exporter", "bin/uv"]
    if collector:
        names.append("configs/otelcol-contrib")
    for name in names:
        (root / name).parent.mkdir(exist_ok=True)
        executable(root / name)
    executable(root / "bin" / "tachometer-scraper", b"timestamp_ns")


def test_validate_setup_stops_on_unusable_nsys(cluster, tmp_path):
    setup_tree(tmp_path)
    cluster["nsys_binary"] = str(tmp_path / "absent-nsys")
    with pytest.raises(SystemExit):
        submit.validate_setup(tmp_path, _make_config({**OBSERVED, "engine": "vllm"}))


def test_dynamo_frontend_needs_the_otel_collector_before_the_allocation(cluster, tmp_path, capsys):
    setup_tree(tmp_path, collector=False)
    config = _make_config({"frontend": {"type": "dynamo"}})
    with pytest.raises(SystemExit):
        submit.validate_setup(tmp_path, config)
    assert "otel-collector-download" in capsys.readouterr().out
    executable(tmp_path / "configs" / "otelcol-contrib")
    submit.validate_setup(tmp_path, config)


def test_disabled_otel_collector_needs_no_binary(cluster, tmp_path):
    setup_tree(tmp_path, collector=False)
    services = [{"name": "otel-collector", "type": "otel-collector", "enabled": False}]
    submit.validate_setup(tmp_path, _make_config({"frontend": {"type": "dynamo"}, "services": services}))
