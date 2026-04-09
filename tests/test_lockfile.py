# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lockfile generation, per-worker fingerprint collection, and loading."""

import json
from pathlib import Path

import yaml

from srtctl.core.lockfile import (
    build_lockfile,
    collect_slurm_context,
    collect_worker_fingerprints,
    load_lockfile_fingerprints,
    write_lockfile,
)


def _write_fingerprint(path: Path, **overrides) -> None:
    """Write a fingerprint JSON file with defaults."""
    fp = {
        "hostname": "node-001",
        "timestamp": "2026-04-09T14:30:00Z",
        "arch": "aarch64",
        "python_version": "3.11.9",
        "pip_packages": ["numpy==1.26.4", "torch==2.6.0"],
    }
    fp.update(overrides)
    path.write_text(json.dumps(fp, indent=2))


def _make_minimal_config():
    """Create a minimal SrtConfig for testing."""
    from srtctl.core.schema import SrtConfig

    return SrtConfig.Schema().load({
        "name": "test-job",
        "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
    })


# ============================================================================
# Per-worker fingerprint collection
# ============================================================================


class TestCollectWorkerFingerprints:
    def test_single_file(self, tmp_path):
        _write_fingerprint(tmp_path / "fingerprint_prefill_w0.json")
        result = collect_worker_fingerprints(tmp_path)

        assert result is not None
        assert "prefill_w0" in result
        assert result["prefill_w0"]["hostname"] == "node-001"

    def test_multiple_workers_kept_separate(self, tmp_path):
        """Each worker's fingerprint is stored independently."""
        _write_fingerprint(
            tmp_path / "fingerprint_prefill_w0.json",
            hostname="prefill-node",
            pip_packages=["numpy==1.26.4", "torch==2.6.0"],
        )
        _write_fingerprint(
            tmp_path / "fingerprint_decode_w0.json",
            hostname="decode-node",
            pip_packages=["sglang==0.4.6", "torch==2.6.0"],
        )

        result = collect_worker_fingerprints(tmp_path)

        assert result is not None
        assert len(result) == 2
        assert result["prefill_w0"]["hostname"] == "prefill-node"
        assert result["decode_w0"]["hostname"] == "decode-node"
        assert result["prefill_w0"]["pip_packages"] != result["decode_w0"]["pip_packages"]

    def test_keys_sorted_by_filename(self, tmp_path):
        """Worker keys appear in sorted order."""
        _write_fingerprint(tmp_path / "fingerprint_decode_w1.json")
        _write_fingerprint(tmp_path / "fingerprint_decode_w0.json")
        _write_fingerprint(tmp_path / "fingerprint_prefill_w0.json")

        result = collect_worker_fingerprints(tmp_path)

        assert list(result.keys()) == ["decode_w0", "decode_w1", "prefill_w0"]

    def test_empty_dir_returns_none(self, tmp_path):
        assert collect_worker_fingerprints(tmp_path) is None

    def test_corrupted_files_skipped(self, tmp_path):
        """Bad JSON files are skipped, good ones still collected."""
        (tmp_path / "fingerprint_bad.json").write_text("not json")
        _write_fingerprint(tmp_path / "fingerprint_good.json")

        result = collect_worker_fingerprints(tmp_path)
        assert result is not None
        assert "good" in result

    def test_all_corrupted_returns_none(self, tmp_path):
        (tmp_path / "fingerprint_bad1.json").write_text("nope")
        (tmp_path / "fingerprint_bad2.json").write_text("{invalid")

        assert collect_worker_fingerprints(tmp_path) is None


# ============================================================================
# SLURM context
# ============================================================================


class TestCollectSlurmContext:
    def test_captures_slurm_vars(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_ACCOUNT", "myaccount")
        monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
        monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[001-004]")

        ctx = collect_slurm_context()
        assert ctx["job_id"] == "12345"
        assert ctx["account"] == "myaccount"
        assert ctx["partition"] == "gpu"
        assert ctx["nodelist"] == "node-[001-004]"

    def test_always_has_user_and_cwd(self):
        ctx = collect_slurm_context()
        assert "user" in ctx
        assert "cwd" in ctx

    def test_missing_slurm_vars_omitted(self, monkeypatch):
        """Outside SLURM, SLURM keys are simply absent."""
        for _, env_var in [("job_id", "SLURM_JOB_ID"), ("cluster", "SLURM_CLUSTER_NAME")]:
            monkeypatch.delenv(env_var, raising=False)

        ctx = collect_slurm_context()
        assert "job_id" not in ctx
        assert "cluster" not in ctx
        assert "user" in ctx


# ============================================================================
# Build lockfile
# ============================================================================


class TestBuildLockfile:
    def test_structure_with_per_worker_fingerprints(self):
        config = _make_minimal_config()
        fps = {"prefill_w0": {"pip_packages": ["a==1.0"]}}
        lockfile = build_lockfile(config, fps)

        assert "_meta" in lockfile
        assert "config" in lockfile
        assert "fingerprints" in lockfile
        assert lockfile["_meta"]["version"] == 1
        assert lockfile["fingerprints"]["prefill_w0"]["pip_packages"] == ["a==1.0"]

    def test_no_fingerprints(self):
        config = _make_minimal_config()
        lockfile = build_lockfile(config)

        assert lockfile["fingerprints"] is None
        assert lockfile["config"]["name"] == "test-job"

    def test_config_section_has_model_fields(self):
        config = _make_minimal_config()
        lockfile = build_lockfile(config)

        assert lockfile["config"]["model"]["path"] == "/model"
        assert lockfile["config"]["model"]["precision"] == "fp8"

    def test_meta_includes_slurm_context(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        config = _make_minimal_config()
        lockfile = build_lockfile(config)

        assert lockfile["_meta"]["slurm"]["job_id"] == "99999"
        assert "user" in lockfile["_meta"]["slurm"]


# ============================================================================
# Write lockfile
# ============================================================================


class TestWriteLockfile:
    def test_initial_write_without_fingerprints(self, tmp_path):
        """Phase 1: write at job start with config + SLURM context, no fingerprints."""
        config = _make_minimal_config()

        assert write_lockfile(tmp_path, config) is True

        data = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text())
        assert data["_meta"]["version"] == 1
        assert data["config"]["name"] == "test-job"
        assert data["fingerprints"] is None

    def test_rewrite_with_per_worker_fingerprints(self, tmp_path):
        """Phase 2: rewrite at job end with per-worker fingerprints."""
        config = _make_minimal_config()
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_fingerprint(log_dir / "fingerprint_prefill_w0.json", hostname="p-node")
        _write_fingerprint(log_dir / "fingerprint_decode_w0.json", hostname="d-node")

        write_lockfile(tmp_path, config)
        assert write_lockfile(tmp_path, config, log_dir) is True

        data = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text())
        assert data["config"]["name"] == "test-job"
        assert data["fingerprints"]["prefill_w0"]["hostname"] == "p-node"
        assert data["fingerprints"]["decode_w0"]["hostname"] == "d-node"

    def test_never_raises_on_failure(self):
        config = _make_minimal_config()
        result = write_lockfile(Path("/proc/nonexistent"), config, Path("/proc/nonexistent"))
        assert result is False


# ============================================================================
# Load lockfile fingerprints
# ============================================================================


class TestLoadLockfileFingerprints:
    def test_from_lockfile_yaml(self, tmp_path):
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(yaml.dump({
            "_meta": {"version": 1},
            "config": {},
            "fingerprints": {
                "prefill_w0": {"pip_packages": ["a==1.0"]},
                "decode_w0": {"pip_packages": ["b==2.0"]},
            },
        }))

        result = load_lockfile_fingerprints(lockfile)
        assert result is not None
        assert "prefill_w0" in result
        assert "decode_w0" in result

    def test_from_output_dir(self, tmp_path):
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(yaml.dump({
            "_meta": {"version": 1},
            "config": {},
            "fingerprints": {"prefill_w0": {"pip_packages": ["b==2.0"]}},
        }))

        result = load_lockfile_fingerprints(tmp_path)
        assert result is not None
        assert result["prefill_w0"]["pip_packages"] == ["b==2.0"]

    def test_from_raw_json(self, tmp_path):
        fp_file = tmp_path / "fingerprint_prefill_w0.json"
        fp_file.write_text(json.dumps({"pip_packages": ["c==3.0"]}))

        result = load_lockfile_fingerprints(fp_file)
        assert result is not None
        assert "prefill_w0" in result
        assert result["prefill_w0"]["pip_packages"] == ["c==3.0"]

    def test_from_dir_with_raw_fingerprints(self, tmp_path):
        """Directory without lockfile falls back to collecting fingerprint files."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        _write_fingerprint(logs_dir / "fingerprint_prefill_w0.json")

        result = load_lockfile_fingerprints(tmp_path)
        assert result is not None
        assert "prefill_w0" in result

    def test_missing_path_returns_none(self, tmp_path):
        assert load_lockfile_fingerprints(tmp_path / "nonexistent") is None

    def test_lockfile_without_fingerprints_returns_none(self, tmp_path):
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(yaml.dump({"_meta": {"version": 1}, "config": {}}))

        result = load_lockfile_fingerprints(lockfile)
        assert result is None

    def test_backward_compat_single_fingerprint(self, tmp_path):
        """Old lockfiles with single 'fingerprint' key still load."""
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(yaml.dump({
            "_meta": {"version": 1},
            "config": {},
            "fingerprint": {"pip_packages": ["old==1.0"]},
        }))

        result = load_lockfile_fingerprints(lockfile)
        assert result is not None
        assert "worker" in result
        assert result["worker"]["pip_packages"] == ["old==1.0"]
