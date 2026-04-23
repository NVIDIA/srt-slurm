# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lockfile generation, per-worker fingerprint collection, and loading."""

import hashlib
import json
from pathlib import Path

import yaml

from srtctl.core.lockfile import (
    _strip_lock_section,
    build_lock_section,
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

    return SrtConfig.Schema().load(
        {
            "name": "test-job",
            "model": {"path": "/model", "container": "/c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "prefill_nodes": 1, "decode_nodes": 1},
        }
    )


def _write_recipe(tmp_path: Path, content: str | None = None) -> None:
    """Write a config.yaml that write_lockfile will read as the recipe."""
    if content is None:
        content = 'name: "test-job"\nmodel:\n  path: "/model"\n  container: "/c.sqsh"\n  precision: "fp8"\n'
    (tmp_path / "config.yaml").write_text(content)


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
# Build lock section
# ============================================================================


class TestBuildLockSection:
    def test_basic_structure(self):
        config = _make_minimal_config()
        lock = build_lock_section(config)

        assert lock["version"] == 2
        assert "generated_at" in lock
        assert "slurm" in lock
        assert "resolved" in lock
        assert lock["resolved"]["model_path"] == "/model"
        assert lock["resolved"]["container_path"] == "/c.sqsh"

    def test_with_fingerprints(self):
        config = _make_minimal_config()
        fps = {"prefill_w0": {"pip_packages": ["a==1.0"]}}
        lock = build_lock_section(config, fps)

        assert lock["fingerprints"]["prefill_w0"]["pip_packages"] == ["a==1.0"]

    def test_without_fingerprints(self):
        config = _make_minimal_config()
        lock = build_lock_section(config)

        assert "fingerprints" not in lock

    def test_with_verification(self):
        from srtctl.core.fingerprint import IdentityCheckResult

        config = _make_minimal_config()
        checks = [
            IdentityCheckResult("frameworks.dynamo", True, "1.0.0"),
            IdentityCheckResult("model.repo", False, "expected X, got Y"),
        ]
        lock = build_lock_section(config, verification=checks)

        assert lock["verification"]["verified"] == 1
        assert lock["verification"]["failed"] == 1

    # TODO: test_with_results — once rollup format is standardized

    def test_resolved_log_dir(self, tmp_path):
        config = _make_minimal_config()
        resolved = tmp_path / "outputs" / "12345" / "logs"
        lock = build_lock_section(config, resolved_log_dir=resolved)

        assert lock["resolved"]["log_dir"] == str(resolved)

    def test_slurm_context_included(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        config = _make_minimal_config()
        lock = build_lock_section(config)

        assert lock["slurm"]["job_id"] == "99999"


# ============================================================================
# Write lockfile (recipe + lock section)
# ============================================================================


class TestWriteLockfile:
    def test_preserves_recipe_text(self, tmp_path):
        """The lockfile starts with the original recipe, verbatim."""
        recipe = 'name: "test-job"\nmodel:\n  path: "/model"\n  container: "/c.sqsh"\n  precision: "fp8"\n'
        _write_recipe(tmp_path, recipe)
        config = _make_minimal_config()

        write_lockfile(tmp_path, config)

        lockfile_text = (tmp_path / "recipe.lock.yaml").read_text()
        assert lockfile_text.startswith('name: "test-job"')

    def test_has_lock_section(self, tmp_path):
        """The lockfile has a lock: section with version and slurm context."""
        _write_recipe(tmp_path)
        config = _make_minimal_config()

        write_lockfile(tmp_path, config)

        data = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text())
        assert "lock" in data
        assert data["lock"]["version"] == 2
        assert "slurm" in data["lock"]
        assert "resolved" in data["lock"]

    def test_has_comment_banner(self, tmp_path):
        """The lockfile has an explanatory comment before the lock section."""
        _write_recipe(tmp_path)
        config = _make_minimal_config()

        write_lockfile(tmp_path, config)

        text = (tmp_path / "recipe.lock.yaml").read_text()
        assert "Lock section — generated by srtctl" in text
        assert "DO NOT edit manually" in text

    def test_recipe_fields_preserved(self, tmp_path):
        """Recipe fields are readable from the lockfile."""
        _write_recipe(tmp_path)
        config = _make_minimal_config()

        write_lockfile(tmp_path, config)

        data = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text())
        assert data["name"] == "test-job"
        assert data["model"]["path"] == "/model"

    def test_rewrite_with_fingerprints(self, tmp_path):
        """Second write includes fingerprints in the lock section."""
        _write_recipe(tmp_path)
        config = _make_minimal_config()
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_fingerprint(log_dir / "fingerprint_prefill_w0.json", hostname="p-node")

        write_lockfile(tmp_path, config)
        write_lockfile(tmp_path, config, log_dir)

        data = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text())
        assert data["lock"]["fingerprints"]["prefill_w0"]["hostname"] == "p-node"

    # TODO: test_rewrite_with_results — once rollup format is standardized

    def test_never_raises(self, tmp_path):
        """write_lockfile returns False on failure, never raises."""
        config = _make_minimal_config()
        # Point to a non-writable path
        result = write_lockfile(Path("/nonexistent/path"), config)
        assert result is False

    def test_lockfile_is_valid_recipe(self, tmp_path):
        """The lockfile can be parsed as a recipe (lock: key is just extra data)."""
        _write_recipe(tmp_path)
        config = _make_minimal_config()
        write_lockfile(tmp_path, config)

        data = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text())
        # Should have both recipe fields and lock section
        assert data["name"] == "test-job"
        assert data["lock"]["version"] == 2


# ============================================================================
# Strip lock section
# ============================================================================


class TestStripLockSection:
    def test_strips_lock_key(self):
        text = 'name: "test"\nlock:\n  version: 2\n  slurm: {}\n'
        result = _strip_lock_section(text)
        assert "lock:" not in result
        assert 'name: "test"' in result

    def test_strips_comment_banner(self):
        text = 'name: "test"\n\n# ====\n# Lock section — generated by srtctl\n# ====\nlock:\n  version: 2\n'
        result = _strip_lock_section(text)
        assert "Lock section" not in result
        assert "lock:" not in result

    def test_no_lock_section_unchanged(self):
        text = 'name: "test"\nmodel:\n  path: "/model"\n'
        result = _strip_lock_section(text)
        assert result == text

    def test_preserves_recipe_content(self):
        text = 'name: "test"\nresources:\n  gpu_type: "gb200"\nlock:\n  version: 1\n'
        result = _strip_lock_section(text)
        assert 'gpu_type: "gb200"' in result
        assert "lock:" not in result

    def test_preserves_nested_lock_key(self):
        """A nested 'lock:' (not at column 0) should NOT be stripped."""
        text = 'name: "test"\nmodel:\n  lock: "some-value"\n  path: "/model"\n'
        result = _strip_lock_section(text)
        assert 'lock: "some-value"' in result
        assert 'path: "/model"' in result

    def test_preserves_indented_lock_key(self):
        """Indented lock: is not a top-level key — should be preserved."""
        text = "backend:\n  config:\n    lock: true\n    timeout: 30\n"
        result = _strip_lock_section(text)
        assert "lock: true" in result
        assert "timeout: 30" in result


# ============================================================================
# Integrity verification
# ============================================================================


class TestVerifyLockIntegrity:
    def test_valid_integrity(self):
        """Integrity hash matches when lock section is unmodified."""
        from srtctl.core.lockfile import verify_lock_integrity

        lock_data = {"version": 2, "slurm": {"job_id": "123"}, "resolved": {"model_path": "/m"}}
        content = yaml.dump(lock_data, default_flow_style=False, sort_keys=True)
        lock_data["integrity"] = hashlib.sha256(content.encode()).hexdigest()

        assert verify_lock_integrity(lock_data) is True

    def test_tampered_data_fails(self):
        """Modifying a field after hash computation should fail."""
        from srtctl.core.lockfile import verify_lock_integrity

        lock_data = {"version": 2, "slurm": {"job_id": "123"}}
        content = yaml.dump(lock_data, default_flow_style=False, sort_keys=True)
        lock_data["integrity"] = hashlib.sha256(content.encode()).hexdigest()
        lock_data["slurm"]["job_id"] = "999"  # tamper

        assert verify_lock_integrity(lock_data) is False

    def test_missing_integrity_field(self):
        """No integrity field returns False."""
        from srtctl.core.lockfile import verify_lock_integrity

        assert verify_lock_integrity({"version": 2}) is False


# ============================================================================
# Load fingerprints from lockfile
# ============================================================================


class TestLoadLockfileFingerprints:
    def test_from_new_lockfile_format(self, tmp_path):
        """Load fingerprints from lock.fingerprints in new format."""
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "lock": {"fingerprints": {"prefill_w0": {"hostname": "node-1"}}},
                }
            )
        )

        fps = load_lockfile_fingerprints(lockfile)
        assert fps["prefill_w0"]["hostname"] == "node-1"

    def test_from_legacy_lockfile(self, tmp_path):
        """Load fingerprints from top-level fingerprints (legacy v1 format)."""
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(
            yaml.dump(
                {
                    "_meta": {"version": 1},
                    "fingerprints": {"prefill_w0": {"hostname": "node-1"}},
                }
            )
        )

        fps = load_lockfile_fingerprints(lockfile)
        assert fps["prefill_w0"]["hostname"] == "node-1"

    def test_from_output_dir(self, tmp_path):
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "lock": {"fingerprints": {"w0": {"hostname": "n1"}}},
                }
            )
        )

        fps = load_lockfile_fingerprints(tmp_path)
        assert fps["w0"]["hostname"] == "n1"

    def test_from_raw_json(self, tmp_path):
        fp_file = tmp_path / "fingerprint_prefill_w0.json"
        _write_fingerprint(fp_file)

        fps = load_lockfile_fingerprints(fp_file)
        assert fps["prefill_w0"]["hostname"] == "node-001"

    def test_from_dir_with_raw_fingerprints(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        _write_fingerprint(logs / "fingerprint_w0.json", hostname="raw-node")

        fps = load_lockfile_fingerprints(tmp_path)
        assert fps["w0"]["hostname"] == "raw-node"

    def test_missing_path_returns_none(self, tmp_path):
        assert load_lockfile_fingerprints(tmp_path / "nonexistent.yaml") is None

    def test_lockfile_without_fingerprints_returns_none(self, tmp_path):
        lockfile = tmp_path / "recipe.lock.yaml"
        lockfile.write_text(yaml.dump({"name": "test", "lock": {"version": 2}}))

        assert load_lockfile_fingerprints(lockfile) is None

    def test_backward_compat_single_fingerprint(self, tmp_path):
        """Legacy lockfile with single 'fingerprint' key."""
        lockfile = tmp_path / "old.yaml"
        lockfile.write_text(yaml.dump({"fingerprint": {"hostname": "old-node"}}))

        fps = load_lockfile_fingerprints(lockfile)
        assert fps["worker"]["hostname"] == "old-node"


# ============================================================================
# Reproduction report
# ============================================================================


class TestReproductionReport:
    """Visual tests for generate_reproduction_report — prints output so you can see it."""

    def test_identical_environments(self, capsys):
        """Two identical runs — everything should match."""
        from srtctl.core.lockfile import generate_reproduction_report

        fp = {
            "hostname": "node-001",
            "arch": "aarch64",
            "os": "Ubuntu 24.04.3 LTS",
            "python_version": "3.12.3",
            "cuda_version": "Cuda compilation tools, release 13.1, V13.1.80",
            "nccl_version": "(2, 28, 9)",
            "gpu": {
                "available": True,
                "driver": "580.126.16",
                "gpus": [
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                ],
            },
            "frameworks": {"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"},
            "env": {"CUDA_VERSION": "13.1.0.036", "NCCL_VERSION": "2.28.9"},
            "pip_packages": {"python3": ["torch==2.10.0", "numpy==2.4.4", "ai-dynamo==1.0.0"]},
        }
        prev_lock = {"slurm": {"job_id": "1234"}, "fingerprints": {"agg_w0": fp}}

        summary, report, issues = generate_reproduction_report(prev_lock, {"agg_w0": fp})

        print("\n=== IDENTICAL ENVIRONMENTS ===")
        for line in report:
            print(line)

        assert len(issues) == 0
        assert any("No issues found" in line for line in report)
        assert any("280" not in line or "match" in line for line in report)

    def test_different_gpu_and_framework(self, capsys):
        """Reproducer has different GPU and framework version — should flag issues."""
        from srtctl.core.lockfile import generate_reproduction_report

        prev_fp = {
            "hostname": "lyris-001",
            "arch": "aarch64",
            "os": "Ubuntu 24.04.3 LTS",
            "python_version": "3.12.3",
            "cuda_version": "Cuda compilation tools, release 13.1, V13.1.80",
            "nccl_version": "(2, 28, 9)",
            "gpu": {
                "available": True,
                "driver": "580.126.16",
                "gpus": [
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                ],
            },
            "frameworks": {"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"},
            "env": {"CUDA_VERSION": "13.1.0.036", "NCCL_VERSION": "2.28.9"},
            "pip_packages": {"python3": ["torch==2.10.0", "numpy==2.4.4", "ai-dynamo==1.0.0"]},
        }
        new_fp = {
            "hostname": "computelab-042",
            "arch": "x86_64",
            "os": "Ubuntu 22.04.5 LTS",
            "python_version": "3.12.3",
            "cuda_version": "Cuda compilation tools, release 12.8, V12.8.93",
            "nccl_version": "(2, 25, 1)",
            "gpu": {
                "available": True,
                "driver": "570.86.15",
                "gpus": [
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                ],
            },
            "frameworks": {"dynamo": "0.8.1", "tensorrt_llm": "1.2.0"},
            "env": {"CUDA_VERSION": "12.8.0", "NCCL_VERSION": "2.25.1"},
            "pip_packages": {"python3": ["torch==2.6.0", "numpy==2.4.4", "ai-dynamo==0.8.1", "vllm==0.8.0"]},
        }
        prev_lock = {"slurm": {"job_id": "9999"}, "fingerprints": {"agg_w0": prev_fp}}

        summary, report, issues = generate_reproduction_report(prev_lock, {"agg_w0": new_fp})

        print("\n=== DIFFERENT GPU + FRAMEWORK ===")
        for line in report:
            print(line)

        assert len(issues) > 0
        assert any("gpu" in issue for issue in issues)
        assert any("dynamo" in issue for issue in issues)
        assert any("ISSUES FOUND" in line for line in report)

    def test_env_and_pip_differences_are_issues(self):
        """Env and pip drift should not produce a clean reproduction report."""
        from srtctl.core.lockfile import generate_reproduction_report

        prev_fp = {
            "arch": "aarch64",
            "os": "Ubuntu 24.04",
            "python_version": "3.12.3",
            "cuda_version": "13.1",
            "nccl_version": "2.28.9",
            "gpu": {"driver": "580.126.16", "gpus": [{"name": "NVIDIA GB200"}]},
            "frameworks": {"dynamo": "1.0.0"},
            "env": {"NCCL_DEBUG": "INFO"},
            "pip_packages": {"python3": ["torch==2.10.0"]},
        }
        new_fp = {
            **prev_fp,
            "env": {"NCCL_DEBUG": "WARN"},
            "pip_packages": {"python3": ["torch==2.11.0"]},
        }
        prev_lock = {"slurm": {"job_id": "1234"}, "fingerprints": {"agg_w0": prev_fp}}

        _summary, report, issues = generate_reproduction_report(prev_lock, {"agg_w0": new_fp})

        assert any("env vars differ" in issue for issue in issues)
        assert any("pip packages differ" in issue for issue in issues)
        assert any("ISSUES FOUND" in line for line in report)
        assert not any("No issues found" in line for line in report)

    def test_identical_no_results(self, capsys):
        """Identical environments with no results — should still show clean report."""
        from srtctl.core.lockfile import generate_reproduction_report

        fp = {
            "hostname": "node-001",
            "arch": "aarch64",
            "os": "Ubuntu 24.04.3 LTS",
            "python_version": "3.12.3",
            "cuda_version": "13.1",
            "nccl_version": "2.28.9",
            "gpu": {
                "available": True,
                "driver": "580.126.16",
                "gpus": [
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                ],
            },
            "frameworks": {"dynamo": "1.0.0"},
            "env": {},
            "pip_packages": {},
        }
        # TODO: add result regression test once rollup format is standardized
        prev_lock = {"slurm": {"job_id": "5555"}, "fingerprints": {"w0": fp}}

        summary, report, issues = generate_reproduction_report(prev_lock, {"w0": fp})

        print("\n=== IDENTICAL, NO RESULTS ===")
        for line in report:
            print(line)

        assert len(issues) == 0
        assert any("No issues found" in line for line in report)

    def test_heterogeneous_workers(self, capsys):
        """Disagg setup: prefill on GB200, decode on H100 — per-worker comparison."""
        from srtctl.core.lockfile import generate_reproduction_report

        prefill_fp = {
            "hostname": "node-001",
            "arch": "aarch64",
            "os": "Ubuntu 24.04",
            "python_version": "3.12.3",
            "cuda_version": "13.1",
            "nccl_version": "2.28.9",
            "gpu": {
                "available": True,
                "driver": "580.126.16",
                "gpus": [
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                ],
            },
            "frameworks": {"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"},
            "env": {},
            "pip_packages": {},
        }
        decode_fp = {
            "hostname": "node-002",
            "arch": "x86_64",
            "os": "Ubuntu 22.04",
            "python_version": "3.12.3",
            "cuda_version": "12.8",
            "nccl_version": "2.25.1",
            "gpu": {
                "available": True,
                "driver": "570.86.15",
                "gpus": [
                    {"name": "NVIDIA H100", "driver": "570.86.15", "memory": "81559 MiB"},
                ],
            },
            "frameworks": {"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"},
            "env": {},
            "pip_packages": {},
        }
        prev_lock = {
            "slurm": {"job_id": "7777"},
            "fingerprints": {
                "prefill_w0": prefill_fp,
                "decode_w0": decode_fp,
            },
        }
        new_fps = {"prefill_w0": prefill_fp, "decode_w0": decode_fp}

        summary, report, issues = generate_reproduction_report(prev_lock, new_fps)

        print("\n=== HETEROGENEOUS WORKERS (same topology) ===")
        for line in report:
            print(line)

        assert len(issues) == 0
        assert any("prefill_w0" in line for line in report)
        assert any("decode_w0" in line for line in report)

    def test_worker_topology_change(self, capsys):
        """Worker added/removed between runs — should flag topology change."""
        from srtctl.core.lockfile import generate_reproduction_report

        fp = {
            "hostname": "node-001",
            "arch": "aarch64",
            "os": "Ubuntu 24.04",
            "python_version": "3.12.3",
            "cuda_version": "13.1",
            "nccl_version": "2.28.9",
            "gpu": {
                "available": True,
                "driver": "580.126.16",
                "gpus": [
                    {"name": "NVIDIA GB200", "driver": "580.126.16", "memory": "189471 MiB"},
                ],
            },
            "frameworks": {"dynamo": "1.0.0"},
            "env": {},
            "pip_packages": {},
        }
        prev_lock = {
            "slurm": {"job_id": "8888"},
            "fingerprints": {
                "prefill_w0": fp,
                "decode_w0": fp,
                "decode_w1": fp,
            },
        }
        new_fps = {"prefill_w0": fp, "decode_w0": fp}

        summary, report, issues = generate_reproduction_report(prev_lock, new_fps)

        print("\n=== WORKER TOPOLOGY CHANGE ===")
        for line in report:
            print(line)

        assert any("topology" in issue.lower() or "decode_w1" in issue for issue in issues)
        assert any("REMOVED" in line for line in report)
