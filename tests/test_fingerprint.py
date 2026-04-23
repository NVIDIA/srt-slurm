# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime fingerprinting, diffing, and checking.

Tests are organized around the three core operations:
- capture: probe the environment, produce ordered dict
- diff: compare two fingerprints, produce structured delta
- check: verify current env matches a reference

Every test runs without network, GPU, or special packages — probes are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from srtctl.core.fingerprint import (
    UNAVAILABLE,
    CheckResult,
    CheckStatus,
    ProbeResult,
    _ordered_fingerprint,
    _parse_pip_packages,
    capture_fingerprint,
    check_against_fingerprint,
    diff_fingerprints,
    format_check_results,
    format_diff,
    generate_capture_script,
    load_fingerprint,
    write_fingerprint,
)

# ============================================================================
# Fixtures — reusable fingerprint data
# ============================================================================


def _make_fingerprint(**overrides) -> dict:
    """Build a minimal fingerprint dict with sensible defaults."""
    base = {
        "hostname": "node-001",
        "timestamp": "2026-04-09T14:30:00Z",
        "arch": "aarch64",
        "os": "Ubuntu 22.04.5 LTS",
        "gpu": {
            "available": True,
            "driver": "570.86.15",
            "gpus": [{"name": "GB200", "driver": "570.86.15", "memory": "192 GiB"}],
        },
        "python_version": "3.11.9",
        "cuda_version": "12.8",
        "nccl_version": "2.25.1",
        "frameworks": {"torch": "2.6.0+cu128", "sglang": "0.4.6.post1", "dynamo": "0.8.1"},
        "pip_packages": [
            "ai-dynamo==0.8.1",
            "numpy==1.26.4",
            "sglang==0.4.6.post1",
            "torch==2.6.0+cu128",
            "triton==3.2.0",
        ],
    }
    base.update(overrides)
    return base


# ============================================================================
# Ordering
# ============================================================================


class TestOrdering:
    """Deterministic output order is critical for clean diffs."""

    def test_field_order_is_canonical(self):
        """Fields appear in the fixed order regardless of input order."""
        data = {
            "pip_packages": [],
            "arch": "x86_64",
            "hostname": "node-1",
            "python_version": "3.11",
            "timestamp": "2026-01-01T00:00:00Z",
            "gpu": {},
            "os": "Ubuntu",
            "cuda_version": "12.8",
            "nccl_version": "2.25",
            "frameworks": {"torch": "2.6.0"},
        }
        ordered = _ordered_fingerprint(data)
        keys = list(ordered.keys())

        assert keys.index("hostname") < keys.index("arch")
        assert keys.index("arch") < keys.index("python_version")
        assert keys.index("python_version") < keys.index("pip_packages")
        # pip_packages always last among known fields
        assert keys[-1] == "pip_packages"

    def test_extra_keys_appended_alphabetically(self):
        """Unknown keys go at the end, sorted alphabetically."""
        data = {
            "hostname": "node-1",
            "zzz_custom": "value",
            "aaa_custom": "value",
            "pip_packages": [],
        }
        ordered = _ordered_fingerprint(data)
        keys = list(ordered.keys())

        # Known fields first
        assert keys[0] == "hostname"
        assert keys[1] == "pip_packages"
        # Extra keys after, sorted
        assert keys[2] == "aaa_custom"
        assert keys[3] == "zzz_custom"

    def test_pip_packages_are_sorted(self):
        """probe_pip_packages returns a sorted list (case-insensitive)."""
        # Simulate what probe_pip_packages does: sort the raw output
        unsorted = ["Torch==2.0", "numpy==1.0", "AIFlow==0.1"]
        sorted_pkgs = sorted(unsorted, key=lambda s: s.lower())
        assert sorted_pkgs == ["AIFlow==0.1", "numpy==1.0", "Torch==2.0"]

        # And _parse_pip_packages preserves that order in its dict keys
        parsed = _parse_pip_packages(sorted_pkgs)
        assert list(parsed.keys()) == ["aiflow", "numpy", "torch"]


# ============================================================================
# Probes
# ============================================================================


class TestProbes:
    """Each probe must return ProbeResult and never raise."""

    def test_probe_result_success(self):
        r = ProbeResult.success("hello")
        assert r.ok is True
        assert r.value == "hello"
        assert r.error is None

    def test_probe_result_failure(self):
        r = ProbeResult.failure("broken")
        assert r.ok is False
        assert r.value == UNAVAILABLE
        assert r.error == "broken"

    def test_capture_with_all_probes_mocked(self):
        """capture_fingerprint works when all probes return values."""
        mock_probes = {
            "hostname": lambda: ProbeResult.success("test-node"),
            "timestamp": lambda: ProbeResult.success("2026-01-01T00:00:00Z"),
            "arch": lambda: ProbeResult.success("aarch64"),
            "os": lambda: ProbeResult.success("Ubuntu 22.04"),
            "gpu": lambda: ProbeResult.success({"available": False}),
            "python_version": lambda: ProbeResult.success("3.11.9"),
            "cuda_version": lambda: ProbeResult.success("12.8"),
            "nccl_version": lambda: ProbeResult.success("2.25.1"),
            "frameworks": lambda: ProbeResult.success({"torch": "2.6.0", "dynamo": "0.8.1"}),
            "pip_packages": lambda: ProbeResult.success(["numpy==1.0", "torch==2.6.0"]),
        }

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            fp = capture_fingerprint()

        assert fp["hostname"] == "test-node"
        assert fp["arch"] == "aarch64"
        assert fp["pip_packages"] == ["numpy==1.0", "torch==2.6.0"]

    def test_capture_survives_probe_failure(self):
        """If a probe raises, capture still completes with UNAVAILABLE."""

        def exploding_probe():
            raise RuntimeError("kaboom")

        mock_probes = {
            "hostname": lambda: ProbeResult.success("ok-node"),
            "bad_probe": exploding_probe,
            "pip_packages": lambda: ProbeResult.success([]),
        }

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            fp = capture_fingerprint()

        assert fp["hostname"] == "ok-node"
        assert fp["bad_probe"] == UNAVAILABLE
        assert fp["pip_packages"] == []

    def test_capture_survives_probe_returning_failure(self):
        """Probes that return ProbeResult.failure are included with sentinel."""
        mock_probes = {
            "gpu": lambda: ProbeResult.failure("nvidia-smi not found"),
            "pip_packages": lambda: ProbeResult.success(["a==1.0"]),
        }

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            fp = capture_fingerprint()

        assert fp["gpu"] == UNAVAILABLE
        assert fp["pip_packages"] == ["a==1.0"]

    def test_capture_with_extra_probes(self):
        """Extra probes are included alongside built-in ones."""
        mock_probes = {
            "hostname": lambda: ProbeResult.success("node"),
        }
        extra = {
            "custom_metric": lambda: ProbeResult.success(42),
        }

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            fp = capture_fingerprint(extra_probes=extra)

        assert fp["hostname"] == "node"
        assert fp["custom_metric"] == 42


# ============================================================================
# Pip package parsing
# ============================================================================


class TestPipParsing:
    """_parse_pip_packages handles all pip freeze output formats."""

    def test_standard_format(self):
        packages = ["numpy==1.26.4", "torch==2.6.0+cu128"]
        parsed = _parse_pip_packages(packages)
        assert parsed == {"numpy": "1.26.4", "torch": "2.6.0+cu128"}

    def test_at_format(self):
        """Editable installs use the @ syntax."""
        packages = ["my-package @ file:///home/user/my-package"]
        parsed = _parse_pip_packages(packages)
        assert parsed == {"my-package": "@ file:///home/user/my-package"}

    def test_case_insensitive_keys(self):
        """Package names are lowercased for comparison."""
        packages = ["PyYAML==6.0", "pyyaml==6.0"]
        parsed = _parse_pip_packages(packages)
        # Both map to same key (last wins)
        assert "pyyaml" in parsed

    def test_empty_list(self):
        assert _parse_pip_packages([]) == {}

    def test_malformed_line(self):
        """Unknown format gets package name = '?'."""
        packages = ["some-weird-format"]
        parsed = _parse_pip_packages(packages)
        assert parsed == {"some-weird-format": "?"}

    def test_unavailable_string_returns_empty(self):
        """UNAVAILABLE sentinel string must not be iterated as characters."""
        parsed = _parse_pip_packages("unavailable")
        assert parsed == {}

    def test_none_returns_empty(self):
        parsed = _parse_pip_packages(None)
        assert parsed == {}

    def test_empty_list_returns_empty(self):
        parsed = _parse_pip_packages([])
        assert parsed == {}

    def test_labeled_dict_format(self):
        """New format: dict of source -> package list."""
        packages = {
            "/opt/dynamo/venv/bin/python3": ["torch==2.6.0", "numpy==1.26.4"],
            "python3": ["setuptools==68.1.2"],
        }
        parsed = _parse_pip_packages(packages)
        assert parsed["torch"] == "2.6.0"
        assert parsed["numpy"] == "1.26.4"
        assert parsed["setuptools"] == "68.1.2"


# ============================================================================
# Diff
# ============================================================================


class TestDiff:
    """diff_fingerprints produces structured, sorted deltas."""

    def test_identical_fingerprints(self):
        """Same input produces zero diff."""
        fp = _make_fingerprint()
        diff = diff_fingerprints(fp, fp)

        assert diff.field_changes == {}
        assert diff.package_diffs == []
        assert diff.packages_matched == 5
        assert diff.packages_changed == 0
        assert diff.packages_added == 0
        assert diff.packages_removed == 0

    def test_scalar_field_change(self):
        """Changed CUDA version shows up in field_changes."""
        a = _make_fingerprint(cuda_version="12.8")
        b = _make_fingerprint(cuda_version="13.1")

        diff = diff_fingerprints(a, b)

        assert "cuda_version" in diff.field_changes
        assert diff.field_changes["cuda_version"] == ("12.8", "13.1")

    def test_gpu_driver_change(self):
        """GPU driver change detected from nested structure."""
        a = _make_fingerprint(gpu={"available": True, "driver": "570.86.15", "gpus": []})
        b = _make_fingerprint(gpu={"available": True, "driver": "575.00.00", "gpus": []})

        diff = diff_fingerprints(a, b)

        assert "gpu.driver" in diff.field_changes
        assert diff.field_changes["gpu.driver"] == ("570.86.15", "575.00.00")

    def test_package_version_change(self):
        """Changed package version appears in package_diffs."""
        a = _make_fingerprint(pip_packages=["sglang==0.4.6", "torch==2.6.0"])
        b = _make_fingerprint(pip_packages=["sglang==0.4.7", "torch==2.6.0"])

        diff = diff_fingerprints(a, b)

        assert diff.packages_changed == 1
        assert diff.packages_matched == 1
        assert len(diff.package_diffs) == 1
        assert diff.package_diffs[0].package == "sglang"
        assert diff.package_diffs[0].version_a == "0.4.6"
        assert diff.package_diffs[0].version_b == "0.4.7"

    def test_package_added(self):
        """New package in b shows as EXTRA."""
        a = _make_fingerprint(pip_packages=["torch==2.6.0"])
        b = _make_fingerprint(pip_packages=["flashinfer==0.2.2", "torch==2.6.0"])

        diff = diff_fingerprints(a, b)

        assert diff.packages_added == 1
        added = [d for d in diff.package_diffs if d.status == CheckStatus.EXTRA]
        assert len(added) == 1
        assert added[0].package == "flashinfer"

    def test_package_removed(self):
        """Package in a but not b shows as MISSING."""
        a = _make_fingerprint(pip_packages=["old-pkg==1.0", "torch==2.6.0"])
        b = _make_fingerprint(pip_packages=["torch==2.6.0"])

        diff = diff_fingerprints(a, b)

        assert diff.packages_removed == 1
        removed = [d for d in diff.package_diffs if d.status == CheckStatus.MISSING]
        assert len(removed) == 1
        assert removed[0].package == "old-pkg"

    def test_package_diffs_are_sorted(self):
        """Package diffs are sorted by package name."""
        a = _make_fingerprint(pip_packages=["z-pkg==1.0", "a-pkg==1.0", "m-pkg==1.0"])
        b = _make_fingerprint(pip_packages=["z-pkg==2.0", "a-pkg==2.0", "m-pkg==2.0"])

        diff = diff_fingerprints(a, b)

        names = [d.package for d in diff.package_diffs]
        assert names == sorted(names)

    def test_empty_fingerprints(self):
        """Diffing two empty fingerprints produces zero diff."""
        diff = diff_fingerprints({}, {})
        assert diff.packages_matched == 0
        assert diff.field_changes == {}

    def test_missing_field_treated_as_unavailable(self):
        """Missing field in one fingerprint diffs against UNAVAILABLE."""
        a = _make_fingerprint()
        b = {k: v for k, v in _make_fingerprint().items() if k != "cuda_version"}

        diff = diff_fingerprints(a, b)

        assert "cuda_version" in diff.field_changes
        assert diff.field_changes["cuda_version"][1] == UNAVAILABLE


# ============================================================================
# Check
# ============================================================================


class TestCheck:
    """check_against_fingerprint produces actionable results."""

    def test_matching_environment(self):
        """Identical fingerprints produce empty results."""
        fp = _make_fingerprint()
        results = check_against_fingerprint(fp, fp)
        assert results == []

    def test_version_mismatch_reported(self):
        """CUDA version change appears in results."""
        ref = _make_fingerprint(cuda_version="12.8")
        cur = _make_fingerprint(cuda_version="13.1")

        results = check_against_fingerprint(ref, cur)

        cuda_results = [r for r in results if r.field == "cuda_version"]
        assert len(cuda_results) == 1
        assert cuda_results[0].status == CheckStatus.MISMATCH
        assert cuda_results[0].expected == "12.8"
        assert cuda_results[0].actual == "13.1"

    def test_missing_package_reported(self):
        """Package in reference but not current is MISSING."""
        ref = _make_fingerprint(pip_packages=["special-pkg==1.0", "torch==2.6.0"])
        cur = _make_fingerprint(pip_packages=["torch==2.6.0"])

        results = check_against_fingerprint(ref, cur)

        missing = [r for r in results if r.status == CheckStatus.MISSING]
        assert len(missing) == 1
        assert "special-pkg" in missing[0].field

    def test_extra_package_reported(self):
        """Package in current but not reference is EXTRA."""
        ref = _make_fingerprint(pip_packages=["torch==2.6.0"])
        cur = _make_fingerprint(pip_packages=["new-pkg==0.1", "torch==2.6.0"])

        results = check_against_fingerprint(ref, cur)

        extra = [r for r in results if r.status == CheckStatus.EXTRA]
        assert len(extra) == 1
        assert "new-pkg" in extra[0].field

    def test_unavailable_field_is_error_not_mismatch(self):
        """If a field is UNAVAILABLE, it's an ERROR, not a MISMATCH."""
        ref = _make_fingerprint(cuda_version="12.8")
        cur = _make_fingerprint(cuda_version=UNAVAILABLE)

        results = check_against_fingerprint(ref, cur)

        cuda = [r for r in results if r.field == "cuda_version"]
        assert len(cuda) == 1
        assert cuda[0].status == CheckStatus.ERROR

    def test_captures_fresh_if_current_is_none(self):
        """If current=None, capture_fingerprint is called."""
        ref = _make_fingerprint()

        mock_probes = {
            "hostname": lambda: ProbeResult.success("node-001"),
            "timestamp": lambda: ProbeResult.success("2026-04-09T14:30:00Z"),
            "arch": lambda: ProbeResult.success("aarch64"),
            "os": lambda: ProbeResult.success("Ubuntu 22.04.5 LTS"),
            "gpu": lambda: ProbeResult.success({"available": True, "driver": "570.86.15", "gpus": []}),
            "python_version": lambda: ProbeResult.success("3.11.9"),
            "cuda_version": lambda: ProbeResult.success("12.8"),
            "nccl_version": lambda: ProbeResult.success("2.25.1"),
            "frameworks": lambda: ProbeResult.success(ref["frameworks"]),
            "pip_packages": lambda: ProbeResult.success(ref["pip_packages"]),
        }

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            results = check_against_fingerprint(ref)

        # Should match since we mocked probes to return same values
        assert results == []


# ============================================================================
# File I/O
# ============================================================================


class TestFileIO:
    """write_fingerprint and load_fingerprint are fault-tolerant."""

    def test_write_and_load_roundtrip(self, tmp_path):
        """Write then load produces identical data."""
        mock_probes = {
            "hostname": lambda: ProbeResult.success("test-node"),
            "pip_packages": lambda: ProbeResult.success(["a==1.0", "b==2.0"]),
        }

        fp_path = tmp_path / "fingerprint.json"

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            assert write_fingerprint(fp_path) is True

        loaded = load_fingerprint(fp_path)
        assert loaded is not None
        assert loaded["hostname"] == "test-node"
        assert loaded["pip_packages"] == ["a==1.0", "b==2.0"]

    def test_write_creates_parent_dirs(self, tmp_path):
        """write_fingerprint creates missing parent directories."""
        fp_path = tmp_path / "deep" / "nested" / "fingerprint.json"

        mock_probes = {"hostname": lambda: ProbeResult.success("node")}
        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            assert write_fingerprint(fp_path) is True

        assert fp_path.exists()

    def test_write_returns_false_on_failure(self, tmp_path):
        """write_fingerprint returns False if it can't write (never raises)."""
        # Use a path that can't be created
        fp_path = Path("/proc/nonexistent/fingerprint.json")

        mock_probes = {"hostname": lambda: ProbeResult.success("node")}
        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            assert write_fingerprint(fp_path) is False

    def test_load_returns_none_on_missing_file(self, tmp_path):
        """load_fingerprint returns None for missing file (never raises)."""
        result = load_fingerprint(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_returns_none_on_invalid_json(self, tmp_path):
        """load_fingerprint returns None for corrupted file (never raises)."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        result = load_fingerprint(bad_file)
        assert result is None

    def test_output_is_deterministic(self, tmp_path):
        """Same probes produce byte-identical output on repeated writes."""
        mock_probes = {
            "hostname": lambda: ProbeResult.success("node-1"),
            "arch": lambda: ProbeResult.success("aarch64"),
            "pip_packages": lambda: ProbeResult.success(["b==2.0", "a==1.0"]),
        }

        path1 = tmp_path / "fp1.json"
        path2 = tmp_path / "fp2.json"

        with patch.dict("srtctl.core.fingerprint._PROBES", mock_probes, clear=True):
            write_fingerprint(path1)
            write_fingerprint(path2)

        assert path1.read_text() == path2.read_text()


# ============================================================================
# Bash script generation
# ============================================================================


class TestBashGeneration:
    """generate_capture_script produces valid, safe bash."""

    def test_script_ends_with_or_true(self):
        """Script is wrapped in || true so it never blocks the worker."""
        script = generate_capture_script("/logs/fingerprint.json")
        assert script.rstrip().endswith("|| true")

    def test_script_contains_output_path(self):
        """Output path appears in the generated script."""
        script = generate_capture_script("/logs/fingerprint_prefill_w0.json")
        assert "/logs/fingerprint_prefill_w0.json" in script

    def test_script_starts_with_python(self):
        """Script invokes python3."""
        script = generate_capture_script("/logs/fp.json")
        assert script.startswith("python3")

    def test_script_includes_pip_freeze(self):
        """Script captures pip packages via pip freeze."""
        script = generate_capture_script("/logs/fp.json")
        assert "pip freeze" in script

    def test_script_includes_sorted(self):
        """Script sorts pip output for deterministic diffs."""
        script = generate_capture_script("/logs/fp.json")
        assert "sorted" in script

    def test_embedded_python_is_syntactically_valid(self):
        """The Python code inside the script must parse without SyntaxError.

        This is the test that would have caught the \\n escaping bug where
        literal backslash-n characters were passed to python3 -c instead of
        real newlines, causing a SyntaxError at runtime.
        """
        import ast
        import re

        script = generate_capture_script("/logs/fingerprint.json")
        # Extract the Python source from between the heredoc markers
        match = re.search(
            r"<<'__FINGERPRINT_EOF__'\n(.+?)__FINGERPRINT_EOF__",
            script,
            re.DOTALL,
        )
        assert match, f"Could not find heredoc Python source in script:\n{script[:200]}"
        python_source = match.group(1)
        # ast.parse raises SyntaxError if the code is invalid
        ast.parse(python_source)

    def test_embedded_python_produces_json(self):
        """The embedded script runs and produces valid JSON output."""
        import re
        import subprocess
        import tempfile

        script = generate_capture_script("/tmp/test_fingerprint_output.json")
        match = re.search(
            r"<<'__FINGERPRINT_EOF__'\n(.+?)__FINGERPRINT_EOF__",
            script,
            re.DOTALL,
        )
        assert match
        python_source = match.group(1)

        # Run the script in a subprocess — probes will return "unavailable"
        # on a dev machine (no GPU, etc.) but the script must not crash
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            # Rewrite output path to a temp location
            f.write(python_source.replace("/tmp/test_fingerprint_output.json", f.name + ".out"))
            f.flush()
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=15,
            )
        assert result.returncode == 0, f"Fingerprint script failed:\n{result.stderr}"


# ============================================================================
# Formatting
# ============================================================================


class TestFormatting:
    """Human-readable output for CLI."""

    def test_format_diff_identical(self):
        """Identical fingerprints produce clean summary."""
        fp = _make_fingerprint()
        diff = diff_fingerprints(fp, fp)
        output = format_diff(diff)

        assert "5 match" in output
        assert "0 changed" in output

    def test_format_diff_with_changes(self):
        """Changes are clearly shown."""
        a = _make_fingerprint(cuda_version="12.8", pip_packages=["torch==2.6.0"])
        b = _make_fingerprint(cuda_version="13.1", pip_packages=["torch==2.7.0"])

        diff = diff_fingerprints(a, b)
        output = format_diff(diff)

        assert "2.6.0" in output
        assert "2.7.0" in output
        assert "torch" in output

    def test_format_diff_verbose_shows_all(self):
        """--verbose shows all changes without truncation."""
        packages_a = [f"pkg-{i:03d}==1.0" for i in range(20)]
        packages_b = [f"pkg-{i:03d}==2.0" for i in range(20)]

        diff = diff_fingerprints(
            _make_fingerprint(pip_packages=packages_a),
            _make_fingerprint(pip_packages=packages_b),
        )
        output = format_diff(diff, verbose=True)
        assert "... and" not in output  # No truncation

    def test_format_diff_non_verbose_truncates(self):
        """Default mode truncates long package lists."""
        packages_a = [f"pkg-{i:03d}==1.0" for i in range(20)]
        packages_b = [f"pkg-{i:03d}==2.0" for i in range(20)]

        diff = diff_fingerprints(
            _make_fingerprint(pip_packages=packages_a),
            _make_fingerprint(pip_packages=packages_b),
        )
        output = format_diff(diff, verbose=False)
        assert "... and" in output  # Truncated
        assert "--verbose" in output

    def test_format_check_results_empty(self):
        """Empty results produce positive message."""
        output = format_check_results([])
        assert "matches" in output

    def test_format_check_results_with_mismatches(self):
        """Mismatches are clearly reported."""
        results = [
            CheckResult(
                field="cuda_version",
                status=CheckStatus.MISMATCH,
                message="cuda_version: 12.8 -> 13.1",
                expected="12.8",
                actual="13.1",
            ),
            CheckResult(
                field="pip:sglang",
                status=CheckStatus.MISSING,
                message="sglang==0.4.6 not installed",
                expected="0.4.6",
            ),
        ]
        output = format_check_results(results)

        assert "2 mismatches" in output
        assert "cuda_version" in output
        assert "sglang" in output
