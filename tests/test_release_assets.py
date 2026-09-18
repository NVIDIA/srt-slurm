# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The release job refuses to publish a partial binary set.

v2.7.0 shipped the x86_64 tachometer-scraper ``.sha256`` without the binary because a
reset connection during the carry-forward download was only warned about; v2.8.0 through
v2.9.1 then carried that gap forward. ``verify_release_assets.sh`` is the gate that makes
the release job fail instead.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "verify_release_assets.sh"
TOOLS = ("tachometer-scraper", "cpu-power-exporter")
TARGETS = {"tachometer-scraper": "unknown-linux-gnu", "cpu-power-exporter": "unknown-linux-musl"}
ARCHES = ("x86_64", "aarch64")


def _asset(tool: str, arch: str) -> str:
    return f"{tool}-{arch}-{TARGETS[tool]}"


def _publish(dist: Path, tool: str, arch: str, *, binary: bool = True, sha256: bool = True) -> None:
    name = _asset(tool, arch)
    payload = f"{name} bytes".encode()
    if binary:
        (dist / name).write_bytes(payload)
    if sha256:
        digest = hashlib.sha256(payload).hexdigest()
        (dist / f"{name}.sha256").write_text(f"{digest}  {name}\n")


def _full_set(dist: Path, *tools: str) -> None:
    for tool in tools:
        for arch in ARCHES:
            _publish(dist, tool, arch)


def _verify(dist: Path, *tools: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), str(dist), *(tools or TOOLS)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_complete_set_passes(tmp_path: Path) -> None:
    _full_set(tmp_path, *TOOLS)
    result = _verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for tool in TOOLS:
        for arch in ARCHES:
            assert f"{tool} {arch}: {_asset(tool, arch)} ok" in result.stdout
    assert "::error::" not in result.stdout
    assert "::warning::" not in result.stdout


def test_tool_with_no_assets_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    # The first release has nothing to carry forward; a binary added later is absent from older releases.
    _full_set(tmp_path, "tachometer-scraper")
    result = _verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::No cpu-power-exporter assets" in result.stdout


def test_empty_dist_warns_for_every_tool(tmp_path: Path) -> None:
    result = _verify(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for tool in TOOLS:
        assert f"::warning::No {tool} assets" in result.stdout


def test_sha256_without_binary_fails(tmp_path: Path) -> None:
    # The v2.7.0 failure: the 71 MB x86_64 download was reset, the 110-byte .sha256 landed.
    _full_set(tmp_path, "cpu-power-exporter")
    _publish(tmp_path, "tachometer-scraper", "aarch64")
    _publish(tmp_path, "tachometer-scraper", "x86_64", binary=False)
    result = _verify(tmp_path)
    assert result.returncode == 1
    assert "::error::tachometer-scraper x86_64: expected one binary and one .sha256" in result.stdout
    assert "found 0 binary and 1 .sha256" in result.stdout
    # The healthy tool is still reported so the log shows what did make it.
    assert "cpu-power-exporter x86_64" in result.stdout


def test_binary_without_sha256_fails(tmp_path: Path) -> None:
    _full_set(tmp_path, "cpu-power-exporter")
    _publish(tmp_path, "tachometer-scraper", "aarch64")
    _publish(tmp_path, "tachometer-scraper", "x86_64", sha256=False)
    result = _verify(tmp_path)
    assert result.returncode == 1
    assert "found 1 binary and 0 .sha256" in result.stdout


def test_missing_architecture_fails(tmp_path: Path) -> None:
    # A reset before either x86_64 file landed leaves only the aarch64 pair: pairing alone would pass that.
    _full_set(tmp_path, "cpu-power-exporter")
    _publish(tmp_path, "tachometer-scraper", "aarch64")
    result = _verify(tmp_path)
    assert result.returncode == 1
    assert "::error::tachometer-scraper x86_64: expected one binary and one .sha256" in result.stdout
    assert "found 0 binary and 0 .sha256" in result.stdout


def test_checksum_mismatch_fails(tmp_path: Path) -> None:
    _full_set(tmp_path, *TOOLS)
    (tmp_path / _asset("tachometer-scraper", "x86_64")).write_bytes(b"truncated")
    result = _verify(tmp_path)
    assert result.returncode == 1
    assert "::error::tachometer-scraper x86_64: checksum mismatch" in result.stdout


def test_one_bad_tool_does_not_hide_the_other(tmp_path: Path) -> None:
    _full_set(tmp_path, "tachometer-scraper")
    _publish(tmp_path, "cpu-power-exporter", "x86_64")
    _publish(tmp_path, "cpu-power-exporter", "aarch64", binary=False)
    result = _verify(tmp_path)
    assert result.returncode == 1
    assert result.stdout.count("::error::") == 1
    assert "::error::cpu-power-exporter aarch64" in result.stdout
    assert "tachometer-scraper x86_64: " in result.stdout
    assert "tachometer-scraper aarch64: " in result.stdout


@pytest.mark.parametrize("argv", [[], ["dist-only"]])
def test_usage_error_without_tools(tmp_path: Path, argv: list[str]) -> None:
    args = [str(tmp_path)] if argv else []
    result = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_missing_dist_directory_fails(tmp_path: Path) -> None:
    result = _verify(tmp_path / "does-not-exist")
    assert result.returncode == 1
    assert "::error::" in result.stdout
