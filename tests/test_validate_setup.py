# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for validate_setup pre-flight check and Makefile arch detection."""

import subprocess
from pathlib import Path

import pytest

from srtctl.cli.submit import validate_setup


class TestValidateSetup:
    """Tests for the validate_setup function."""

    def test_passes_when_all_binaries_exist(self, tmp_path: Path):
        """validate_setup succeeds when all required binaries are present."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()

        # Should not raise
        validate_setup(tmp_path)

    def test_fails_when_nats_missing(self, tmp_path: Path):
        """validate_setup fails when nats-server is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_etcd_missing(self, tmp_path: Path):
        """validate_setup fails when etcd is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_uv_missing(self, tmp_path: Path):
        """validate_setup fails when bin/uv is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "configs" / "etcd").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_all_missing(self, tmp_path: Path):
        """validate_setup fails when nothing has been set up."""
        with pytest.raises(SystemExit):
            validate_setup(tmp_path)


class TestMakefileArchDetection:
    """Test that the file | grep pattern used in Makefile matches correctly.

    The Makefile uses `file <binary> | grep -q "$ARCH_FILE_PATTERN"` to check
    if an existing binary matches the requested architecture. These tests verify
    the pattern works by creating minimal ELF binaries and checking `file` output.
    """

    # Minimal ELF headers: just enough for `file` to identify the architecture
    # ELF magic + class(64-bit) + data(little-endian) + version + OS/ABI + padding + type + machine
    ELF_X86_64 = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\x3e\x00"
    ELF_AARCH64 = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\xb7\x00"

    @staticmethod
    def _file_description(path: Path) -> str:
        """Get just the description part of file(1) output (after the colon)."""
        result = subprocess.run(["file", str(path)], capture_output=True, text=True)
        return result.stdout.split(":", 1)[1] if ":" in result.stdout else result.stdout

    def test_file_detects_x86_64(self, tmp_path: Path):
        """file(1) description for x86_64 ELF contains 'x86-64' (hyphen, not underscore)."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(self.ELF_X86_64 + b"\x00" * 44)
        desc = self._file_description(binary)
        assert "x86-64" in desc, f"Expected 'x86-64' in: {desc}"
        assert "x86_64" not in desc, f"file uses hyphen not underscore: {desc}"

    def test_file_detects_aarch64(self, tmp_path: Path):
        """file(1) description for aarch64 ELF contains 'aarch64'."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(self.ELF_AARCH64 + b"\x00" * 44)
        desc = self._file_description(binary)
        assert "aarch64" in desc, f"Expected 'aarch64' in: {desc}"

    def test_x86_64_not_matched_by_aarch64_pattern(self, tmp_path: Path):
        """An x86_64 binary must not match the aarch64 pattern."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(self.ELF_X86_64 + b"\x00" * 44)
        desc = self._file_description(binary)
        assert "aarch64" not in desc

    def test_aarch64_not_matched_by_x86_pattern(self, tmp_path: Path):
        """An aarch64 binary must not match the x86-64 pattern."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(self.ELF_AARCH64 + b"\x00" * 44)
        desc = self._file_description(binary)
        assert "x86-64" not in desc
