# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for validate_setup pre-flight check and Makefile arch detection."""

import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from srtctl.cli.submit import validate_setup
from srtctl.core.schema import (
    BenchmarkConfig,
    CpuPowerConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
)

# Minimal ELF headers: just enough for `file` to identify the architecture
# ELF magic + class(64-bit) + data(little-endian) + version + OS/ABI + padding + type + machine
ELF_X86_64 = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\x3e\x00" + b"\x00" * 44
ELF_AARCH64 = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\xb7\x00" + b"\x00" * 44


def _install(path: Path, image: bytes) -> None:
    """Write an executable binary of a known architecture, as make setup does."""
    path.write_bytes(image)
    path.chmod(0o755)


class TestValidateSetup:
    """Tests for the validate_setup function."""

    def test_passes_when_all_binaries_exist(self, tmp_path: Path):
        """validate_setup succeeds when all required binaries are present."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()
        (tmp_path / "bin" / "tachometer-scraper").touch()

        # Should not raise
        validate_setup(tmp_path)

    def test_fails_when_nats_missing(self, tmp_path: Path):
        """validate_setup fails when nats-server is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()
        (tmp_path / "bin" / "tachometer-scraper").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_etcd_missing(self, tmp_path: Path):
        """validate_setup fails when etcd is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()
        (tmp_path / "bin" / "tachometer-scraper").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_uv_missing(self, tmp_path: Path):
        """validate_setup fails when bin/uv is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "tachometer-scraper").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_tachometer_scraper_missing(self, tmp_path: Path):
        """validate_setup fails when the compute-architecture scraper is missing."""
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()

        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    def test_fails_when_all_missing(self, tmp_path: Path):
        """validate_setup fails when nothing has been set up."""
        with pytest.raises(SystemExit):
            validate_setup(tmp_path)

    @staticmethod
    def _setup_tree(tmp_path: Path) -> None:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "nats-server").touch()
        (tmp_path / "configs" / "etcd").touch()
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").touch()
        (tmp_path / "bin" / "tachometer-scraper").touch()

    @staticmethod
    def _config(cpu_power_enabled: bool) -> SrtConfig:
        return SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="manual"),
            telemetry=TelemetryConfig(
                enabled=cpu_power_enabled,
                cpu_power=CpuPowerConfig(enabled=cpu_power_enabled),
            ),
        )

    def test_cpu_power_exporter_is_not_required_when_disabled(self, tmp_path: Path):
        """Recipes that never scrape ACPI rails must not need the exporter."""
        self._setup_tree(tmp_path)

        validate_setup(tmp_path, self._config(cpu_power_enabled=False))

    def test_cpu_power_exporter_is_required_when_enabled(self, tmp_path: Path):
        """A recipe that enables CPU power cannot run without the exporter."""
        self._setup_tree(tmp_path)

        with pytest.raises(SystemExit):
            validate_setup(tmp_path, self._config(cpu_power_enabled=True))

    @staticmethod
    def _foreign_arch() -> tuple[bytes, str]:
        """An architecture that is deliberately not this host's.

        make setup targets the compute nodes, which are routinely a different
        machine than the one submitting, so the check must follow bin/uv.
        """
        if platform.machine() == "aarch64":
            return ELF_X86_64, "x86_64"
        return ELF_AARCH64, "aarch64"

    def test_cpu_power_enabled_passes_once_the_exporter_is_installed(self, tmp_path: Path):
        """The compute architecture is bin/uv's, not the submit host's."""
        if shutil.which("file") is None:
            pytest.skip("file(1) is not installed")
        self._setup_tree(tmp_path)
        image, _arch = self._foreign_arch()
        _install(tmp_path / "bin" / "uv", image)
        _install(tmp_path / "bin" / "cpu-power-exporter", image)

        validate_setup(tmp_path, self._config(cpu_power_enabled=True))

    def test_an_exporter_that_cannot_be_executed_is_rejected(self, tmp_path: Path):
        """A partial download leaves a file srun cannot start mid-allocation."""
        self._setup_tree(tmp_path)
        exporter = tmp_path / "bin" / "cpu-power-exporter"
        exporter.write_bytes(ELF_X86_64)
        exporter.chmod(0o644)

        with pytest.raises(SystemExit):
            validate_setup(tmp_path, self._config(cpu_power_enabled=True))

    def test_an_exporter_built_for_another_architecture_is_rejected(self, tmp_path: Path):
        """A checkout carried between clusters keeps a binary that cannot run."""
        if shutil.which("file") is None:
            pytest.skip("file(1) is not installed")
        self._setup_tree(tmp_path)
        compute_image, _arch = self._foreign_arch()
        other = ELF_AARCH64 if compute_image is ELF_X86_64 else ELF_X86_64
        _install(tmp_path / "bin" / "uv", compute_image)
        _install(tmp_path / "bin" / "cpu-power-exporter", other)

        with pytest.raises(SystemExit):
            validate_setup(tmp_path, self._config(cpu_power_enabled=True))


class TestMakefileArchDetection:
    """Test that the file | grep pattern used in Makefile matches correctly.

    The Makefile uses `file <binary> | grep -q "$ARCH_FILE_PATTERN"` to check
    if an existing binary matches the requested architecture. These tests verify
    the pattern works by creating minimal ELF binaries and checking `file` output.
    """

    @staticmethod
    def _file_description(path: Path) -> str:
        """Get just the description part of file(1) output (after the colon).

        These tests assert what file(1) says, so a host without it has nothing
        to assert against -- skipping is the honest answer, and the assertions
        below are untouched wherever it is installed.
        """
        if shutil.which("file") is None:
            pytest.skip("file(1) is not installed")
        result = subprocess.run(["file", str(path)], capture_output=True, text=True, check=False)
        return result.stdout.split(":", 1)[1] if ":" in result.stdout else result.stdout

    def test_file_detects_x86_64(self, tmp_path: Path):
        """file(1) description for x86_64 ELF contains 'x86-64' (hyphen, not underscore)."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(ELF_X86_64)
        desc = self._file_description(binary)
        assert "x86-64" in desc, f"Expected 'x86-64' in: {desc}"
        assert "x86_64" not in desc, f"file uses hyphen not underscore: {desc}"

    def test_file_detects_aarch64(self, tmp_path: Path):
        """file(1) description for aarch64 ELF contains 'aarch64'."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(ELF_AARCH64)
        desc = self._file_description(binary)
        assert "aarch64" in desc, f"Expected 'aarch64' in: {desc}"

    def test_x86_64_not_matched_by_aarch64_pattern(self, tmp_path: Path):
        """An x86_64 binary must not match the aarch64 pattern."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(ELF_X86_64)
        desc = self._file_description(binary)
        assert "aarch64" not in desc

    def test_aarch64_not_matched_by_x86_pattern(self, tmp_path: Path):
        """An aarch64 binary must not match the x86-64 pattern."""
        binary = tmp_path / "fake_bin"
        binary.write_bytes(ELF_AARCH64)
        desc = self._file_description(binary)
        assert "x86-64" not in desc
