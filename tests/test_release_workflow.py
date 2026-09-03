# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for release carry-forward and the optional cpu-power-exporter download.

The carry-forward and publish guards live in shell inside
``.github/workflows/release.yaml``. The tests below lift those exact scripts out
of the workflow and run them against a stub ``gh``, so the behaviour under test
is the one that actually ships.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yaml"

TACHOMETER_ASSETS = [
    "tachometer-scraper-x86_64-unknown-linux-gnu",
    "tachometer-scraper-aarch64-unknown-linux-gnu",
]
CPU_ASSETS = [
    "cpu-power-exporter-x86_64-unknown-linux-musl",
    "cpu-power-exporter-aarch64-unknown-linux-musl",
]


def _complete(assets: list[str]) -> list[str]:
    return [name for asset in assets for name in (asset, f"{asset}.sha256")]


def _step_script(step_name: str) -> str:
    """The `run:` body of a named step in the release job."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for step in workflow["jobs"]["release"]["steps"]:
        if step.get("name") == step_name:
            return step["run"]
    raise AssertionError(f"no step named {step_name!r} in {WORKFLOW}")


def _stub_gh(bin_dir: Path, previous: str, assets: list[str]) -> None:
    """A `gh` that reports one previous release holding `assets`.

    An empty `previous` means no release exists; `assets` of `["<fail>"]` makes
    `gh release view` fail the way a transient API error would.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "gh"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"previous = {previous!r}\n"
        f"assets = {assets!r}\n"
        "argv = sys.argv[1:]\n"
        "if argv[:2] == ['release', 'list']:\n"
        "    print(previous) if previous else None\n"
        "elif argv[:2] == ['release', 'view']:\n"
        "    if assets == ['<fail>']:\n"
        "        sys.exit(1)\n"
        "    print('\\n'.join(assets))\n"
        "elif argv[:2] == ['release', 'download']:\n"
        "    prefix = argv[argv.index('--pattern') + 1].rstrip('*')\n"
        "    dest = argv[argv.index('--dir') + 1]\n"
        "    os.makedirs(dest, exist_ok=True)\n"
        "    for name in assets:\n"
        "        if name.startswith(prefix):\n"
        "            open(os.path.join(dest, name), 'w').close()\n"
        "elif argv[:2] == ['release', 'create']:\n"
        "    pass\n"
        "else:\n"
        "    sys.exit('unexpected gh invocation: ' + ' '.join(argv))\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _run(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        env={**os.environ, **env, "PATH": f"{cwd / 'stub-bin'}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )


class TestCarryForward:
    """The previous release must hold the whole set, or none of it."""

    SCRIPT_NAME = "Carry forward unchanged binaries from the previous release"

    def _carry(self, tmp_path: Path, assets: list[str], previous: str = "v1.2.3"):
        _stub_gh(tmp_path / "stub-bin", previous, assets)
        return _run(
            _step_script(self.SCRIPT_NAME),
            tmp_path,
            {"TACHOMETER_CHANGED": "true", "CPU_POWER_EXPORTER_CHANGED": "false"},
        )

    def test_a_complete_previous_set_is_carried_forward(self, tmp_path: Path):
        result = self._carry(tmp_path, _complete(TACHOMETER_ASSETS) + _complete(CPU_ASSETS))
        assert result.returncode == 0, result.stderr
        for name in _complete(CPU_ASSETS):
            assert (tmp_path / "dist" / name).exists(), f"{name} was not carried forward"

    def test_a_partial_previous_set_fails_the_release(self, tmp_path: Path):
        # The x86_64 half is intact; aarch64 lost its binary and checksum.
        result = self._carry(tmp_path, _complete(CPU_ASSETS)[:2])
        assert result.returncode != 0
        assert "missing cpu-power-exporter assets" in result.stdout

    def test_a_previous_set_missing_only_a_checksum_fails_the_release(self, tmp_path: Path):
        result = self._carry(tmp_path, _complete(CPU_ASSETS)[:-1])
        assert result.returncode != 0
        assert "cpu-power-exporter-aarch64-unknown-linux-musl.sha256" in result.stdout

    def test_a_binary_the_previous_release_never_had_is_only_a_warning(self, tmp_path: Path):
        result = self._carry(tmp_path, _complete(TACHOMETER_ASSETS))
        assert result.returncode == 0, result.stderr
        assert "::warning::Previous release v1.2.3 has no cpu-power-exporter assets" in result.stdout
        assert list((tmp_path / "dist").glob("cpu-power-exporter-*")) == []

    def test_the_first_release_has_nothing_to_carry(self, tmp_path: Path):
        result = self._carry(tmp_path, [], previous="")
        assert result.returncode == 0, result.stderr
        assert "No previous release to carry cpu-power-exporter forward from" in result.stdout

    def test_an_api_failure_is_not_treated_as_an_absent_binary(self, tmp_path: Path):
        result = self._carry(tmp_path, ["<fail>"])
        assert result.returncode != 0
        assert "Could not list assets of previous release" in result.stdout


class TestPublishGuard:
    """A release ships a binary for both architectures or not at all."""

    SCRIPT_NAME = "Create release"

    def _publish(self, tmp_path: Path, dist: list[str]):
        _stub_gh(tmp_path / "stub-bin", "v1.2.3", [])
        (tmp_path / "dist").mkdir()
        for name in dist:
            (tmp_path / "dist" / name).touch()
        return _run(_step_script(self.SCRIPT_NAME), tmp_path, {"TAG": "v1.2.4", "TARGET": "abc123"})

    def test_a_complete_set_publishes(self, tmp_path: Path):
        result = self._publish(tmp_path, _complete(TACHOMETER_ASSETS) + _complete(CPU_ASSETS))
        assert result.returncode == 0, result.stdout + result.stderr

    def test_one_architecture_alone_does_not_publish(self, tmp_path: Path):
        result = self._publish(tmp_path, _complete(TACHOMETER_ASSETS) + _complete(CPU_ASSETS)[:2])
        assert result.returncode != 0
        assert "would ship an incomplete cpu-power-exporter set" in result.stdout

    def test_a_binary_absent_from_the_release_entirely_publishes(self, tmp_path: Path):
        result = self._publish(tmp_path, _complete(TACHOMETER_ASSETS))
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
class TestOptionalExporterDownload:
    """cpu-power-exporter is optional by default, but a pinned release is not."""

    @staticmethod
    def _sandbox(tmp_path: Path) -> Path:
        """A copy of the Makefile with its own bin/, so the checkout is untouched."""
        shutil.copy(REPO_ROOT / "Makefile", tmp_path / "Makefile")
        stub_bin = tmp_path / "stub-bin"
        stub_bin.mkdir()
        # A curl that always fails stands in for a release without the asset.
        (stub_bin / "curl").write_text("#!/bin/sh\nexit 22\n")
        # `file` is not installed everywhere, and the answer it would give for
        # the placeholder binary below is fixed anyway.
        (stub_bin / "file").write_text('#!/bin/sh\necho "$1: ELF 64-bit LSB executable, x86-64"\n')
        for stub in stub_bin.iterdir():
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        return tmp_path

    def _make(self, target: str, tmp_path: Path, release: str = "latest") -> subprocess.CompletedProcess:
        sandbox = tmp_path if (tmp_path / "Makefile").exists() else self._sandbox(tmp_path)
        return subprocess.run(
            ["make", "-C", str(sandbox), target, f"CPU_POWER_EXPORTER_RELEASE={release}", "ARCH=x86_64"],
            env={**os.environ, "PATH": f"{sandbox / 'stub-bin'}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _install(sandbox: Path, release: str) -> None:
        """Pretend a previous run installed `release`."""
        (sandbox / "bin").mkdir(exist_ok=True)
        (sandbox / "bin" / "cpu-power-exporter").write_text("stale binary\n")
        (sandbox / "bin" / ".cpu-power-exporter.release").write_text(release)

    def test_setup_warns_instead_of_failing_on_the_default_release(self, tmp_path: Path):
        result = self._make("cpu-power-exporter-setup", tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cpu-power-exporter unavailable" in result.stdout

    def test_setup_fails_when_a_release_was_pinned(self, tmp_path: Path):
        """The pin exists to get one specific version; a warning would defeat it."""
        result = self._make("cpu-power-exporter-setup", tmp_path, release="v0.0.0-absent")
        assert result.returncode != 0
        assert "cpu-power-exporter unavailable" not in result.stdout

    def test_a_failed_pinned_download_leaves_no_stale_binary(self, tmp_path: Path):
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")

        result = self._make("cpu-power-exporter-setup", sandbox, release="v0.0.0-absent")

        assert result.returncode != 0
        assert not (sandbox / "bin" / "cpu-power-exporter").exists(), (
            "validate-setup only checks existence, so a leftover binary would pass as the pinned one"
        )

    def test_the_default_release_keeps_an_installed_binary(self, tmp_path: Path):
        """`latest` has no identity to compare against, so it must not churn."""
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")

        result = self._make("cpu-power-exporter-setup", sandbox)

        assert result.returncode == 0, result.stdout + result.stderr
        assert (sandbox / "bin" / "cpu-power-exporter").exists()

    def test_an_explicit_download_still_fails(self, tmp_path: Path):
        result = self._make("cpu-power-exporter-download", tmp_path)
        assert result.returncode != 0

    @staticmethod
    def _stub_cargo(sandbox: Path) -> None:
        """A cargo that produces a binary without a toolchain or a network."""
        stub = sandbox / "stub-bin" / "cargo"
        stub.write_text("#!/bin/sh\nmkdir -p target/release\nprintf 'built' > target/release/cpu-power-exporter\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    def test_a_local_build_invalidates_the_release_marker(self, tmp_path: Path):
        """A working-tree build is not the released tag the marker names."""
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")
        self._stub_cargo(sandbox)

        result = self._make("cpu-power-exporter", sandbox)

        assert result.returncode == 0, result.stdout + result.stderr
        assert (sandbox / "bin" / "cpu-power-exporter").read_text() == "built"
        assert not (sandbox / "bin" / ".cpu-power-exporter.release").exists()

    def test_a_pinned_download_after_a_local_build_is_not_skipped(self, tmp_path: Path):
        """Otherwise the pin silently keeps whatever the working tree happened to build."""
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")
        self._stub_cargo(sandbox)
        assert self._make("cpu-power-exporter", sandbox).returncode == 0

        result = self._make("cpu-power-exporter-download", sandbox, release="v1.0.0")

        assert result.returncode != 0, "the download was skipped as already installed"
        assert "already installed" not in result.stdout

    @staticmethod
    def _path_without_file(sandbox: Path) -> str:
        """A PATH holding what the recipe runs, minus the `file` it inspects with."""
        tools = sandbox / "no-file-bin"
        tools.mkdir(exist_ok=True)
        for name in ("make", "sh", "rm", "cat", "grep", "mktemp", "install", "sha256sum", "cp", "chmod"):
            real = shutil.which(name)
            assert real is not None, f"{name} is needed to run the download recipe"
            (tools / name).symlink_to(real)
        return f"{sandbox / 'stub-bin'}:{tools}"

    def test_an_unverifiable_architecture_does_not_delete_the_binary(self, tmp_path: Path):
        """A missing file(1) is not evidence that the installed binary is wrong.

        Deleting it and then failing to fetch a replacement -- which the
        warn-on-failure path does quietly -- would leave no exporter at all.
        """
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")
        (sandbox / "stub-bin" / "file").unlink()

        result = subprocess.run(
            ["make", "-C", str(sandbox), "cpu-power-exporter-download", "ARCH=x86_64"],
            env={**os.environ, "PATH": self._path_without_file(sandbox)},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0, "the stub curl cannot fetch anything"
        assert "file(1) is not installed" in result.stdout
        assert "not a x86_64 binary" not in result.stdout
        assert (sandbox / "bin" / "cpu-power-exporter").exists()

    def test_an_unverifiable_architecture_is_not_reported_as_installed(self, tmp_path: Path):
        """Skipping would report an unchecked binary as the pinned one."""
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")
        (sandbox / "stub-bin" / "file").unlink()

        result = subprocess.run(
            [
                "make",
                "-C",
                str(sandbox),
                "cpu-power-exporter-download",
                "ARCH=x86_64",
                "CPU_POWER_EXPORTER_RELEASE=v1.0.0",
            ],
            env={**os.environ, "PATH": self._path_without_file(sandbox)},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0, "a pin that cannot be verified or fetched must fail loudly"
        assert "already installed" not in result.stdout

    @staticmethod
    def _report_arch(sandbox: Path, arch: str) -> None:
        """Make the `file` stub report `arch` for whatever it is asked about."""
        stub = sandbox / "stub-bin" / "file"
        stub.write_text(f'#!/bin/sh\necho "$1: ELF 64-bit LSB executable, {arch}"\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    def test_the_default_release_drops_a_wrong_architecture_binary(self, tmp_path: Path):
        """A checkout reused across architectures must not keep the other one's binary.

        The warn-on-failure path would otherwise leave a binary that cannot run
        on this host, and validate-setup accepts it because it only checks that
        the file is there.
        """
        sandbox = self._sandbox(tmp_path)
        self._install(sandbox, "v1.0.0")
        self._report_arch(sandbox, "aarch64")

        result = self._make("cpu-power-exporter-setup", sandbox)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "cpu-power-exporter unavailable" in result.stdout
        assert not (sandbox / "bin" / "cpu-power-exporter").exists()
        assert not (sandbox / "bin" / ".cpu-power-exporter.release").exists()

    def test_the_scraper_download_drops_a_wrong_architecture_binary(self, tmp_path: Path):
        sandbox = self._sandbox(tmp_path)
        (sandbox / "bin").mkdir(exist_ok=True)
        (sandbox / "bin" / "tachometer-scraper").write_text("stale binary\n")
        self._report_arch(sandbox, "aarch64")

        result = self._make("tachometer-scraper-download", sandbox)

        assert result.returncode != 0
        assert not (sandbox / "bin" / "tachometer-scraper").exists()
