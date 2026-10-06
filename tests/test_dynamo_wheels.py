"""Tests for staged ai-dynamo wheel helpers."""

from pathlib import Path

import pytest

from srtctl.runtime_scripts import dynamo_wheels


def test_detect_target_arch_uses_make_setup_compute_arch(monkeypatch, tmp_path: Path):
    """prefetch derives target arch from bin/uv installed by make setup ARCH=..."""
    source_dir = tmp_path / "source"
    source_dir.joinpath("bin").mkdir(parents=True)
    source_dir.joinpath("bin", "uv").touch()

    monkeypatch.setattr(
        dynamo_wheels,
        "_describe_file",
        lambda _path: "ELF 64-bit LSB executable, ARM aarch64",
    )

    assert dynamo_wheels.detect_target_arch(source_dir, env={}) == "aarch64"


def test_download_command_uses_target_arch(tmp_path: Path):
    """pip download is constrained to the compute architecture."""
    command = dynamo_wheels.build_download_command(
        version="1.2.0.dev20260426",
        wheel_dir=tmp_path,
        arch="aarch64",
        python_version="3.12",
        index_url="https://pypi.org/simple",
        extra_index_url="https://pypi.nvidia.com",
    )

    assert "--platform" in command
    assert "manylinux_2_28_aarch64" in command
    assert "manylinux2014_aarch64" in command
    assert "manylinux_2_28_x86_64" not in command
    assert "--python-version" in command
    assert "3.12" in command


def test_prefetch_does_not_accept_runtime_wheel_for_wrong_arch(monkeypatch, tmp_path: Path):
    """An existing x86 runtime wheel does not satisfy an aarch64 prefetch."""
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    version = "1.2.0.dev20260426"
    (wheel_dir / f"ai_dynamo-{version}-py3-none-any.whl").touch()
    (wheel_dir / f"ai_dynamo_runtime-{version}-cp312-abi3-manylinux_2_28_x86_64.whl").touch()

    calls = []

    def fake_run(command, check):
        assert check is True
        calls.append(command)
        (wheel_dir / f"ai_dynamo_runtime-{version}-cp312-abi3-manylinux_2_28_aarch64.whl").touch()

    monkeypatch.setattr(dynamo_wheels.subprocess, "run", fake_run)

    dynamo_wheels.prefetch(
        env={
            "DYNAMO_VERSION": version,
            "DYNAMO_WHEEL_ARCH": "aarch64",
            "DYNAMO_WHEEL_HOST_DIR": str(wheel_dir),
        }
    )

    assert calls
    assert "manylinux_2_28_aarch64" in calls[0]
    assert (wheel_dir / f"ai_dynamo_runtime-{version}-cp312-abi3-manylinux_2_28_aarch64.whl").exists()


def test_install_requires_runtime_wheel_for_compute_arch(monkeypatch, tmp_path: Path):
    """Install rejects a staged runtime wheel for the wrong compute architecture."""
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    version = "1.2.0.dev20260426"
    (wheel_dir / f"ai_dynamo-{version}-py3-none-any.whl").touch()
    (wheel_dir / f"ai_dynamo_runtime-{version}-cp312-abi3-manylinux_2_28_x86_64.whl").touch()

    monkeypatch.setattr(dynamo_wheels, "_already_installed", lambda _version: False)

    with pytest.raises(SystemExit, match="aarch64"):
        dynamo_wheels.install(
            env={
                "DYNAMO_VERSION": version,
                "SRTCTL_COMPUTE_ARCH": "aarch64",
                "DYNAMO_WHEEL_DIRS": str(wheel_dir),
            }
        )


@pytest.mark.parametrize(
    ("index_env", "index_url", "extra_index_url"),
    [
        ({}, "https://pypi.org/simple", "https://pypi.nvidia.com"),
        (
            {"DYNAMO_INDEX_URL": "https://packages.example/simple", "DYNAMO_EXTRA_INDEX_URL": "https://gpu.example"},
            "https://packages.example/simple",
            "https://gpu.example",
        ),
    ],
)
def test_install_resolves_dependencies_with_exact_staged_pins(
    monkeypatch, tmp_path: Path, index_env: dict[str, str], index_url: str, extra_index_url: str
):
    """Staged installs allow pip to fetch dependencies and honor index overrides."""
    version = "1.2.0.dev20260426"
    (tmp_path / f"ai_dynamo-{version}-py3-none-any.whl").touch()
    (tmp_path / f"ai_dynamo_runtime-{version}-cp312-abi3-manylinux_2_28_aarch64.whl").touch()
    calls = []
    imports = []

    def fake_run(command, check):
        assert check is True
        calls.append(command)

    monkeypatch.setattr(dynamo_wheels, "_already_installed", lambda _version: False)
    monkeypatch.setattr(dynamo_wheels.subprocess, "run", fake_run)
    monkeypatch.setattr(dynamo_wheels.importlib, "import_module", imports.append)

    dynamo_wheels.install(
        env={
            "DYNAMO_VERSION": version,
            "DYNAMO_WHEEL_ARCH": "aarch64",
            "DYNAMO_WHEEL_DIRS": str(tmp_path),
            **index_env,
        }
    )

    assert len(calls) == 1
    command = calls[0]
    assert "--no-deps" not in command
    assert "--no-index" not in command
    assert command[command.index("--index-url") + 1] == index_url
    assert command[command.index("--extra-index-url") + 1] == extra_index_url
    assert command[command.index("--find-links") + 1] == str(tmp_path)
    assert "--pre" in command
    assert command[-2:] == [f"ai-dynamo-runtime=={version}", f"ai-dynamo=={version}"]
    assert imports == ["dynamo.llm"]
