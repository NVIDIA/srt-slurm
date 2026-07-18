import os
import subprocess
import sys
from pathlib import Path


PATCHER = (
    Path(__file__).parents[1]
    / "configs"
    / "patches"
    / "vllm_numa_bind_hash_fix.py"
)


def _run(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VLLM_PARALLEL_CONFIG_PATH"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_patches_explicit_parallel_config_path(tmp_path: Path) -> None:
    target = tmp_path / "parallel.py"
    target.write_text(
        "ignored_factors = {\n"
        '            "_api_process_rank",\n'
        "        }\n"
    )

    result = _run(target)

    assert result.returncode == 0
    content = target.read_text()
    assert '"numa_bind",' in content
    assert '"numa_bind_nodes",' in content
    assert '"numa_bind_cpus",' in content


def test_missing_parallel_config_is_nonfatal(tmp_path: Path) -> None:
    result = _run(tmp_path / "missing.py")

    assert result.returncode == 0
    assert "skipping version-specific hotfix" in result.stderr


def test_patch_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "parallel.py"
    target.write_text(
        "ignored_factors = {\n"
        '            "_api_process_rank",\n'
        '            "numa_bind",\n'
        "        }\n"
    )

    result = _run(target)

    assert result.returncode == 0
    assert "Already patched" in result.stderr
