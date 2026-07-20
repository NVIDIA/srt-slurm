from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from srtctl.cli.mixins.host_memory_stage import (
    HostMemoryStageMixin,
    host_memory_telemetry_script,
    stale_vllm_offload_cleanup_script,
)


def test_cleanup_script_is_fail_closed_and_owner_scoped() -> None:
    script = stale_vllm_offload_cleanup_script(5)
    assert "command -v fuser" in script
    assert "refusing to delete" in script
    assert '-uid "$(id -u)"' in script
    assert "-name 'vllm_offload_*.mmap'" in script
    assert "-mmin +5" in script
    assert 'fuser -s -- "$path"' in script
    assert 'rm -f -- "$path"' in script


def test_telemetry_script_captures_required_signals() -> None:
    script = host_memory_telemetry_script(30)
    for signal in (
        "MemAvailable",
        "Mlocked",
        "Unevictable",
        "memory.current",
        "memory.max",
        "memory.events",
        "ps -eo",
        "sstat -j",
        "df -B1 /dev/shm",
    ):
        assert signal in script


@patch("srtctl.cli.mixins.host_memory_stage.start_srun_process")
def test_prepare_runs_cleanup_then_noncritical_sampler(mock_srun: MagicMock, tmp_path: Path) -> None:
    cleanup_proc = MagicMock()
    cleanup_proc.wait.return_value = 0
    telemetry_proc = MagicMock()
    mock_srun.side_effect = [cleanup_proc, telemetry_proc]

    class Harness(HostMemoryStageMixin):
        pass

    harness = Harness()
    harness.config = SimpleNamespace()
    harness.runtime = SimpleNamespace(
        environment={
            "SRTCTL_CLEAN_STALE_VLLM_OFFLOAD": "1",
            "SRTCTL_HOST_MEMORY_DIAGNOSTICS": "1",
        },
        nodes=SimpleNamespace(
            het=True,
            prefill_group=("p0",),
            worker=("p0", "d0"),
        ),
        log_dir=tmp_path,
        srun_options={},
    )

    managed = harness.prepare_host_memory_diagnostics()

    assert mock_srun.call_count == 2
    assert mock_srun.call_args_list[0].kwargs["nodelist"] == ["p0"]
    assert mock_srun.call_args_list[0].kwargs["het_group"] == 0
    cleanup_proc.wait.assert_called_once_with(timeout=300)
    assert len(managed) == 1
    assert managed[0].critical is False
    assert managed[0].popen is telemetry_proc
