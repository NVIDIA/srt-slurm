from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "configs" / "3282821-lyris-f946-mooncake-rank-local-numaint-power-r3-20260830" / "setup.sh"


def test_setup_is_valid_bash_and_combines_only_required_runtime_patches() -> None:
    subprocess.run(["bash", "-n", str(SETUP)], check=True)
    content = SETUP.read_text()

    assert "/configs/patches/vllm_numa_interleave.py" in content
    assert "apply_f946_mooncake_rank_local_hca_patch.py" in content
    assert "install-mooncake-store-0312.sh" in content
    assert "numactl --interleave=0,1 true" in content
    assert "Mooncake 0.3.12.post1" in content
