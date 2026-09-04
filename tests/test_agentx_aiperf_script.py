from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_agentx_aiperf_driver_is_self_contained_and_applies_defaults(tmp_path: Path) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    capture = tmp_path / "aiperf-args.txt"
    result_dir = tmp_path / "results"

    _write_executable(
        venv_bin / "python",
        """#!/usr/bin/env bash
cat >/dev/null
echo "Using fake aiperf"
""",
    )
    _write_executable(
        venv_bin / "aiperf",
        """#!/usr/bin/env bash
if [ "${1:-}" = "profile" ] && [ "${2:-}" = "--help" ]; then
    printf '%s\n' \
        '--warmup-grace-period' \
        '--agentic-cache-warmup-duration' \
        '--trace-max-osl' \
        '--max-context-length' \
        '--vllm-start-profile-after-seconds'
    exit 0
fi
printf '%s\n' "$@" > "$AIPERF_TEST_CAPTURE"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "AIPERF_VENV": str(venv_bin.parent),
            "AIPERF_TEST_CAPTURE": str(capture),
            "RESULT_DIR": str(result_dir),
            "MODEL": "deepseek-ai/DeepSeek-V4-Pro",
            "MODEL_PREFIX": "dsv4",
            "FRAMEWORK": "dynamo-vllm",
            "PRECISION": "fp4",
            "CONC": "8",
            "RESULT_FILENAME": "agentx-test",
            "DURATION": "1",
        }
    )
    env.pop("INFMAX_CONTAINER_WORKSPACE", None)
    env.pop("AGENTIC_DIR", None)

    script = Path(__file__).parents[1] / "configs" / "agentx_aiperf.sh"
    subprocess.run(["bash", str(script)], check=True, env=env, capture_output=True, text=True)

    args = capture.read_text().splitlines()
    assert "--agentic-cache-warmup-duration" in args
    assert args[args.index("--agentic-cache-warmup-duration") + 1] == "600"
    assert "--warmup-grace-period" in args
    assert args[args.index("--warmup-grace-period") + 1] == "1800"
    assert "--trace-max-osl" in args
    assert args[args.index("--trace-max-osl") + 1] == "8192"
    assert "--max-context-length" in args
    assert args[args.index("--max-context-length") + 1] == "1000000"
    assert (result_dir / "env.txt").is_file()
    assert (result_dir / "aiperf_command.txt").is_file()
    assert (result_dir / "aiperf.log").is_file()
