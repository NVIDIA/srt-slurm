# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess


OLD_BLOCK = '''        use_msa_decode = self.impl.should_use_msa_decode(self.layer_name)
        query_fp8 = (
            torch.empty(
                (num_tokens, self.q_size),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )
            if use_msa_decode
            else None
        )
'''


def test_query_fp8_setup_script_is_strict_and_idempotent(tmp_path: Path):
    vllm_root = tmp_path / "vllm"
    target = vllm_root / "models/minimax_m3/nvidia/model.py"
    target.parent.mkdir(parents=True)
    target.write_text(OLD_BLOCK)
    (vllm_root / "__init__.py").write_text('__version__ = "test-version"\n')

    script = (
        Path(__file__).parents[1]
        / "configs/patches/patch-minimax-m3-query-fp8-graph-stability.sh"
    )
    env = os.environ | {"PYTHONPATH": str(tmp_path)}

    first = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True, check=True
    )
    second = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True, check=True
    )

    patched = target.read_text()
    assert 'getattr(self.impl, "use_cutlass_decode", False)' in patched
    assert "use_msa_decode =" not in patched
    assert "MINIMAX_QUERY_FP8_PATCH=applied" in first.stdout
    assert "MINIMAX_QUERY_FP8_PATCH=already-applied" in second.stdout


def test_query_fp8_setup_script_rejects_unknown_source(tmp_path: Path):
    vllm_root = tmp_path / "vllm"
    target = vllm_root / "models/minimax_m3/nvidia/model.py"
    target.parent.mkdir(parents=True)
    target.write_text("unexpected source\n")
    (vllm_root / "__init__.py").write_text('__version__ = "test-version"\n')

    script = (
        Path(__file__).parents[1]
        / "configs/patches/patch-minimax-m3-query-fp8-graph-stability.sh"
    )
    env = os.environ | {"PYTHONPATH": str(tmp_path)}
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)

    assert result.returncode != 0
    assert "Refusing to patch" in result.stderr


def test_numa_interleave_query_fp8_wrapper_runs_both_setup_scripts():
    wrapper = (
        Path(__file__).parents[1]
        / "configs/patches/vllm-numa-interleave-query-fp8.sh"
    ).read_text()

    query_patch = "patch-minimax-m3-query-fp8-graph-stability.sh"
    numa_patch = "vllm-numa-interleave.sh"
    assert wrapper.index(query_patch) < wrapper.index(numa_patch)
