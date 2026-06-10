#!/usr/bin/env python3
"""Patch vLLM's CuteDSL fmax helper for CUTLASS NVVM ABI drift.

vLLM v0.22.1 selects the nvvm.fmax calling convention from CUDA_VERSION. CUDA
13 images with nvidia-cutlass-dsl 4.5.2 still expose the older
nvvm.fmax(res, a, b, ...) signature, so the CUDA-version gate calls it as
nvvm.fmax(a, b, ...) and raises:

    TypeError: fmax() missing 1 required positional argument: 'b'

This patch switches the helper to inspect the actual nvvm.fmax signature.
"""

from __future__ import annotations

import sys
from pathlib import Path


TARGET_REL = "vllm/vllm_flash_attn/cute/utils.py"
START = "@dsl_user_op\ndef fmax("
END = "\n\n@cute.jit\ndef fmax_reduce("
MARKER = "SRT hotfix: choose nvvm.fmax ABI by inspecting its signature."


NEW_FMAX = '''@dsl_user_op
def fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    # SRT hotfix: choose nvvm.fmax ABI by inspecting its signature.
    a_ir = Float32(a).ir_value(loc=loc, ip=ip)
    b_ir = Float32(b).ir_value(loc=loc, ip=ip)
    c_ir = Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None

    first_param = next(iter(inspect.signature(nvvm.fmax).parameters), None)
    if first_param == "res":
        return Float32(nvvm.fmax(T.f32(), a_ir, b_ir, c=c_ir, loc=loc, ip=ip))
    return Float32(nvvm.fmax(a_ir, b_ir, c=c_ir, loc=loc, ip=ip))
'''


def find_installed_file(relpath: str) -> Path:
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / relpath
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find installed {relpath} on sys.path")


def patch_file(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"[vllm-fmax-hotfix] {path}: already patched")
        return False

    start = text.find(START)
    if start < 0:
        raise RuntimeError(f"Could not find fmax helper start in {path}")

    end = text.find(END, start)
    if end < 0:
        raise RuntimeError(f"Could not find fmax_reduce boundary in {path}")

    patched = text[:start] + NEW_FMAX + text[end:]
    path.write_text(patched)
    print(f"[vllm-fmax-hotfix] patched {path}")
    return True


def main() -> None:
    target = find_installed_file(TARGET_REL)
    patch_file(target)


if __name__ == "__main__":
    main()
