#!/usr/bin/env python3
"""Patch CUTLASS DSL global destructor metadata for CUDA 13 MLIR.

The v0.22.1 container has a CUTLASS DSL install that can emit
`llvm.mlir.global_dtors` without the `data` attribute. The bundled MLIR
verifier requires that attribute, so CuteDSL compilation fails in
cutlass_dsl/cutlass.py::post_compile_hook with:

    'llvm.mlir.global_dtors' op requires attribute 'data'

This patch keeps the old `llvm.mlir_global_dtors(dtors, priorities)` call
compatible with the installed generated bindings, then adds the missing
attribute directly on the operation.
"""

from __future__ import annotations

import sys
from pathlib import Path


TARGET_REL = "cutlass/cutlass_dsl/tvm_ffi_provider.py"
MARKER = "SRT hotfix: ensure llvm.mlir.global_dtors has data attr."

ANCHOR = "        # append the unload function to the global destructors\n"
ENSURE_DATA = f'''        # {MARKER}
        if "data" not in global_dtors.attributes:
            global_dtors.attributes["data"] = ir.ArrayAttr.get([])

'''

PRIORITIES_BLOCK = '''        global_dtors.attributes["priorities"] += [
            ir.IntegerAttr.get(self.i32_type, 65535)
        ]  # the default priority
'''

DATA_APPEND = '''        global_dtors.attributes["data"] += [
            ir.FlatSymbolRefAttr.get(unload_func_wrapper_symbol)
        ]  # required by newer LLVM dialect verifiers
'''


def candidate_paths(relpath: str) -> list[Path]:
    paths: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry)
        paths.append(root / relpath)
        paths.append(root / "nvidia_cutlass_dsl" / "python_packages" / relpath)
    return paths


def find_installed_file(relpath: str) -> Path:
    for candidate in candidate_paths(relpath):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find installed {relpath} on sys.path")


def patch_file(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"[cutlass-global-dtors-hotfix] {path}: already patched")
        return False

    patched = text
    changed = False

    if 'global_dtors.attributes["data"] += [' not in patched:
        if PRIORITIES_BLOCK not in patched:
            raise RuntimeError(f"Could not find priorities append block in {path}")
        patched = patched.replace(
            PRIORITIES_BLOCK, PRIORITIES_BLOCK + DATA_APPEND, 1
        )
        changed = True

    if ENSURE_DATA not in patched:
        if ANCHOR not in patched:
            raise RuntimeError(f"Could not find global dtors append anchor in {path}")
        patched = patched.replace(ANCHOR, ENSURE_DATA + ANCHOR, 1)
        changed = True

    if changed:
        path.write_text(patched)
        print(f"[cutlass-global-dtors-hotfix] patched {path}")
        return True

    print(f"[cutlass-global-dtors-hotfix] {path}: already compatible")
    return False


def main() -> None:
    target = find_installed_file(TARGET_REL)
    patch_file(target)


if __name__ == "__main__":
    main()
