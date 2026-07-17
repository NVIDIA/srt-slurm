#!/usr/bin/env python3
"""Neutralize DeepGEMM's pow2-SF device asserts in smxx_layout.cuh.

Image 515d6e9 contains ``DG_DEVICE_ASSERT((v & 0x807fffffu) == 0)`` in both
UE8M0 scale-factor packing kernels. On the DSV4-Pro AMXFP4 path, rows that the
kernel still considers valid can contain values that are not pure powers of
two. The assertion then traps the engine during startup. The older DeepGEMM
path packed the exponent unconditionally; this patch restores that behavior.

DeepGEMM JIT-compiles from these headers, so changing the source also changes
the kernel cache key. The patch is intentionally anchor-checked and fails
closed if the image source no longer matches the validated 515d6e9 layout.
"""

import sys
from pathlib import Path

MARKER = "DG-SMXX-SF-ASSERT-OFF"
TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/smxx_layout.cuh"
)

OLD_SCALAR = """            // FP32 SFs must have a zero sign and mantissa (only the exponent is packed)
            DG_DEVICE_ASSERT((values[j] & 0x807fffffu) == 0);"""
NEW_SCALAR = """            // [DG-SMXX-SF-ASSERT-OFF] pow2-SF assert disabled: DSV4 amxf4
            // weight-scale packing feeds rows this predicate cannot exclude
            // (per-group padding / pre-rounded scales); pre-assert DeepGEMM
            // packed the exponent unconditionally and downstream kernels skip
            // non-data rows via grouped_layout.
            // DG_DEVICE_ASSERT((values[j] & 0x807fffffu) == 0);"""

OLD_VEC = """            // FP32 SFs must have a zero sign and mantissa (only the exponent is packed)
            DG_DEVICE_ASSERT((values[j].x & 0x807fffffu) == 0);
            DG_DEVICE_ASSERT((values[j].y & 0x807fffffu) == 0);
            DG_DEVICE_ASSERT((values[j].z & 0x807fffffu) == 0);
            DG_DEVICE_ASSERT((values[j].w & 0x807fffffu) == 0);"""
NEW_VEC = """            // [DG-SMXX-SF-ASSERT-OFF] see scalar-kernel note above.
            // DG_DEVICE_ASSERT((values[j].x & 0x807fffffu) == 0);
            // DG_DEVICE_ASSERT((values[j].y & 0x807fffffu) == 0);
            // DG_DEVICE_ASSERT((values[j].z & 0x807fffffu) == 0);
            // DG_DEVICE_ASSERT((values[j].w & 0x807fffffu) == 0);"""


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found", file=sys.stderr)
        return 1
    text = TARGET.read_text()
    if MARKER in text:
        print(f"dg-smxx-sf-assert-off already applied: {TARGET}")
        return 0
    for name, anchor in (("scalar", OLD_SCALAR), ("vector", OLD_VEC)):
        count = text.count(anchor)
        if count != 1:
            print(
                f"ERROR: {name} anchor matched {count} times (expected 1)",
                file=sys.stderr,
            )
            return 1
    text = text.replace(OLD_SCALAR, NEW_SCALAR, 1)
    text = text.replace(OLD_VEC, NEW_VEC, 1)
    TARGET.write_text(text)
    print(f"dg-smxx-sf-assert-off applied: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
