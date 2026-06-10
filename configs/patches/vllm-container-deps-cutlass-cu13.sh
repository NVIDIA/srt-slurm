#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

python3 -m pip install --force-reinstall --no-deps nvidia-cutlass-dsl-libs-cu13==4.5.2

python3 - <<'PY'
import inspect
from pathlib import Path

from cutlass._mlir.dialects import nvvm
import cutlass.cutlass_dsl.tvm_ffi_provider as tvm_ffi_provider

fmax_sig = inspect.signature(nvvm.fmax)
first_param = next(iter(fmax_sig.parameters), None)
provider_path = Path(tvm_ffi_provider.__file__)
provider_text = provider_path.read_text()
has_global_dtors_data = 'global_dtors.attributes["data"]' in provider_text

print(f"[cutlass-cu13] nvvm.fmax signature: {fmax_sig}")
print(f"[cutlass-cu13] tvm_ffi_provider: {provider_path}")
print(f"[cutlass-cu13] global_dtors data handling: {has_global_dtors_data}")

if first_param == "res":
    raise RuntimeError(
        "CUTLASS cu13 reinstall did not take effect: nvvm.fmax still uses "
        "the base signature with a leading 'res' parameter"
    )

if not has_global_dtors_data:
    raise RuntimeError(
        "CUTLASS cu13 reinstall did not take effect: global_dtors data "
        "handling is still missing"
    )
PY
