#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="$(python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
TARGET="${VLLM_ROOT}/models/minimax_m3/nvidia/model.py"

python3 - "${TARGET}" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
source = target.read_text()

old = '''        use_msa_decode = self.impl.should_use_msa_decode(self.layer_name)
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
new = '''        query_fp8 = (
            torch.empty(
                (num_tokens, self.q_size),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )
            if getattr(self.impl, "use_cutlass_decode", False)
            else None
        )
'''

if old in source:
    target.write_text(source.replace(old, new, 1))
    state = "applied"
elif new in source:
    state = "already-applied"
else:
    raise SystemExit(
        f"Refusing to patch {target}: expected query_fp8 source block was not found"
    )

verified = target.read_text()
if new not in verified or old in verified:
    raise SystemExit(f"query_fp8 patch verification failed for {target}")

print(f"MINIMAX_QUERY_FP8_PATCH={state}")
print(f"MINIMAX_QUERY_FP8_PATCH_TARGET={target}")
PY

python3 - <<'PY'
import vllm

print(f"MINIMAX_QUERY_FP8_PATCH_VLLM_VERSION={vllm.__version__}")
PY
