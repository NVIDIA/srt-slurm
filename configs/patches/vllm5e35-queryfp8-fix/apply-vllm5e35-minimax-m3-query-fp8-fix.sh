#!/usr/bin/env bash
set -euo pipefail

target="${MINIMAX_M3_MODEL_PY:-/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/nvidia/model.py}"

if [[ ! -f "${target}" ]]; then
    echo "MiniMax-M3 query_fp8 fix: ${target} is absent; nothing to patch in this container"
    exit 0
fi

python3 - "${target}" <<'PY'
from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
original_sha256 = "b82cdcb337369d5c53adb8708b44e51c8f7c052e2f44dc8b24232cecaad5bcc9"
patched_sha256 = "0ec1822d956c3a581c38ff9d054a1c04c992182515aa630d3cee0dc703c215bf"
contents = path.read_bytes()
source_sha256 = hashlib.sha256(contents).hexdigest()

if source_sha256 == patched_sha256:
    compile(contents, str(path), "exec")
    print(f"MiniMax-M3 query_fp8 fix: already applied in {path}")
    raise SystemExit(0)
if source_sha256 != original_sha256:
    raise SystemExit(
        "MiniMax-M3 query_fp8 fix: refusing unexpected source: "
        f"sha256={source_sha256}, path={path}"
    )

source = contents.decode()
vulnerable = '''        use_msa_decode = self.impl.should_use_msa_decode(self.layer_name)
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
fixed = '''        query_fp8 = (
            torch.empty(
                (num_tokens, self.q_size),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )
            if getattr(self.impl, "use_cutlass_decode", False)
            else None
        )
'''

vulnerable_count = source.count(vulnerable)
fixed_count = source.count(fixed)
if vulnerable_count != 1 or fixed_count != 0:
    raise SystemExit(
        "MiniMax-M3 query_fp8 fix: refusing unexpected source layout: "
        f"vulnerable_count={vulnerable_count}, fixed_count={fixed_count}, path={path}"
    )

patched = source.replace(vulnerable, fixed, 1).encode()
compile(patched, str(path), "exec")
if hashlib.sha256(patched).hexdigest() != patched_sha256:
    raise SystemExit(f"MiniMax-M3 query_fp8 fix: patched hash mismatch: {path}")

source_stat = path.stat()
fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(patched)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_path, stat.S_IMODE(source_stat.st_mode))
    if os.geteuid() == 0:
        os.chown(temporary_path, source_stat.st_uid, source_stat.st_gid)
    os.replace(temporary_path, path)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)

print(f"MiniMax-M3 query_fp8 fix: applied to {path}")
print(f"MiniMax-M3 query_fp8 fix: original sha256={original_sha256}")
print(f"MiniMax-M3 query_fp8 fix: verified sha256={patched_sha256}")
PY
