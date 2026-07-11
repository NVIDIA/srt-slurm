#!/bin/bash
# Apply only the Dynamo runtime compatibility needed by the expiry3600 wheel.
#
# This intentionally does not apply the old vLLM MooncakeStore PR123/resumed
# request patch or the external Rust frontend body-limit override. The
# b7ca6ed image already carries the PR123-side vLLM changes, and this recipe
# keeps srt/Dynamo on its normal non-Rust frontend path.
set -euo pipefail

SRC=/configs/patches/dynamo_kv_connector_protocols.py
DST=$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("dynamo.vllm.kv_connector_protocols")
print(spec.origin if spec is not None and spec.origin is not None else "")
PY
)

if [ -n "$DST" ]; then
    if [ ! -f "$SRC" ]; then
        echo "ERROR: patch source $SRC not found" >&2
        exit 1
    fi
    cp "$SRC" "$DST"
    echo "Patched legacy dynamo MultiConnector support -> $DST"
else
    echo "dynamo.vllm.kv_connector_protocols not present; skipping legacy Dynamo MultiConnector protocol patch"
fi

python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("dynamo.vllm.main")
if spec is None or spec.origin is None:
    raise SystemExit("ERROR: cannot locate dynamo.vllm.main for expiry3600 ABI compatibility patch")

path = Path(spec.origin)
text = path.read_text()
needle = "        ignore_weights=should_register_model_ignore_weights(config),\n"
if needle in text:
    path.write_text(text.replace(needle, ""))
    print(f"patched Dynamo register_model ignore_weights compatibility in {path}")
elif "ignore_weights=should_register_model_ignore_weights(config)" in text:
    raise SystemExit(
        f"ERROR: found Dynamo ignore_weights call in unexpected format; refusing to patch {path}"
    )
else:
    print(f"Dynamo register_model ignore_weights compatibility patch already applied in {path}")
PY
