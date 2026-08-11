#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Serialize the wheel install across co-located srun tasks. With
# --ntasks-per-node > 1, every task on a node runs this preamble concurrently
# against the same shared container root; concurrent pip installs into the same
# site-packages race and corrupt each other. An exclusive flock serializes them,
# and a sentinel lets every task after the first skip the (idempotent) reinstall.
#
# The lock/sentinel are anchored in the active Python env (sys.prefix) — the
# exact resource being protected — NOT in /tmp: a bind-mounted /tmp can be shared
# between co-located containers, which would over-serialize independent installs
# and leave stale locks/sentinels across jobs. sys.prefix lives inside the
# container root, so it is shared by every task sharing that site-packages yet
# private to each container instance.
set -euo pipefail

LOCK_DIR="$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "${HOME:-/root}")"
LOCK="${LOCK_DIR}/.srtctl_dynamo_wheel_install.lock"
SENTINEL="${LOCK_DIR}/.srtctl_dynamo_wheel_install.complete"

(
    flock -x 200
    if [ -f "${SENTINEL}" ]; then
        echo "dynamo wheel install already completed in this environment, skipping"
    else
        python3 -m pip install --ignore-installed PyYAML==6.0.3
        pip install --no-cache-dir /dynamo_wheels/*.whl

        # Patch handler_base.py: add conversation_id= arg after disagg_request_id block
        python3 - <<'PYEOF'
import importlib.util, pathlib, sys

spec = importlib.util.find_spec("dynamo.trtllm.request_handlers.handler_base")
if spec is None:
    print("WARNING: could not find handler_base module, skipping patch", file=sys.stderr)
    sys.exit(0)

path = pathlib.Path(spec.origin)
old = "                    ),\n"
new = "                    ),\n                    conversation_id=session_id_from_request(request),\n"
marker = "                    disagg_request_id=get_global_disagg_request_id("
injection = "                    conversation_id=session_id_from_request(request),"

text = path.read_text()
if injection in text:
    print(f"handler_base.py already patched, skipping")
    sys.exit(0)

# Find the closing paren of disagg_request_id(...) and insert after it
idx = text.find(marker)
if idx == -1:
    print("WARNING: patch target not found in handler_base.py, skipping", file=sys.stderr)
    sys.exit(0)

close = text.find(old, idx)
if close == -1:
    print("WARNING: could not find closing paren after disagg_request_id, skipping", file=sys.stderr)
    sys.exit(0)

text = text[:close + len(old)] + injection + "\n" + text[close + len(old):]
path.write_text(text)
print(f"Patched {path}")
PYEOF

        touch "${SENTINEL}"
    fi
) 200>"${LOCK}"
