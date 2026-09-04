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
#
# CAVEAT: sys.prefix is only container-private when python3 resolves to the
# container's own interpreter. A recipe that puts a venv on shared storage ahead
# of the container on PATH (e.g. backend.prefill_environment PATH=/lustre/.../bin)
# moves sys.prefix — and therefore this sentinel — onto that shared, job-spanning
# path. The sentinel is keyed on wheel content below precisely so that case stays
# correct: a persistent sentinel then only suppresses reinstalls of the SAME
# wheels, and swapping /dynamo_wheels still re-triggers the install rather than
# silently leaving the previous job's build in place.
set -euo pipefail

LOCK_DIR="$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "${HOME:-/root}")"
LOCK="${LOCK_DIR}/.srtctl_dynamo_wheel_install.lock"

# Identify the wheel set by content, not by mere presence of a sentinel. Hashing
# costs one extra read of the wheels per task, which is noise next to the model
# load these same tasks are about to do, and it is the only identity that catches
# a rebuilt wheel published under an unchanged filename. Falls back to
# name/size/mtime, then to a fixed id (restoring the old install-once behavior)
# so a missing coreutils binary degrades instead of failing the worker.
WHEEL_ID="$(
    { sha256sum /dynamo_wheels/*.whl 2>/dev/null \
        || stat -c '%n %s %Y' /dynamo_wheels/*.whl 2>/dev/null; } \
        | sha256sum 2>/dev/null | cut -c1-16 || true
)"
[ -n "${WHEEL_ID}" ] || WHEEL_ID="unknown"
SENTINEL="${LOCK_DIR}/.srtctl_dynamo_wheel_install.${WHEEL_ID}.complete"

(
    flock -x 200
    if [ -f "${SENTINEL}" ]; then
        echo "dynamo wheel install already completed for wheel set ${WHEEL_ID}, skipping"
    else
        # --ignore-installed: the container's PyYAML is frequently distro-packaged,
        # which pip refuses to uninstall ("distutils installed project").
        python3 -m pip install --ignore-installed PyYAML==6.0.3
        pip install --no-cache-dir /dynamo_wheels/*.whl

        # Drop sentinels from previous wheel sets so this directory records the
        # one build actually installed rather than accumulating every past id.
        rm -f "${LOCK_DIR}"/.srtctl_dynamo_wheel_install.*.complete
        touch "${SENTINEL}"
        echo "dynamo wheel install completed for wheel set ${WHEEL_ID}"
    fi
) 200>"${LOCK}"
