#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stage a model directory from (slow, shared) storage to node-local storage.
#
# Runs once per allocated worker node (srun --ntasks-per-node=1) inside the
# worker container, BEFORE workers start. Idempotent: if the destination already
# matches the source (name+size manifest), the copy is skipped, so re-runs on a
# warm node are cheap. Symlink-safe: dereferences with `cp -L` / `find -L` so the
# staged copy contains real files. On any failure it exits non-zero, which fails
# the job (no silent fallback to the slow path).
#
# Usage: stage_model.sh <SOURCE_DIR> <DEST_DIR>
#   SOURCE_DIR  in-container path of the shared model (srtctl mounts it at /model)
#   DEST_DIR    node-local path to stage into (e.g. /raid/scratch/models/<name>)
#
# Env:
#   STAGE_PARALLEL  per-node copy fan-out (default 16)
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: stage_model.sh <SOURCE_DIR> <DEST_DIR>" >&2
    exit 2
fi
SRC="$1"
DEST="$2"
PARALLEL="${STAGE_PARALLEL:-16}"
HOST="$(hostname)"

if [ ! -d "$SRC" ]; then
    echo "[stage:$HOST] ERROR: source dir not found: $SRC" >&2
    exit 1
fi
mkdir -p "$DEST"

# name+size manifest (dereferences symlinks); relative paths, sorted.
manifest() { (cd "$1" && find -L . -type f -printf '%P\t%s\n' 2>/dev/null | LC_ALL=C sort); }

if diff -q <(manifest "$SRC") <(manifest "$DEST") >/dev/null 2>&1; then
    echo "[stage:$HOST] $DEST already matches $SRC (manifest hit) — skipping copy"
    exit 0
fi

echo "[stage:$HOST] staging $SRC -> $DEST (parallel=$PARALLEL)"
start=$(date +%s)
( cd "$SRC" && find -L . -type f -print0 \
    | xargs -0 -P "$PARALLEL" -I{} bash -c '
        rel="$1"; dest_root="$2"
        mkdir -p "$dest_root/$(dirname "$rel")"
        cp -L "$rel" "$dest_root/$rel"
      ' _ {} "$DEST" )

# Verify: destination must now match the source manifest.
if diff -q <(manifest "$SRC") <(manifest "$DEST") >/dev/null 2>&1; then
    echo "[stage:$HOST] done in $(( $(date +%s) - start ))s; manifest verified"
else
    echo "[stage:$HOST] ERROR: manifest mismatch after copy" >&2
    exit 1
fi
