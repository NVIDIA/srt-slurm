#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json from agentperf per-phase stats.

The client writes one ``<stem>__traj<N>.json`` per phase under
``<log_dir>/agentperf/`` containing the same statistics as the human-readable
phase summary. This rollup flattens each phase into one record so downstream
consumers don't need to know the client's filename scheme.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# model__<N>u__phase<i>__dur<D>s__settle<S>s__traj<T>.json
# dur/settle are floats in the client and render with a decimal point when
# non-whole (e.g. dur300.5s), so those two groups must accept decimals.
STEM_RE = re.compile(r"__(\d+)u__phase(\d+)__dur(\d+(?:\.\d+)?)s__settle(\d+(?:\.\d+)?)s__traj(\d+)\.json$")


def main(log_dir: str) -> int:
    results_dir = Path(log_dir) / "agentperf"
    if not results_dir.is_dir():
        print(f"no agentperf results dir at {results_dir}", file=sys.stderr)
        return 0

    phases = []
    for path in sorted(results_dir.glob("*__traj*.json")):
        m = STEM_RE.search(path.name)
        if not m:
            print(f"skipping {path.name}: does not match the phase-stem pattern", file=sys.stderr)
            continue
        try:
            stats = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable {path.name}: {exc}", file=sys.stderr)
            continue
        phases.append(
            {
                "file": path.name,
                "concurrency": int(m.group(1)),
                "phase_idx": int(m.group(2)),
                "phase_timeout_seconds": float(m.group(3)),
                "settling_time_seconds": float(m.group(4)),
                "trajectories_per_user": int(m.group(5)),
                "stats": stats,
            }
        )

    if not phases:
        print(f"no agentperf phase stats found under {results_dir}", file=sys.stderr)
        return 0

    out = Path(log_dir) / "benchmark-rollup.json"
    out.write_text(json.dumps({"benchmark": "agentperf", "phases": phases}, indent=1))
    print(f"wrote {out} ({len(phases)} phase(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
