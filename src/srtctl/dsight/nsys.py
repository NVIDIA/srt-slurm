# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read existing Nsight SQLite exports; never launch a profiler."""

from __future__ import annotations

import re
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .importer import Importer


def cpu_samples(run: Importer, c: sqlite3.Connection, utc: int, lo: int, hi: int) -> dict[str, Any] | None:
    """Retain timestamped callchains for the frontend PID, with no duration inference."""
    pid_rows = c.execute(
        "SELECT ((e.globalTid >> 24) & 16777215),count(*) n FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id WHERE coalesce(e.text,s.value) LIKE 'preprocess.%' GROUP BY 1 ORDER BY n DESC"
    ).fetchall()
    if not pid_rows:
        return None
    frontend_pid = pid_rows[0][0]
    names = []
    names_by_id = {}
    stacks = []
    stack_ids = {}
    samples = []
    # A temporary selection avoids joining every system-wide sample's callchain.
    c.execute(
        "CREATE TEMP TABLE picked_samples AS SELECT id,start,globalTid FROM COMPOSITE_EVENTS WHERE start>=? AND start<=? AND ((globalTid>>24)&16777215)=?",
        (lo, hi, frontend_pid),
    )
    c.execute("CREATE INDEX picked_samples_id ON picked_samples(id)")
    rows = c.execute(
        "SELECT e.id,e.start,e.globalTid,cc.stackDepth,coalesce(s.value,'[unresolved]'),coalesce(cc.unresolved,0) FROM picked_samples e JOIN SAMPLING_CALLCHAINS cc ON cc.id=e.id LEFT JOIN StringIds s ON s.id=cc.symbol ORDER BY e.id,cc.stackDepth"
    )
    previous = None
    frames = []
    current = None

    def save() -> None:
        if current is None:
            return
        stack = tuple(frames)
        if stack not in stack_ids:
            stack_ids[stack] = len(stacks)
            stacks.append(list(stack))
        samples.append([run.t(utc + current[1]), str(current[2]), stack_ids[stack], current[0]])

    for sample_id, time_ns, tid, _depth, name, unresolved in rows:
        if sample_id != previous:
            save()
            frames = []
            current = (sample_id, time_ns, tid)
            previous = sample_id
        if unresolved:
            name = "[unresolved] " + name
        if name not in names_by_id:
            names_by_id[name] = len(names)
            names.append(name)
        frames.append(names_by_id[name])
    save()
    samples.sort(key=lambda s: s[0])
    return {
        "pid": frontend_pid,
        "names": names,
        "stacks": stacks,
        "samples": samples,
        "attribution": "frontend process samples; not request-specific CPU time",
        "evidence": "COMPOSITE_EVENTS.id -> SAMPLING_CALLCHAINS.id -> StringIds.id",
    }


def read_profiles(run: Importer) -> None:
    run.profiles_data = []
    if run.sqlites and not run.sqlites.exists():
        raise ValueError(f"Nsight SQLite path does not exist: {run.sqlites}")
    paths = ([run.sqlites] if run.sqlites.is_file() else sorted(run.sqlites.rglob("*.sqlite"))) if run.sqlites else []
    if run.sqlites and not paths:
        run.warnings.append("No *.sqlite exports found in supplied Nsight directory; .nsys-rep is not parsed directly.")
    for p in paths:
        c = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            required = {"NVTX_EVENTS", "StringIds", "TARGET_INFO_SESSION_START_TIME", "ANALYSIS_DETAILS"}
            if not required.issubset(tables):
                run.warnings.append(
                    f"{p.name}: missing Nsight tables {sorted(required - tables)}; export not imported."
                )
                c.close()
                continue
            m = re.search(r"_(prefill|decode)_w(\d+)_profile_rank(\d+)", p.stem)
            wid, rank = (
                (f"{m[1]}-{m[2]}", int(m[3]))
                if m
                else ("frontend" if re.search(r"_frontend_\d+", p.stem) else "unmapped", None)
            )
            utc, system = c.execute(
                "SELECT utcEpochNs,systemClockNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1"
            ).fetchone()
            start, stop = c.execute("SELECT startTime,stopTime FROM ANALYSIS_DETAILS LIMIT 1").fetchone()
            cap_start, cap_end = start - system, stop - system
            sid = run.source(p, "nsight_sqlite")
            env = (
                dict(c.execute("SELECT name,value FROM TARGET_INFO_SYSTEM_ENV WHERE name IN ('Hostname','HostName')"))
                if "TARGET_INFO_SYSTEM_ENV" in tables
                else {}
            )
            names = []
            name_ids = {}
            events = []
            bad = c.execute(
                "SELECT count(*) FROM NVTX_EVENTS WHERE end IS NOT NULL AND (end < start OR start < ? OR end > ?)",
                (cap_start, cap_end),
            ).fetchone()[0]
            lo, hi = run.origin - utc, run.origin + int(run.duration * 1e9) - utc
            query = """SELECT e.rowid,e.start,e.end,coalesce(e.text,s.value),e.globalTid
                FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id
                WHERE e.end IS NOT NULL AND e.start>=? AND e.end<=? AND e.end>=e.start
                  AND e.start<=? AND e.end>=?
                ORDER BY e.start,e.end DESC"""
            kept = 0
            timed = 0
            truncated = False
            for rowid, a, b, name, tid in c.execute(query, (cap_start, cap_end, hi, lo)):
                timed += 1
                # Keep the engine iteration/scheduler/forward context and frontend stages.
                # Tiny operator annotations remain in the original report, not this HTML.
                interesting = (name or "").startswith(
                    (
                        "[Executor]",
                        "preprocess.",
                        "route.",
                        "router.",
                        "tokenize",
                        "detokenize",
                        "_schedule",
                        "_forward_step",
                        "_prepare_inputs",
                        "_fetch_new_requests",
                        "prepare_resources",
                        "LLM.generate_async",
                        "RpcWorker.submit",
                        "kv_router.",
                        "transport.",
                        "compute_",
                    )
                )
                if not interesting or (name == "detokenize" and b - a < 100_000):
                    continue
                if kept >= run.max_profile_events:
                    truncated = True
                    break
                if name not in name_ids:
                    name_ids[name] = len(names)
                    names.append(name)
                events.append([run.t(utc + a), run.t(utc + b), name_ids[name], str(tid), rowid])
                kept += 1
            pid = len(run.profiles_data)
            item = {
                "id": pid,
                "worker": wid,
                "rank": rank,
                "host": env.get("Hostname", env.get("HostName")),
                "file": p.name,
                "evidence_source": sid,
                "epoch_ns": str(utc),
                "system_clock_ns": str(system),
                "capture": [run.t(utc + cap_start), run.t(utc + cap_end)],
                "invalid_or_boundary_ranges": bad,
                "timed_ranges_scanned": timed,
                "names": names,
                "events": events,
                "truncated": truncated,
                "cuda": "CUPTI_ACTIVITY_KIND_KERNEL" in tables,
                "attribution": "worker/rank/time context; no per-request NVTX identity",
                "clock": "Nsight session UTC anchor; cross-host skew not calibrated",
            }
            if "DIAGNOSTIC_EVENT" in tables:
                item["diagnostics"] = list(
                    {r[0] for r in c.execute("SELECT text FROM DIAGNOSTIC_EVENT WHERE severity IN (2,3)")}
                )
            item["detail_policy"] = (
                "Selected categories; detokenize ranges shorter than 100 microseconds remain in original SQLite only"
            )
            if wid == "frontend" and {"SAMPLING_CALLCHAINS", "COMPOSITE_EVENTS"}.issubset(tables):
                item["cpu"] = cpu_samples(run, c, utc, lo, hi)
            run.profiles_data.append(item)
            if wid in run.workers:
                run.workers[wid]["profiles"].append(pid)
            c.close()
            if item["truncated"]:
                run.warnings.append(
                    f"{p.name}: NVTX event limit reached ({run.max_profile_events}); query coverage is partial."
                )

        except sqlite3.Error as exc:
            raise ValueError(f"{p}: unsupported or corrupt Nsight SQLite export: {exc}") from exc
        finally:
            c.close()
