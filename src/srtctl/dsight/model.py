# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request progress and runtime activity are different measurements.

Progress partitions client wall time between recorded milestones. Runtime spans
retain their original parentage and inclusive durations; they are never summed
to construct TTFT. Both views carry the evidence used for each boundary.
"""

from __future__ import annotations

from typing import Any

from .sources import canonical_role

SCHEMA = "srtctl-trace/1"
Record = dict[str, Any]


def activity_label(span: Record) -> tuple[str, str, str] | None:
    role = span["role"].capitalize()
    name = span["name"]
    if name == "request.preprocessing":
        return "Preprocessing", "activity", "Frontend request preparation, including tokenization."
    if name == "kv_router.select_worker":
        phase = span.get("routing_context", {}).get("phase", "Router")
        return (
            f"{phase.capitalize()} selection",
            "activity",
            span.get("routing_context", {}).get("basis", "Worker selection; the phase is not recorded on this span."),
        )
    if name == "worker.admission":
        return (
            f"{role} ingress and transport setup",
            "activity",
            ("Envelope decoding and response transport setup; not engine scheduler admission."),
        )
    if name.startswith("worker.operation."):
        return (
            f"{role} operation",
            "envelope",
            ("Inclusive parent of dispatch and response pumping, including backend waits. Not GPU compute time."),
        )
    if name == "request.dispatch":
        return (
            f"{role} backend stream creation",
            "activity",
            (
                "Runtime segment.generate. The Dynamo Python path creates a response stream; "
                "this does not prove engine submission or first-token completion."
            ),
        )
    if name.startswith("response.streaming."):
        return (
            f"{role} response pump",
            "activity",
            ("Starts before awaiting the first response; includes initial backend wait, generation, and publishing."),
        )
    if name == "response.streaming":
        return (
            "Frontend response stream",
            "concurrent",
            (
                "First final SSE event available through stream completion/drop. Concurrent with worker generation; "
                "not client first-token receipt."
            ),
        )
    return None


def lifecycle(request: Record) -> Record:
    """Produce one deterministic model for the browser, CLI, Python, and MCP.

    No timestamp proximity join or clock correction is performed. Multiple
    attempts, repeated milestones, or invalid timestamps leave only the
    measured client split; raw activities remain available for inspection.
    Without supported OTel activities, no request breakdown is constructed.
    """
    activities: list[Record] = []
    for span in request["spans"]:
        definition = activity_label(span)
        if definition:
            label, kind, description = definition
            activities.append({**span, "label": label, "kind": kind, "description": description})
    if not activities:
        return {
            "request": request["id"],
            "available": False,
            "layout": "cumulative-milestones",
            "stages": [],
            "activities": [],
            "milestones": [],
            "issues": [],
            "rows": [],
            "timing": "",
        }
    activities.sort(key=lambda s: (s["start"], -s["end"], s["trace"], s["id"]))
    keys = {(s["trace"], s["id"]): s for s in activities}
    for span in activities:
        parent = keys.get((span["trace"], span.get("parent")))
        span["parent_activity"] = parent["id"] if parent else None
        span["depth"] = 1 if parent else 0

    milestones: list[Record] = []
    issues: list[str] = []
    if any(s["end"] < s["start"] for s in activities):
        issues.append("A recorded activity ends before it starts; clocks are not corrected")
    for span in activities:
        parent = keys.get((span["trace"], span.get("parent")))
        if parent and (span["start"] < parent["start"] or span["end"] > parent["end"]):
            issues.append("A recorded child is outside its parent; clocks are not corrected")
            break

    def add(name: str, label: str, role: str | None = None, boundary: str = "end") -> None:
        matches = [
            s
            for s in activities
            if s["name"] == name and (not role or canonical_role(s["role"]) == canonical_role(role))
        ]
        if len(matches) > 1:
            issues.append(f"Repeated {role or ''} {name}: cannot infer one sequential attempt")
        elif matches:
            span = matches[0]
            milestones.append(
                {
                    "time": span[boundary],
                    "label": label,
                    "role": span["role"],
                    "span_id": span["id"],
                    "trace": span["trace"],
                    "boundary": boundary,
                    "evidence": span["evidence"],
                    "host": span["host"],
                }
            )

    add("request.preprocessing", "Preprocessing complete")
    for role in ("prefill", "decode", "agg"):
        routes = [
            s
            for s in activities
            if s["name"] == "kv_router.select_worker"
            and canonical_role(s.get("routing_context", {}).get("phase", "")) == role
        ]
        if len(routes) > 1:
            issues.append(f"Repeated {role} selection: cannot infer one sequential attempt")
        elif routes:
            span = routes[0]
            milestones.append(
                {
                    "time": span["end"],
                    "label": f"{role.capitalize()} route selected",
                    "role": "frontend",
                    "span_id": span["id"],
                    "trace": span["trace"],
                    "boundary": "end",
                    "evidence": span["evidence"],
                    "host": span["host"],
                    "phase_basis": span["routing_context"]["basis"],
                }
            )
        add("worker.admission", f"{role.capitalize()} transport ready", role)
        add("request.dispatch", f"{role.capitalize()} response stream created", role)
        if role == "prefill":
            add("worker.operation.prefill", "Prefill response handling complete", role)
    add("response.streaming", "First frontend SSE ready", boundary="start")
    first = request["first"]
    if len(request["server_ids"]) > 1:
        issues.append("Multiple Dynamo request IDs: progress is not linearized across attempts")
    # P/D routing and setup can overlap. Order measured boundaries by time,
    # not by a presumed engine execution sequence; this asserts no causality.
    milestones.sort(key=lambda item: (item["time"], item.get("trace", ""), item.get("span_id", ""), item["label"]))
    previous = request["start"]
    if first is None:
        issues.append("Client first-token timing is unavailable; no TTFT partition is inferred")
    for item in milestones if first is not None else []:
        if item["time"] < previous or item["time"] > first:
            issues.append("Server milestones are outside client TTFT; clocks are not corrected")
            break
        previous = item["time"]
    if first is not None and not request["start"] <= first <= request["end"]:
        issues.append("Client TTFT endpoint is outside the request; no duration partition is inferred")
    usable_first = first if first is not None and request["start"] <= first <= request["end"] else None
    if issues or usable_first is None:
        milestones = []
    if usable_first is not None:
        milestones.append(
            {
                "time": usable_first,
                "label": "First client token",
                "role": "client",
                "evidence": request["evidence"],
                "boundary": "client metric",
            }
        )
    milestones.append(
        {
            "time": request["end"],
            "label": "Client complete",
            "role": "client",
            "evidence": request["evidence"],
            "boundary": "request_end_ns",
        }
    )
    stages: list[Record] = []
    previous = {
        "time": request["start"],
        "label": "Client request start",
        "role": "client",
        "evidence": request["evidence"],
        "boundary": "request_start_ns",
    }
    for i, item in enumerate(milestones):
        stages.append(
            {
                "id": f"progress:{i}",
                "name": "milestone_interval",
                "kind": "progress",
                "row": i,
                "label": item["label"],
                "start": previous["time"],
                "end": item["time"],
                "role": item["role"],
                "host": item.get("host", "client"),
                "from_boundary": previous,
                "to_boundary": item,
                "evidence": item["evidence"],
                "source_span_id": item.get("span_id"),
                "description": f"Elapsed time from {previous['label']} to {item['label']}. "
                "Includes any intervening gaps and concurrent work; not exclusive engine time.",
            }
        )
        previous = item
    return {
        "request": request["id"],
        "available": True,
        "layout": "cumulative-milestones",
        "stages": stages,
        "activities": activities,
        "milestones": milestones,
        "issues": issues,
        "rows": [[s["id"] for s in stages[: i + 1]] for i in range(len(stages))],
        "timing": "Appended blocks measure time between the named milestones. "
        "Runtime activity retains its inclusive parent/child intervals on separate aligned tracks.",
    }
