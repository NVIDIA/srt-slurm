# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt native AgentPerf records to the client timing contract.

AgentPerf user.py writes epoch seconds and TTFT seconds. Its phase-end JSONL is
the analysis artifact; requests.jsonl provides the HTTP correlation header.
Join those using recorded identities, never timestamp proximity. Prompt and
response contents are deliberately excluded from the normalized dataset.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any


def identity(row: dict[str, Any]) -> tuple:
    return tuple(row.get(k) for k in ("request_id", "user_id", "conversation_id", "conversation_idx"))


def seconds_ns(value: Any) -> int:
    seconds = Decimal(str(value))
    if not seconds.is_finite():
        raise ValueError("Non-finite AgentPerf timestamp")
    return int(seconds * 1_000_000_000)


class AgentPerfAdapter:
    def __init__(self, path: Path, source: Callable[[Path, str], int]) -> None:
        self.source_id = source(path, "client")
        self.windows: dict[int, tuple[dict[str, Any], list[int]]] = {}
        self.analysis: dict[tuple, tuple[dict[str, Any], list[int]]] = {}
        manifest = path.with_name("phase_manifest.jsonl")
        if manifest.exists():
            sid = source(manifest, "agentperf_phase_manifest")
            with manifest.open() as stream:
                for line, text in enumerate(stream, 1):
                    row = json.loads(text)
                    phase = int(row["phase_idx"])
                    if phase in self.windows:
                        raise ValueError(f"{manifest}: duplicate phase {phase}; select one preserved run")
                    self.windows[phase] = (row, [sid, line])
        # The end-of-phase artifact has fully decoded metrics. Only keep the
        # fields used by this view, never response_text/messages/chunks.
        fields = (
            "request_id",
            "user_id",
            "conversation_id",
            "conversation_idx",
            "start_time",
            "end_time",
            "ttft",
            "success",
            "server_input_tokens",
            "server_output_tokens",
            "server_cached_tokens",
        )
        if path.name == "requests.jsonl":
            for artifact in sorted(path.parent.glob("*__phase*__traj*.jsonl")):
                match = re.search(r"__phase(\d+)__", artifact.name)
                if not match:
                    continue
                sid = source(artifact, "agentperf_phase_analysis")
                with artifact.open() as stream:
                    for line, text in enumerate(stream, 1):
                        row = json.loads(text)
                        key = (int(match[1]), *identity(row))
                        if key in self.analysis:
                            raise ValueError(f"{artifact}: ambiguous AgentPerf analysis identity {key}")
                        self.analysis[key] = ({k: row.get(k) for k in fields}, [sid, line])

    def normalize(self, raw: dict[str, Any], line: int) -> dict[str, Any]:
        phase = raw.get("phase_idx", 0)
        analysis = self.analysis.get((phase, *identity(raw)))
        row, evidence = analysis or (raw, [self.source_id, line])
        start, end = seconds_ns(row["start_time"]), seconds_ns(row["end_time"])
        window = self.windows.get(phase)
        measured = None
        if window:
            boundaries = window[0]
            measured = seconds_ns(boundaries["settling_end"]) <= start < seconds_ns(boundaries["actual_phase_end"])
        session = (
            f"phase-{phase}/user-{raw.get('user_id', 'unknown')}/conversation-{raw.get('conversation_id', 'unknown')}"
        )
        metric_fields = {
            "input_sequence_length": "server_input_tokens",
            "output_sequence_length": "server_output_tokens",
            "usage_prompt_cache_read_tokens": "server_cached_tokens",
        }
        metrics = {dest: {"value": row.get(src)} for dest, src in metric_fields.items()}
        metrics["time_to_first_token"] = {"value": row["ttft"] * 1000 if row.get("ttft") is not None else None}
        return {
            "metadata": {
                "x_request_id": raw.get("x_request_id") or raw.get("request_id"),
                "request_start_ns": start,
                "request_end_ns": end,
                "root_correlation_id": session,
                "x_correlation_id": session,
                "conversation_id": raw.get("conversation_id"),
                "turn_index": raw.get("conversation_idx"),
                "benchmark_phase": "profiling" if measured else "settling" if measured is False else None,
            },
            "metrics": metrics,
            "error": row.get("success") is False,
            "client_kind": "agentperf",
            "timing_quality": "phase analysis" if analysis else "request log (liveness)",
            "timing_evidence": evidence,
            "identity_evidence": [self.source_id, line],
            "phase_evidence": window[1] if window else None,
            "original_start_time": row["start_time"],
            "original_end_time": row["end_time"],
            "phase_idx": phase,
        }
