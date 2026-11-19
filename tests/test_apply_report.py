# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the apply submission reporter."""

import io
import json

import pytest

from srtctl.cli.apply_report import ApplyReport
from srtctl.contract.failures import FailureCode, SrtFailure, reraise_as


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_inactive_report_is_a_passthrough():
    out = io.StringIO()
    report = ApplyReport(json_out=out)

    with pytest.raises(ValueError, match="boom"), report.item(recipe_path="s.yaml"):
        raise ValueError("boom")

    assert out.getvalue() == ""
    assert report.exit_code == 0


def test_item_indexes_and_identity_merge():
    out = io.StringIO()
    report = ApplyReport(active=True, json_out=out)

    with (
        pytest.raises(SrtFailure),
        report.item(recipe_path="s.yaml"),
        report.item(parameters={"c": 4}),
        reraise_as(FailureCode.RECIPE_INVALID, ValueError),
    ):
        raise ValueError("bad")
    with pytest.raises(RuntimeError), report.item(recipe_path="s.yaml"):
        raise RuntimeError("other")

    first, second = _records(out)
    assert first["submission_index"] == 1
    assert first["recipe_path"] == "s.yaml"
    assert first["parameters"] == {"c": 4}
    assert first["code"] == "recipe_invalid"
    assert second["submission_index"] == 2
    assert second["code"] == "unknown"


def test_system_exit_passes_through_untouched():
    out = io.StringIO()
    report = ApplyReport(active=True, json_out=out)

    with pytest.raises(SystemExit), report.item(recipe_path="s.yaml"):
        raise SystemExit(4)

    assert out.getvalue() == ""


def test_submitted_records_identity_and_is_kept():
    out = io.StringIO()
    report = ApplyReport(active=True, json_out=out)

    with report.item(recipe_path="s.yaml"):
        report.submitted({"status": "ok"})

    assert report.submissions == _records(out)
    assert report.submissions[0]["recipe_path"] == "s.yaml"
    assert report.submissions[0]["schema_version"] == 1


def test_failure_reported_once_across_nested_scopes():
    out = io.StringIO()
    report = ApplyReport(active=True, json_out=out)

    with pytest.raises(SrtFailure), report.item(recipe_path="s.yaml"), report.item(parameters={"c": 4}):
        raise SrtFailure(FailureCode.SBATCH_REJECTED, "no")

    assert len(_records(out)) == 1


def test_fail_is_command_level_and_not_repeated():
    out = io.StringIO()
    report = ApplyReport(active=True, json_out=out)

    failure = report.fail(ValueError("bad flag"))
    report.fail(failure)

    (record,) = _records(out)
    assert record["code"] == "unknown"
    assert "submission_index" not in record


def test_finish_emits_no_submissions_only_when_empty():
    out = io.StringIO()
    ApplyReport(active=True, json_out=out).finish()
    assert _records(out) == [{"schema_version": 1, "status": "no-submissions"}]

    out = io.StringIO()
    report = ApplyReport(active=True, json_out=out)
    report.fail(ValueError("x"))
    out.truncate(0)
    out.seek(0)
    report.finish()
    assert out.getvalue() == ""


class _ClosedPipe(io.StringIO):
    def write(self, s: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")


def test_closed_stdout_never_interrupts_the_run(tmp_path, capsys):
    failures_file = tmp_path / "failures.jsonl"
    report = ApplyReport(active=True, json_out=_ClosedPipe(), failures_file=failures_file)
    prepared = False

    with pytest.raises(SrtFailure), report.item(recipe_path="r.yaml"):
        report.accepted("42", "job", tmp_path / "42")
        prepared = True
        raise OSError("disk full")

    assert prepared
    assert json.loads(failures_file.read_text())["slurm_job_id"] == "42"
    assert report.exit_code == 3
    assert "Could not write record to stdout" in capsys.readouterr().err
