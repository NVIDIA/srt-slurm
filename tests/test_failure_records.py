# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the failure record contract."""

import pytest

from srtctl.contract.failures import Cause, FailureCode, SrtFailure, as_failure, reraise_as
from srtctl.version import version_info


def test_record_has_fixed_fields_and_omits_unobserved():
    failure = SrtFailure(FailureCode.SUBMISSION_FAILED, "sbatch printed no job ID", cause=Cause(producer="sbatch"))
    failure.identity = {"slurm_job_id": "42", "restart_count": 0, "variant": None}

    assert failure.record() == {
        "schema_version": 1,
        "status": "error",
        "slurm_job_id": "42",
        "restart_count": 0,
        "code": "submission_failed",
        "error": "sbatch printed no job ID",
        "cause": {"producer": "sbatch"},
    }


def test_human_line_shows_code_subcode_and_accepted_job():
    failure = SrtFailure(FailureCode.SBATCH_REJECTED, "Slurm rejected the job", subcode="ESLURM_INVALID_TIME_LIMIT")
    assert failure.human_line() == "Error [sbatch_rejected / ESLURM_INVALID_TIME_LIMIT]: Slurm rejected the job"

    failure.identity = {"slurm_job_id": "42", "output_dir": "/o/42"}
    assert failure.human_line() == (
        "Error [sbatch_rejected / ESLURM_INVALID_TIME_LIMIT]: Slurm rejected the job (job 42 accepted, output /o/42)"
    )


def test_as_failure_wraps_unknown_and_passes_failures_through():
    failure = SrtFailure(FailureCode.PREFLIGHT_FAILED, "boom")
    assert as_failure(failure) is failure

    wrapped = as_failure(ValueError("bad"))
    assert wrapped.code is FailureCode.UNKNOWN
    assert str(wrapped) == "bad"
    assert as_failure(KeyError()).args[0] == "KeyError"


def test_reraise_as_relabels_only_the_listed_types():
    with pytest.raises(SrtFailure) as excinfo, reraise_as(FailureCode.RECIPE_INVALID, ValueError):
        raise ValueError("bad")
    assert (excinfo.value.code, excinfo.value.message) == (FailureCode.RECIPE_INVALID, "bad")
    assert isinstance(excinfo.value.__cause__, ValueError)

    failure = SrtFailure(FailureCode.PREFLIGHT_FAILED, "kept")
    with pytest.raises(SrtFailure) as excinfo, reraise_as(FailureCode.RECIPE_INVALID, RuntimeError):
        raise failure
    assert excinfo.value is failure

    with pytest.raises(KeyError), reraise_as(FailureCode.RECIPE_INVALID, ValueError):
        raise KeyError("other")


def test_version_info_reports_record_schema_version():
    assert version_info().record_schema_version == 1
    assert "records v1" in str(version_info())


def test_human_line_without_output_dir_has_no_job_suffix():
    failure = SrtFailure(FailureCode.METADATA_WRITE_FAILED, "boom")
    failure.identity = {"slurm_job_id": "42", "restart_count": 0}
    assert failure.human_line() == "Error [metadata_write_failed]: boom"
