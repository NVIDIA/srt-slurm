# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the failure records `srtctl apply --json` emits."""

import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli import submit as submit_cli
from srtctl.contract.failures import FailureCode, SrtFailure
from srtctl.core.docker import DockerLauncher
from srtctl.core.validation import PreflightIssue

MINIMAL_CONFIG = {
    "schema": 2,
    "name": "test-job",
    "model": {"path": "/models/test-model", "container": "test-container.sqsh", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
    "benchmark": {"type": "manual"},
}
SWEEP_CONFIG = {
    **MINIMAL_CONFIG,
    "sweep": {"tokens": [1, 2]},
    "roles": {**MINIMAL_CONFIG["roles"], "prefill": {"nodes": 1, "workers": 1, "args": {"x": "{tokens}"}}},
}
TWO_VARIANTS = {"base": MINIMAL_CONFIG, "override_x": {"name": "x-job"}, "override_y": {"name": "y-job"}}


def _sbatch(stdout: str = "Submitted batch job 42") -> MagicMock:
    return MagicMock(stdout=stdout, stderr="")


def _apply(
    monkeypatch,
    tmp_path: Path,
    *,
    config: dict | None = None,
    json_mode: bool = True,
    run: Any = None,
    real_preflight: bool = False,
    setup: Any = None,
    extra_patches: tuple = (),
    target: Path | str | None = None,
    args: tuple = (),
    command: str = "apply",
    output: bool = True,
) -> int:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.dump(config or MINIMAL_CONFIG))
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    output_args = ["-o", str(tmp_path)] if output else []
    argv = ["srtctl", command, "-f", str(target or cfg), *output_args, *args] + (["--json"] if json_mode else [])
    monkeypatch.setattr(sys, "argv", argv)
    code = 0
    with (
        patch("subprocess.run", **({"side_effect": run} if run else {"return_value": _sbatch()})),
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit.create_job_record"),
        patch("srtctl.cli.submit.validate_setup", **(setup or {})),
        ExitStack() as stack,
    ):
        if not real_preflight:
            stack.enter_context(patch("srtctl.cli.submit._assert_preflight_passed"))
        for extra in extra_patches:
            stack.enter_context(extra)
        try:
            submit_cli.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code


def _records(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_success_prints_accepted_then_submitted(monkeypatch, tmp_path, capsys):
    code = _apply(monkeypatch, tmp_path)
    accepted, submitted = _records(capsys)
    assert code == 0
    assert (accepted["status"], submitted["status"]) == ("accepted", "submitted")
    assert accepted["submission_index"] == submitted["submission_index"] == 1
    assert (accepted["job_name"], accepted["output_dir"]) == (submitted["job_name"], submitted["output_dir"])


def test_federated_sbatch_output_parses_job_id(monkeypatch, tmp_path, capsys):
    _apply(monkeypatch, tmp_path, run=lambda *a, **k: _sbatch("Submitted batch job 42 on cluster west"))
    assert _records(capsys)[0]["slurm_job_id"] == "42"


def test_sbatch_rejection_record(monkeypatch, tmp_path, capsys):
    stderr = "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"

    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["sbatch"], output="", stderr=stderr)

    code = _apply(monkeypatch, tmp_path, run=run)
    (record,) = _records(capsys)
    assert code == 1
    assert record["code"] == "sbatch_rejected"
    assert record["subcode"] == "ESLURM_INVALID_ACCOUNT"
    assert record["cause"]["producer"] == "sbatch"


def _fail_copy(name: str):
    real = shutil.copyfile

    def copyfile(src, dst, *args, **kwargs):
        if Path(dst).name == name:
            raise OSError("disk full")
        return real(src, dst, *args, **kwargs)

    return patch("shutil.copyfile", side_effect=copyfile)


def _fail_open_json():
    real = open

    def fake_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("42.json"):
            raise OSError("disk full")
        return real(file, mode, *args, **kwargs)

    return patch("builtins.open", side_effect=fake_open)


POST_SUBMIT = [
    (lambda: _fail_copy("config.yaml"), "job_preparation_failed"),
    (lambda: _fail_copy("sbatch_script.sh"), "sbatch_archive_failed"),
    (_fail_open_json, "metadata_write_failed"),
]


@pytest.mark.parametrize(("make_patch", "code"), POST_SUBMIT)
def test_accepted_then_post_submit_write_fails(monkeypatch, tmp_path, capsys, make_patch, code):
    exit_code = _apply(monkeypatch, tmp_path, extra_patches=(make_patch(),))
    accepted, error = _records(capsys)
    assert accepted["status"] == "accepted"
    assert error["status"] == "error"
    assert error["code"] == code
    assert error["slurm_job_id"] == "42"
    assert exit_code == 3


def test_accepted_then_failure_without_json_names_the_job(monkeypatch, tmp_path, capsys):
    exit_code = _apply(monkeypatch, tmp_path, json_mode=False, extra_patches=(_fail_open_json(),))
    out = capsys.readouterr().out
    assert "Error [metadata_write_failed]" in out
    assert "job 42 accepted" in out
    assert not any(line.startswith("{") for line in out.splitlines())
    assert exit_code == 3


def test_preflight_failure_keeps_every_issue(monkeypatch, tmp_path, capsys):
    issues = [
        PreflightIssue("model-missing", "model.path", "no model"),
        PreflightIssue("container-missing", "model.container", "no container"),
    ]
    failed = SimpleNamespace(ok=False, errors=issues)
    exit_code = _apply(
        monkeypatch,
        tmp_path,
        real_preflight=True,
        extra_patches=(
            patch("srtctl.cli.submit.preflight_config_variants", return_value=[failed]),
            patch("srtctl.cli.submit.load_cluster_config", return_value={}),
        ),
    )
    (record,) = _records(capsys)
    assert record["code"] == "preflight_failed"
    assert [issue["code"] for issue in record["issues"]] == ["model-missing", "container-missing"]
    assert exit_code == 1


def _setup_incomplete() -> MagicMock:
    return MagicMock(side_effect=SrtFailure(FailureCode.SETUP_INCOMPLETE, "Required binaries not found"))


def test_missing_setup_binaries_is_setup_incomplete(monkeypatch, tmp_path, capsys):
    assert _apply(monkeypatch, tmp_path, setup={"new": _setup_incomplete()}) == 1
    (record,) = _records(capsys)
    assert record["code"] == "setup_incomplete"


def test_unsupported_schema(monkeypatch, tmp_path, capsys):
    config = {k: v for k, v in MINIMAL_CONFIG.items() if k != "schema"}
    exit_code = _apply(monkeypatch, tmp_path, config=config)
    (record,) = _records(capsys)
    assert record["code"] == "recipe_schema_unsupported"
    assert exit_code == 1


def test_unexpected_exception_is_unknown(monkeypatch, tmp_path, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("sbatch is on fire")

    exit_code = _apply(monkeypatch, tmp_path, run=boom)
    (record,) = _records(capsys)
    assert record["code"] == "unknown"
    assert "on fire" in record["error"]
    assert exit_code == 1


def test_launcher_validate_problem_is_recipe_invalid(monkeypatch, tmp_path, capsys):
    launcher = MagicMock()
    launcher.name = "fake"
    launcher.validate.return_value = ["needs a GPU"]
    exit_code = _apply(
        monkeypatch, tmp_path, extra_patches=(patch("srtctl.cli.submit.get_launcher", return_value=launcher),)
    )
    (record,) = _records(capsys)
    assert record["code"] == "recipe_invalid"
    assert exit_code == 1


def test_docker_foreground_run_failure_keeps_job_exit_code(monkeypatch, tmp_path, capsys):
    launcher = DockerLauncher()
    with (
        patch.object(DockerLauncher, "validate", return_value=[]),
        patch.object(DockerLauncher, "start", return_value=7),
        patch("srtctl.cli.submit.get_launcher", return_value=launcher),
    ):
        exit_code = _apply(monkeypatch, tmp_path)
    records = _records(capsys)
    assert [r["status"] for r in records] == ["accepted", "submitted"]
    assert exit_code == 7


def test_directory_aborts_on_setup_incomplete(monkeypatch, tmp_path, capsys):
    validate = _setup_incomplete()
    directory = _recipes(tmp_path, a=MINIMAL_CONFIG, b=MINIMAL_CONFIG)
    assert _apply(monkeypatch, tmp_path, setup={"new": validate}, target=directory) == 1
    (record,) = _records(capsys)
    assert record["code"] == "setup_incomplete"
    assert validate.call_count == 1


def _recipes(tmp_path: Path, **files: Any) -> Path:
    directory = tmp_path / "recipes"
    directory.mkdir()
    for name, content in files.items():
        (directory / f"{name}.yaml").write_text(content if isinstance(content, str) else yaml.dump(content))
    return directory


def _fail_preflight_on_call(number: int):
    calls = iter(range(1, 100))

    def check(*args, **kwargs):
        if next(calls) == number:
            raise SrtFailure(FailureCode.PREFLIGHT_FAILED, "model missing")

    return patch("srtctl.cli.submit._assert_preflight_passed", side_effect=check)


@pytest.mark.parametrize("json_mode", [True, False])
def test_argparse_error_is_prose_only(monkeypatch, tmp_path, capsys, json_mode):
    assert _apply(monkeypatch, tmp_path, json_mode=json_mode, args=("--bogus",)) == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments: --bogus" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--set", "novalue"), "KEY=VALUE"),
        (("--set", "a..b=1"), "empty segment"),
    ],
)
@pytest.mark.parametrize("json_mode", [True, False])
def test_invocation_errors_exit_2(monkeypatch, tmp_path, capsys, args, message, json_mode):
    code = _apply(monkeypatch, tmp_path, json_mode=json_mode, args=args)
    captured = capsys.readouterr()
    assert code == 2
    assert "usage:" in captured.err
    if not json_mode:
        assert "{" not in captured.out
        return
    (record,) = [json.loads(line) for line in captured.out.splitlines()]
    assert record["code"] == "invocation_invalid"
    assert message in record["error"]
    assert not record["error"].startswith("srtctl apply: error:")


@pytest.mark.parametrize("json_mode", [True, False])
def test_bad_selector_and_missing_recipe_exit_2(monkeypatch, tmp_path, capsys, json_mode):
    for target in (tmp_path / "cfg.yaml:bogus", tmp_path / "missing.yaml"):
        assert _apply(monkeypatch, tmp_path, json_mode=json_mode, target=target) == 2
        out = capsys.readouterr().out
        assert ("{" in out) is json_mode
        assert "invocation_invalid" in out


def test_prose_error_line_is_not_wrapped(monkeypatch, tmp_path, capsys):
    missing = tmp_path / ("long-recipe-name-" * 8 + ".yaml")
    assert _apply(monkeypatch, tmp_path, json_mode=False, target=missing) == 2
    (line,) = [line for line in capsys.readouterr().out.splitlines() if "invocation_invalid" in line]
    assert str(missing) in line


def test_empty_stdin_exit_2(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(read=lambda: ""))
    assert _apply(monkeypatch, tmp_path, target="-") == 2
    assert _records(capsys)[0]["code"] == "invocation_invalid"


def test_help_is_not_an_error(monkeypatch, tmp_path, capsys):
    assert _apply(monkeypatch, tmp_path, args=("--help",)) == 0
    assert "{" not in capsys.readouterr().out


def test_yaml_syntax_error_is_recipe_invalid(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n")
    assert _apply(monkeypatch, tmp_path, target=bad) == 1
    (record,) = _records(capsys)
    assert record["code"] == "recipe_invalid"
    assert record["submission_index"] == 1
    assert record["recipe_path"] == str(bad)


def test_set_path_not_in_recipe_is_invocation_invalid(monkeypatch, tmp_path, capsys):
    assert _apply(monkeypatch, tmp_path, args=("--set", "model.path.x=1")) == 2
    assert _records(capsys)[0]["code"] == "invocation_invalid"


@pytest.mark.parametrize("json_mode", [True, False])
def test_directory_mixed_results_exit_3(monkeypatch, tmp_path, capsys, json_mode):
    directory = _recipes(tmp_path, a=MINIMAL_CONFIG, b="name: [broken\n")
    assert _apply(monkeypatch, tmp_path, json_mode=json_mode, target=directory) == 3
    out = capsys.readouterr().out
    if not json_mode:
        assert "Error [recipe_invalid]" in out
        return
    accepted, submitted, error = [json.loads(line) for line in out.splitlines()]
    assert (accepted["status"], submitted["status"], error["status"]) == ("accepted", "submitted", "error")
    assert (submitted["submission_index"], error["submission_index"]) == (1, 2)
    assert submitted["recipe_path"].endswith("a.yaml")
    assert error["recipe_path"].endswith("b.yaml")


def test_sweep_item_two_fails_after_item_one_accepted(monkeypatch, tmp_path, capsys):
    code = _apply(monkeypatch, tmp_path, config=SWEEP_CONFIG, extra_patches=(_fail_preflight_on_call(2),))
    accepted, submitted, error = _records(capsys)
    assert code == 3
    assert (accepted["submission_index"], submitted["submission_index"], error["submission_index"]) == (1, 1, 2)
    assert error["parameters"] == {"tokens": 2}


def test_override_variant_identity(monkeypatch, tmp_path, capsys):
    override = {"base": MINIMAL_CONFIG, "override_x": {"name": "x-job"}}
    code = _apply(monkeypatch, tmp_path, config=override, extra_patches=(_fail_preflight_on_call(1),))
    (record,) = _records(capsys)
    assert code == 1
    assert record["variant"] == "override_x"


def test_empty_directory_no_submissions(monkeypatch, tmp_path, capsys):
    directory = _recipes(tmp_path)
    assert _apply(monkeypatch, tmp_path, target=directory) == 0
    assert _records(capsys) == [{"schema_version": 1, "status": "no-submissions"}]


@pytest.mark.parametrize("json_mode", [True, False])
def test_failures_file_written_with_and_without_json(monkeypatch, tmp_path, json_mode):
    path = tmp_path / "out" / "failures.jsonl"
    _apply(monkeypatch, tmp_path, json_mode=json_mode, config={"name": "x"}, args=("--failures-file", str(path)))
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["status"] == "error"


def test_unwritable_failures_file_keeps_the_error_path(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    args = ("--failures-file", str(blocker / "failures.jsonl"))
    exit_code = _apply(monkeypatch, tmp_path, args=args, extra_patches=(_fail_open_json(),))
    captured = capsys.readouterr()
    assert [json.loads(line)["status"] for line in captured.out.splitlines()] == ["accepted", "error"]
    assert "Could not write failure record" in captured.err and "Traceback" not in captured.err
    assert exit_code == 3


def test_dry_run_directory_with_bad_file_still_exits_0(monkeypatch, tmp_path):
    directory = _recipes(tmp_path, a=MINIMAL_CONFIG, b="name: [broken\n")
    assert _apply(monkeypatch, tmp_path, command="dry-run", json_mode=False, target=directory) == 0


def test_dry_run_failure_shows_the_code_for_files_and_directories(monkeypatch, tmp_path, capsys):
    directory = _recipes(tmp_path, b="name: [broken\n")
    assert _apply(monkeypatch, tmp_path, command="dry-run", json_mode=False, target=directory / "b.yaml") == 1
    assert _apply(monkeypatch, tmp_path, command="dry-run", json_mode=False, target=directory) == 0
    assert capsys.readouterr().out.count("Error [recipe_invalid]") == 2


def test_set_parse_error_reaches_failures_file(monkeypatch, tmp_path):
    path = tmp_path / "f.jsonl"
    assert _apply(monkeypatch, tmp_path, args=("--set", "novalue", "--failures-file", str(path))) == 2
    (record,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["code"] == "invocation_invalid"


def test_non_apply_bad_flag_emits_no_record(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["srtctl", "dry-run", "-f", "x.yaml", "--bogus", "--json"])
    with pytest.raises(SystemExit) as excinfo:
        submit_cli.main()
    assert excinfo.value.code == 2
    assert "{" not in capsys.readouterr().out


def test_dry_run_bad_set_still_exits_2_with_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["srtctl", "dry-run", "-f", "x.yaml", "--set", "novalue"])
    with pytest.raises(SystemExit) as excinfo:
        submit_cli.main()
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "usage:" in captured.err
    assert "{" not in captured.out


@pytest.mark.parametrize(
    ("target", "config", "args"),
    [
        (":override_x", None, ()),
        ("dir", None, ("--set", "name=x")),
        (None, TWO_VARIANTS, ("--serve-only",)),
        (None, SWEEP_CONFIG, ("--serve-only",)),
        (None, None, ("--serve-only", "--mock")),
        (None, None, ("--serve-only", "--sweep")),
    ],
    ids=[
        "empty-path",
        "set-on-dir",
        "serve-only-variants",
        "serve-only-sweep",
        "serve-only-mock",
        "serve-only-force-sweep",
    ],
)
def test_unsupported_invocation_shapes_exit_2(monkeypatch, tmp_path, capsys, target, config, args):
    if target == "dir":
        target = _recipes(tmp_path, a=MINIMAL_CONFIG)
    assert _apply(monkeypatch, tmp_path, config=config, target=target, args=args) == 2
    assert _records(capsys)[-1]["code"] == "invocation_invalid"


def test_empty_recipe_file_is_recipe_invalid(monkeypatch, tmp_path, capsys):
    empty = tmp_path / "empty.yaml"
    empty.write_text("\n")
    assert _apply(monkeypatch, tmp_path, target=empty) == 1
    (record,) = _records(capsys)
    assert record["code"] == "recipe_invalid"
    assert record["recipe_path"] == str(empty)


def test_directory_submits_override_and_sweep_files(monkeypatch, tmp_path, capsys):
    directory = _recipes(tmp_path, a_override=TWO_VARIANTS, b_sweep=SWEEP_CONFIG)
    assert _apply(monkeypatch, tmp_path, target=directory) == 0
    submitted = [r for r in _records(capsys) if r["status"] == "submitted"]
    assert [(Path(r["recipe_path"]).name, r.get("variant")) for r in submitted] == [
        ("a_override.yaml", "override_x"),
        ("a_override.yaml", "override_y"),
        ("b_sweep.yaml", None),
        ("b_sweep.yaml", None),
    ]


def test_srtslurm_output_dir_is_the_default_output_base(monkeypatch, tmp_path, capsys):
    base = tmp_path / "cluster-outputs"
    setting = patch(
        "srtctl.cli.submit.get_srtslurm_setting",
        side_effect=lambda key, default=None: str(base) if key == "output_dir" else default,
    )
    assert _apply(monkeypatch, tmp_path, output=False, extra_patches=(setting,)) == 0
    assert _records(capsys)[0]["output_dir"] == str(base / "42")
