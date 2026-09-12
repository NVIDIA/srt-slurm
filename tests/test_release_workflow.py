"""Exercise the release workflow's actual Bash against local Git history."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/release.yaml"
REPOSITORY = "NVIDIA/srt-slurm"
SCRAPER = "src/tachometer/tachometer-scraper/src/main.rs"
EXPORTER = "src/cpu-power-exporter/src/main.rs"


def workflow() -> dict:
    # BaseLoader preserves GitHub's `on` key instead of treating it as a YAML 1.1 boolean.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def step_script(job: str, step_id: str) -> str:
    return next(step["run"] for step in workflow()["jobs"][job]["steps"] if step.get("id") == step_id)


@dataclass(frozen=True)
class History:
    root: Path
    bin_dir: Path

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, changes: dict[str, str | None]) -> str:
        for name, contents in changes.items():
            path = self.root / name
            if contents is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", "fixture")
        return self.git("rev-parse", "HEAD")

    def run(
        self,
        step: str,
        target: str,
        releases: list[dict] | None = None,
        prs: list[list[dict]] | None = None,
        previous: str = "",
        tag: str = "v1.0.1",
        fail: str = "",
    ) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
        output = self.bin_dir / "outputs"
        output.write_text("")
        env = {
            **os.environ,
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GH_REPO": REPOSITORY,
            "GH_TOKEN": "fixture-token",
            "TARGET": target,
            "TAG": tag,
            "PREVIOUS": previous,
            "GITHUB_OUTPUT": str(output),
            "FAKE_RELEASES": json.dumps(releases or []),
            "FAKE_PRS": json.dumps(prs or [[]]),
            "FAKE_GH_FAIL": fail,
        }
        result = subprocess.run(
            ["bash", "-c", step_script("version" if step == "plan" else "release", step)],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return result, outputs


@pytest.fixture
def history(tmp_path: Path) -> History:
    root = tmp_path / "repo"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if os.environ['FAKE_GH_FAIL'] == args[0]:\n"
        "    print('fixture API failure', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "if args[:2] == ['release', 'list']:\n"
        "    assert args[args.index('--json') + 1] == 'tagName,isDraft', args\n"
        "    print(os.environ['FAKE_RELEASES'])\n"
        "elif args[0] == 'api':\n"
        "    expected = 'repos/' + os.environ['GH_REPO'] + '/commits/' + os.environ['TARGET'] + '/pulls?per_page=100'\n"
        "    assert expected in args and '--paginate' in args and '--slurp' not in args, args\n"
        "    for page in json.loads(os.environ['FAKE_PRS']):\n"
        "        print(json.dumps(page))\n"
        "else:\n"
        "    raise AssertionError('Unexpected gh invocation: ' + repr(args))\n"
    )
    fake_gh.chmod(0o755)
    repo = History(root, bin_dir)
    repo.git("init", "--quiet", "-b", "main")
    repo.git("config", "user.name", "Release test")
    repo.git("config", "user.email", "release-test@example.invalid")
    repo.git("config", "commit.gpgsign", "false")
    repo.git("config", "core.hooksPath", "/dev/null")
    # Any workflow tag fetch remains local; no test can reach a remote repository.
    repo.git("remote", "add", "origin", str(root))
    repo.commit({"README.md": "initial\n", SCRAPER: "scraper\n", EXPORTER: "exporter\n"})
    return repo


def released(tag: str = "v1.0.0", *, draft: bool = False) -> dict:
    return {"tagName": tag, "isDraft": draft}


def associated_pr(target: str, *, feature: bool = True) -> dict:
    return {
        "merged_at": "2026-09-12T00:00:00Z",
        "merge_commit_sha": target,
        "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        "labels": [{"name": "new-feature"}] if feature else [],
        "head": {"repo": {"full_name": "contributor/srt-slurm"}},
    }


def assert_success(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def test_workflow_trust_and_publication_wiring() -> None:
    document = workflow()
    assert document["on"] == {"push": {"branches": ["main"]}}
    assert document["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert document["concurrency"] == {"group": "release", "cancel-in-progress": "false"}
    jobs = document["jobs"]
    assert jobs["version"]["outputs"]["target"] == "${{ github.sha }}"
    for name, job in jobs.items():
        if name == "release":
            assert job["permissions"] == {"contents": "write"}
        else:
            assert "write" not in job.get("permissions", {}).values()
        assert sum(step.get("uses", "").startswith("actions/checkout@") for step in job["steps"]) == 1
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["ref"] in {"${{ github.sha }}", "${{ needs.version.outputs.target }}"}
                assert step["with"]["persist-credentials"] == "false"
                if name in {"version", "release"}:
                    assert step["with"]["fetch-depth"] == "0"
                assert "allow-unsafe-pr-checkout" not in step["with"]
    assert "always()" in jobs["release"]["if"]
    assert "needs.version.outputs.publish == 'true'" in jobs["release"]["if"]
    carry = next(step for step in jobs["release"]["steps"] if "Carry forward" in step.get("name", ""))
    assert carry["env"]["PREVIOUS"] == "${{ needs.version.outputs.previous }}"
    assert "gh release list" not in carry["run"]
    create = next(step for step in jobs["release"]["steps"] if step.get("name") == "Create release")
    assert "steps.guard.outputs.publish == 'true'" in create["if"]
    release_steps = jobs["release"]["steps"]
    guard_index = next(index for index, step in enumerate(release_steps) if step.get("id") == "guard")
    # Checkout cleans the worktree, so it must run before release assets arrive.
    assert release_steps[0]["uses"].startswith("actions/checkout@")
    assert release_steps[guard_index + 1] is create
    assert 0 < release_steps.index(carry) < guard_index
    for index, step in enumerate(release_steps):
        if step.get("uses", "").startswith("actions/download-artifact@"):
            assert 0 < index < guard_index
    for step in release_steps[:guard_index]:
        assert "steps.guard.outputs" not in step.get("if", "")
    expected_assets = {
        "build-tachometer-scraper": {
            "tachometer-scraper-x86_64-unknown-linux-gnu",
            "tachometer-scraper-aarch64-unknown-linux-gnu",
        },
        "build-cpu-power-exporter": {
            "cpu-power-exporter-x86_64-unknown-linux-musl",
            "cpu-power-exporter-aarch64-unknown-linux-musl",
        },
    }
    for name, assets in expected_assets.items():
        matrix = jobs[name]["strategy"]["matrix"]["include"]
        assert {item["platform"] for item in matrix} == {"linux/amd64", "linux/arm64"}
        assert {item["asset"] for item in matrix} == assets


def test_first_release_builds_both_binaries(history: History) -> None:
    result, outputs = history.run("plan", history.git("rev-parse", "HEAD"))
    assert_success(result)
    assert outputs == {
        "publish": "true",
        "previous": "",
        "tag": "v1.0.0",
        "tachometer": "true",
        "cpu_power_exporter": "true",
    }


@pytest.mark.parametrize("changed,expected", [(SCRAPER, ("true", "false")), (EXPORTER, ("false", "true"))])
def test_failed_intervening_release_is_included(history: History, changed: str, expected: tuple[str, str]) -> None:
    history.git("tag", "v1.0.0")
    history.commit({changed: "unreleased binary change\n"})
    target = history.commit({"README.md": "later documentation-only push\n"})
    result, outputs = history.run("plan", target, [released()])
    assert_success(result)
    assert (outputs["tachometer"], outputs["cpu_power_exporter"]) == expected
    assert outputs["tag"] == "v1.0.1"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("Cargo.toml", ("true", "true")),
        ("Cargo.lock", ("true", "true")),
        ("rust-toolchain.toml", ("true", "true")),
        (".dockerignore", ("true", "true")),
        (".github/workflows/release.yaml", ("true", "true")),
        ("docker/Dockerfile.tachometer-scraper", ("true", "false")),
        ("docker/Dockerfile.cpu-power-exporter", ("false", "true")),
        ("Cargo.toml.example", ("false", "false")),
    ],
)
def test_binary_inputs(history: History, path: str, expected: tuple[str, str]) -> None:
    history.git("tag", "v1.0.0")
    target = history.commit({path: "changed\n"})
    result, outputs = history.run("plan", target, [released()])
    assert_success(result)
    assert (outputs["tachometer"], outputs["cpu_power_exporter"]) == expected


def test_rename_out_of_binary_tree_still_rebuilds(history: History) -> None:
    history.git("tag", "v1.0.0")
    target = history.commit({SCRAPER: None, "archive/main.rs": "scraper\n"})
    result, outputs = history.run("plan", target, [released()])
    assert_success(result)
    assert outputs["tachometer"] == "true"


def test_highest_published_version_is_baseline(history: History) -> None:
    history.git("tag", "v1.0.9")
    history.git("tag", "-a", "v1.0.10", "-m", "annotated release")
    target = history.commit({"README.md": "docs\n"})
    result, outputs = history.run(
        "plan", target, [released("v1.0.9"), released("v9.0.0", draft=True), released("v1.0.10")]
    )
    assert_success(result)
    assert outputs["previous"] == "v1.0.10"
    # The existing bump policy includes drafts; only the binary baseline excludes them.
    assert outputs["tag"] == "v9.0.1"
    assert outputs["tachometer"] == outputs["cpu_power_exporter"] == "false"


@pytest.mark.parametrize("source_repo", ["contributor/srt-slurm", REPOSITORY])
def test_feature_label_on_matching_merged_pr(history: History, source_repo: str) -> None:
    history.git("tag", "v1.0.0")
    target = history.commit({"README.md": "feature\n"})
    pr = associated_pr(target)
    pr["head"]["repo"]["full_name"] = source_repo
    result, outputs = history.run("plan", target, [released()], [[], [pr]])
    assert_success(result)
    assert outputs["tag"] == "v1.1.0"


@pytest.mark.parametrize("mismatch", ["unmerged", "sha", "branch", "repository", "label"])
def test_unrelated_pr_does_not_change_version_policy(history: History, mismatch: str) -> None:
    history.git("tag", "v1.0.0")
    target = history.commit({"README.md": "docs\n"})
    pr = associated_pr(target)
    if mismatch == "unmerged":
        pr["merged_at"] = None
    elif mismatch == "sha":
        pr["merge_commit_sha"] = "0" * 40
    elif mismatch == "branch":
        pr["base"]["ref"] = "development"
    elif mismatch == "repository":
        pr["base"]["repo"]["full_name"] = "other/srt-slurm"
    else:
        pr["labels"] = [{"name": "new-feature-extra"}]
    result, outputs = history.run("plan", target, [released()], [[pr]])
    assert_success(result)
    assert outputs["tag"] == "v1.0.1"


@pytest.mark.parametrize("step", ["plan", "guard"])
@pytest.mark.parametrize("relationship", ["equal", "ancestor"])
def test_already_published_targets_are_skipped(history: History, step: str, relationship: str) -> None:
    target = history.git("rev-parse", "HEAD")
    if relationship == "ancestor":
        history.commit({"README.md": "newer release\n"})
    history.git("tag", "v1.0.0")
    result, outputs = history.run(step, target, [released()], previous="v1.0.0")
    assert_success(result)
    assert outputs["publish"] == "false"


@pytest.mark.parametrize("step", ["plan", "guard"])
def test_divergent_release_fails_closed(history: History, step: str) -> None:
    base = history.git("rev-parse", "HEAD")
    history.commit({"README.md": "released branch\n"})
    history.git("tag", "v1.0.0")
    history.git("checkout", "--quiet", "-b", "divergent", base)
    target = history.commit({"README.md": "different branch\n"})
    result, outputs = history.run(step, target, [released()], previous="v1.0.0")
    assert result.returncode != 0
    assert outputs.get("publish") != "true"


@pytest.mark.parametrize("step,fail", [("plan", "release"), ("plan", "api"), ("guard", "release")])
def test_api_failures_do_not_publish(history: History, step: str, fail: str) -> None:
    history.git("tag", "v1.0.0")
    target = history.commit({"README.md": "docs\n"})
    result, outputs = history.run(step, target, [released()], previous="v1.0.0", fail=fail)
    assert result.returncode != 0
    assert outputs.get("publish") != "true"


def test_guard_rejects_changed_baseline_in_partial_rerun(history: History) -> None:
    history.git("tag", "v1.0.0")
    history.commit({SCRAPER: "newly released\n"})
    history.git("tag", "v1.0.1")
    target = history.commit({"README.md": "pending target\n"})
    result, outputs = history.run("guard", target, [released("v1.0.1")], previous="v1.0.0", tag="v1.0.2")
    assert result.returncode != 0
    assert outputs.get("publish") != "true"


def test_guard_rejects_existing_tag_at_other_commit(history: History) -> None:
    history.git("tag", "v1.0.0")
    history.git("tag", "v1.0.1")
    target = history.commit({"README.md": "pending target\n"})
    result, outputs = history.run("guard", target, [released()], previous="v1.0.0")
    assert result.returncode != 0
    assert outputs.get("publish") != "true"


@pytest.mark.parametrize("existing_tag", [False, True])
def test_guard_allows_unchanged_baseline_and_matching_tag(history: History, existing_tag: bool) -> None:
    history.git("tag", "v1.0.0")
    target = history.commit({"README.md": "pending target\n"})
    if existing_tag:
        history.git("tag", "v1.0.1")
    result, outputs = history.run("guard", target, [released()], previous="v1.0.0")
    assert_success(result)
    assert outputs["publish"] == "true"


def test_guard_allows_first_release(history: History) -> None:
    result, outputs = history.run("guard", history.git("rev-parse", "HEAD"), tag="v1.0.0")
    assert_success(result)
    assert outputs["publish"] == "true"


def test_binary_change_after_hundreds_of_unpublished_files(history: History) -> None:
    history.git("tag", "v1.0.0")
    changes = {f"docs/page-{index}.md": "documentation\n" for index in range(350)}
    changes[SCRAPER] = "unreleased binary change\n"
    target = history.commit(changes)
    result, outputs = history.run("plan", target, [released()])
    assert_success(result)
    assert outputs["tachometer"] == "true"
    assert outputs["cpu_power_exporter"] == "false"


@pytest.mark.parametrize("step", ["plan", "guard"])
def test_missing_published_tag_fails_closed(history: History, step: str) -> None:
    result, outputs = history.run(step, history.git("rev-parse", "HEAD"), [released()], previous="v1.0.0")
    assert result.returncode != 0
    assert outputs.get("publish") != "true"
