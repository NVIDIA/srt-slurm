# Review guide

Criteria for reviewing a PR to srt-slurm, by a person or an agent. The Design Rules in `CLAUDE.md` and the subsystem `AGENTS.md` files are the reference; this file says what to check and how much it matters.

## Blocking

- **A Design Rule is broken.** Name the rule and the existing mechanism to use instead. Typical findings: a new `if x.type == "<name>"` outside the owning registry, `getattr`/`hasattr` on a backend or frontend, a port derived from another port instead of a `PortKind`, a cluster difference hard-coded in Python instead of a `ClusterConfig` field, a second reader of an overridable setting, a new type that is really a mode.
- **A Design Rule baseline grew.** Entries in `tests/test_design_rules.py` may only be deleted.
- **The feature is incomplete.** User-visible config or behavior needs a test (dry-run for visible config, mock orchestrator for behavior), a `docs/` page or section, an example under `examples/`, and a regenerated `docs/schema-reference.md`.
- **The launch changed without a snapshot diff, or the snapshot diff is unexplained.** `tests/snapshots/launch/` shows every srun a recipe starts. Read it as the cluster-side effect of the PR: each changed env var, mount, flag, or placement must be intended and explained in the description.
- **Upstream behavior is assumed, not cited.** Code that depends on what an upstream flag, endpoint, or connector does needs the upstream source at the version the container ships, linked from the PR.
- **A generated file was edited by hand**, or `srtctl schema-docs --check`, Ruff, `ty`, or the tests fail.
- **Secrets or internal references.** No literal tokens in config fields (they land in lockfiles and logs), and no internal hostnames, private links, or session references in code, docs, or the PR description.

## Non-blocking

- Naming, comment wording, and small structure changes that do not affect behavior. Mark them as nits.
- A cleaner way to write the same logic when the PR's version is correct and tested.
- Follow-up work outside the PR's scope. Suggest an issue.

## How to review

1. Read the description, then the diff of `tests/snapshots/launch/` and `docs/schema-reference.md` first: they show the user-visible and cluster-visible effect.
2. Check each touched directory's `AGENTS.md` for its rules.
3. For behavior claims, look for the test that proves them; ask for one when it is missing.
4. When the same comment comes up in more than one PR, propose turning it into a Design Rule, a check in `tests/test_design_rules.py`, or a line in this file.
