---
name: design-rule-sweep
description: Periodic cleanup of srt-slurm drift - shrink the Design Rule baselines, fix stale agent docs, and open small focused PRs.
---

# Design-rule sweep

1. **Baselines.** Run `uv run pytest tests/test_design_rules.py -v`. Pick one baselined site in `tests/test_design_rules.py`, fix it (typed field access, a protocol member, a frontend attribute, an allocated `PortKind`), delete its baseline entry, and run the suites for that module. If a fix needs test fakes to grow real fields, update the fakes rather than keeping the reflective access.
2. **New rules.** Look through recent review comments for a finding that recurs. If it can be checked on the AST, add a rule to `tests/test_design_rules.py` with a remediation message and a baseline of current sites; otherwise add a line to `REVIEW.md`.
3. **Stale guidance.** Run `uv run pytest tests/test_agent_docs.py -v` and read the `AGENTS.md` files of directories that changed recently against the code. Fix instructions that no longer match; link to `docs/` rather than copying explanations.
4. **Types.** `uv run ty check src/srtctl/` must stay clean; fix new diagnostics instead of suppressing them.
5. **One concern per PR.** Keep each sweep PR to one rule or one directory so it reviews quickly, and run `make check` before opening it.
