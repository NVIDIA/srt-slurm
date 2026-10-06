---
name: add-config-field
description: Add, rename, or re-type a recipe or srtslurm.yaml field in srt-slurm so that it ships complete (schema, resolver, dry-run, docs, example, tests).
---

# Add or change a config field

1. **Decide where it lives.** A per-cluster or per-hardware value is a `ClusterConfig` field in `src/srtctl/core/schema.py`, read once into `RuntimeContext`. A per-recipe value goes on the owning dataclass (`SrtConfig` section, a backend dataclass in `src/srtctl/backends/`, `FrontendConfig`, a service config). A value a role can override gets one resolver on the backend in the `get_config_for_mode` style; every consumer calls it.
2. **Add the field** to a frozen dataclass with a default that keeps existing recipes unchanged. Describe it in the class docstring `Attributes:` block or in a `#` comment directly above the field.
3. **Validate** at load time (`__post_init__`, the schema validators, or `Frontend.validate`) with a message that names the field and the fix, so `srtctl dry-run` catches mistakes.
4. **Wire consumers** through the resolver or base-class member, never `getattr`. A new backend answer is a new `Backend` member, with an inherited neutral default when appropriate or an abstract hook when every backend needs its own implementation.
5. **srun-visible fields** (env, mounts, srun options, host setup): show them in `show_config_details()` in `src/srtctl/cli/submit.py` and add a `tests/test_dry_run.py` case.
6. **Regenerate generated docs:** `uv run srtctl schema-docs`. Never edit `docs/schema-reference.md` by hand.
7. **Document** the field in the relevant `docs/` page and use it in an example under `examples/` when it is user-facing.
8. **Snapshots:** `make snapshots`. Commit any `tests/snapshots/launch/` change and make sure each diff line is intended.
9. **Check:** `make check` (Ruff, blocking `ty`, schema-doc check, all tests).
