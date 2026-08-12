# Portable benchmark manifest

A single, machine-readable file that fully describes a benchmark run, so a
different team can reproduce the native baseline and build a matched image with
**no private paths, out-of-band files, or tribal knowledge**.

This directory is the **schema + validator** skeleton. It defines the target a
handoff must hit; the generator that emits a manifest from a completed run is a
separate, not-yet-built piece.

## Layout

| File | What it is |
|------|------------|
| `models.py` | The manifest schema, as Pydantic v2 models. Source of truth. |
| `validate.py` | The validator: schema conformance + semantic consistency checks, with a `--online` stub. |
| `errors.py` | Structured issue types + stable `IssueCode`s the validator emits. |
| `export_schema.py` | Emits `schema/benchmark-manifest.schema.json` from the models. |
| `schema/benchmark-manifest.schema.json` | Generated JSON Schema, for non-Python consumers. Do not edit by hand. |

The reference example lives at
`examples/benchmark-manifest/dsv4-pro-8k1k-b300/manifest.yaml`, and tests are in
`tests/test_benchmark_manifest.py`.

## Use

```bash
# validate a manifest
srtctl-validate-manifest path/to/manifest.yaml
# or, without installing:
uv run python -m srtctl.benchmark_manifest.validate path/to/manifest.yaml

# regenerate the JSON Schema after changing the models
uv run python -m srtctl.benchmark_manifest.export_schema \
    -o src/srtctl/benchmark_manifest/schema/benchmark-manifest.schema.json
```

Exit code is non-zero if there is any **error**-severity issue. Warnings (e.g. a
missing SBOM) do not fail the run.

## What the validator checks today (static, no network)

| Failure mode | `IssueCode` |
|---|---|
| Image (or OCI checkpoint) pinned by a tag, not a digest | `FLOATING_TAG` |
| Malformed `sha256:` digest, or a bad bare 64-hex sha256 (wheel / bundle file / effective_config / SBOM) | `BAD_DIGEST_FORMAT` |
| Wheel with only a version string (no sha/commit) | `UNPINNED_WHEEL` |
| Commit not a full 40-hex sha — `source.commit`, wheel `git_commit`, conversion source/compat commits | `BAD_COMMIT` / `MISSING_COMMIT` |
| `manifest_version` major this validator cannot read | `UNSUPPORTED_MANIFEST_VERSION` |
| Served model name != requested | `MODEL_NAME_MISMATCH` |
| Checkpoint given as a local path (absolute or mount-relative) | `PRIVATE_PATH` |
| Checkpoint with neither HF id+revision nor OCI digest | `CHECKPOINT_UNDERSPECIFIED` |
| `backend.effective` != `backend.requested` (silent fallback) | `SILENT_BACKEND_FALLBACK` |
| Dynamo wheel not baked (floating runtime install) | `RUNTIME_INSTALL_WHEEL` |
| Result point with no / duplicate `point_id` | `DANGLING_CONFIG_TO_PERF` |
| Capability declared available but no invocation | `INCOMPLETE_CAPABILITY` |
| No SBOM recorded (warning, non-blocking) | `MISSING_SBOM` |
| Any structural/type error against the models | `SCHEMA_ERROR` |

Hashes are checked for *format* only (offline). Whether a `sha256:`/commit actually
resolves, and whether `trtllm_source_commit` falls inside the compatibility window,
is left to the `--online` layer (not yet implemented).

## Known gaps (deliberately visible, not hidden)

- **Generator** — not built. The manifest is meant to be machine-emitted from a
  run's own artifacts (`agg_*.json`, `trtllm_config_*.yaml`, batch script, image
  SBOM). Until it exists, manifests are hand-written and the strong
  silent-fallback / model-name guarantees depend on that generator capturing the
  *effective* values from the run logs — the static checker can only confirm the
  recorded effective matches the requested one.
- **`--online`** — not implemented. Registry/HF/git reachability checks emit a
  single `ONLINE_NOT_IMPLEMENTED` warning so the gap is visible.
- **Dynamo capabilities** — `worker_without_frontend` and
  `prefill_only_discovery` are declared in the schema but must be confirmed with
  the Dynamo team before they can be marked `available: true` with a real
  invocation.
- **Conversion execution** — the schema pins the conversion *contract* (inputs →
  outputs); actually running the conversion is a separate ticket.
