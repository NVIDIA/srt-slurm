# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark manifest schema (Pydantic v2 models).

These models ARE the schema. They validate structure and types, and they emit
JSON Schema via ``BenchmarkManifest.model_json_schema()`` for external consumers
(see ``export_schema.py``).

Design notes
------------
* Required core vs recommended. Fields that BLOCK reproduction are required.
  Fields that are strongly recommended but not blocking are ``Optional`` with a
  ``None`` default. Keeping the required set small is deliberate — it is what
  keeps authors (and the generator) from being overwhelmed.

* Policy lives in the validator, not here. The models stay structural: digests
  and commits are plain strings here so the validator can emit precise,
  human-readable failures ("floating tag, not a digest") instead of an opaque
  regex mismatch. See ``validate.py``.

* Generated vs authored. Most fields are meant to be machine-emitted from a
  completed run (results file, engine config, batch script, image SBOM). Only a
  small set is authored by a human: owners, capability invocations, and the
  conversion compatibility window. Field docstrings note which is which.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_VERSION = "0.1"


class _Model(BaseModel):
    """Base for all manifest models: reject unknown keys so drift is caught."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


class Owners(_Model):
    """Named owners on both sides of the handoff (authored)."""

    trtllm: str = Field(..., description="Owner on the TRT-LLM (native) side.")
    dynamo: str = Field(..., description="Owner on the Dynamo (replicating) side.")


class Metadata(_Model):
    name: str = Field(..., description="Short benchmark name, e.g. 'dsv4-pro-8k1k-b300'.")
    description: str | None = Field(None, description="One-line human description.")
    owners: Owners


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


class ImageRef(_Model):
    """A container image, pinned by content digest (generated)."""

    ref: str = Field(..., description="Human-readable image reference, e.g. 'nvcr.io/.../release:1.3.0rc23'.")
    digest: str = Field(..., description="Content digest, 'sha256:<64 hex>'. The validator rejects a bare tag.")


class WheelRef(_Model):
    """A Python wheel, pinned by SHA or git commit (generated)."""

    name: str = Field(..., description="Distribution name, e.g. 'ai-dynamo'.")
    version: str | None = Field(None, description="Version string, if any (informational, NOT the pin).")
    sha256: str | None = Field(None, description="Artifact sha256. Provide this OR git_commit.")
    git_commit: str | None = Field(None, description="Source commit the wheel was built from. Provide this OR sha256.")


class FileRef(_Model):
    """A file inside the srt-slurm tree, referenced by path + hash (generated)."""

    srt_slurm_relpath: str = Field(..., description="Path relative to the srt-slurm repo root.")
    sha256: str = Field(..., description="sha256 of the file at the recorded commit.")


class SbomRef(FileRef):
    """Software bill of materials for the built image (recommended, generated).

    A dependency lock (e.g. ``pip freeze`` captured inside the built image). This
    is what catches the incompatible-nested-dependency failure class.
    """


class ConversionInputs(_Model):
    """Inputs that pin how the Dynamo image is produced (authored + generated)."""

    trtllm_image_digest: str = Field(..., description="Base TRT-LLM image digest to convert from.")
    trtllm_source_commit: str = Field(..., description="TRT-LLM source commit the base image was built at.")
    dynamo_wheel: WheelRef = Field(..., description="Dynamo wheel to bake in (pinned, not runtime-installed).")
    compat_min_commit: str = Field(..., description="Low end of the TRT-LLM source-commit range the wheel supports.")
    compat_max_commit: str = Field(..., description="High end of that range ('HEAD@<date>' allowed).")


class ConversionOutputs(_Model):
    """Outputs of the conversion (generated once the image is built; optional)."""

    dynamo_image_digest: str | None = Field(None, description="Digest of the built Dynamo image, once baked.")
    build_command: str | None = Field(None, description="Exact command that produced it.")
    preserved_comm: list[str] = Field(
        default_factory=list,
        description="Engine-provided comm libs that MUST survive conversion, e.g. ['nixl', 'ucx', 'libfabric'].",
    )


class Conversion(_Model):
    """The image-conversion I/O contract for this run.

    This fixes what the conversion must satisfy. Actually running the conversion
    is a separate ticket; only the contract is pinned here.
    """

    inputs: ConversionInputs
    outputs: ConversionOutputs = Field(default_factory=ConversionOutputs)


class Artifacts(_Model):
    base_image: ImageRef = Field(..., description="TRT-LLM base image, by digest.")
    wheels: list[WheelRef] = Field(default_factory=list, description="Wheels installed on top, pinned.")
    sbom: SbomRef | None = Field(None, description="Built-image dependency lock (recommended).")


# ---------------------------------------------------------------------------
# source
# ---------------------------------------------------------------------------


class Source(_Model):
    """Where the recipe/client/sweep scripts come from (generated)."""

    repo: str = Field(..., description="srt-slurm repository URL.")
    branch: str = Field(..., description="Branch name.")
    commit: str = Field(..., description="40-hex commit SHA. The validator rejects short or non-hex values.")
    bundle: list[FileRef] = Field(
        default_factory=list,
        description="Recipe/client/sweep files by relpath + sha256 at that commit (SHA-reference, not a copy).",
    )
    launcher_patches: list[str] = Field(
        default_factory=list,
        description="Out-of-band edits applied at launch, declared explicitly (e.g. an ENROOT_REMAP_ROOT sed).",
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class Checkpoint(_Model):
    """Checkpoint by a shareable reference — never a private local path (generated).

    Provide EITHER (hf_id + hf_revision) OR oci_digest. The validator enforces
    the one-of and rejects anything that looks like a bare filesystem path.
    """

    hf_id: str | None = Field(None, description="Hugging Face model id, e.g. 'deepseek-ai/DeepSeek-V4-Pro'.")
    hf_revision: str | None = Field(None, description="Hugging Face revision (commit/tag) pinning the weights.")
    oci_digest: str | None = Field(None, description="OCI/registry digest, if the checkpoint is stored that way.")


class Model(_Model):
    served: str = Field(..., description="Model name the server actually served (from the engine/worker logs).")
    requested: str = Field(..., description="Model name the client requested. Validator flags served != requested.")
    checkpoint: Checkpoint


# ---------------------------------------------------------------------------
# topology + runtime
# ---------------------------------------------------------------------------


class WorkerGroup(_Model):
    """One side of a disaggregated deployment (generated)."""

    num_workers: int = Field(..., ge=1)
    tp: int = Field(..., ge=1, description="Tensor-parallel size.")
    ep: int | None = Field(None, ge=1, description="Expert-parallel size, if applicable.")
    dp_attn: int | None = Field(None, ge=1, description="Data-parallel attention size, if applicable.")
    nodes: int | None = Field(None, ge=0, description="Nodes allocated to this group.")


class Topology(_Model):
    prefill: WorkerGroup
    decode: WorkerGroup


class BackendSelection(_Model):
    """Requested vs effective backend — the silent-fallback guard (generated).

    ``requested`` is what the recipe asked for; ``effective`` is what the engine
    actually ran, read from its logs. The validator fails if they differ.
    """

    requested: str
    effective: str


class Backends(_Model):
    moe: BackendSelection = Field(..., description="MoE kernel backend, e.g. TRTLLM vs MEGAMOE_DEEPGEMM.")
    comm: BackendSelection = Field(..., description="KV/comm transfer backend, e.g. NIXL vs UCX.")


class ClusterRequirements(_Model):
    """Cluster-specific settings that a naive replay would miss (recommended)."""

    ucx: dict = Field(default_factory=dict, description="UCX env/settings that must be reproduced.")
    numa: dict = Field(default_factory=dict, description="GPU/NIC/CPU affinity requirements.")
    mounts: list[str] = Field(default_factory=list)
    preserve_engine_comm: list[str] = Field(
        default_factory=list, description="Comm libs from the engine image that must not be overwritten."
    )


class Runtime(_Model):
    effective_config: FileRef = Field(
        ..., description="Reference + hash to the engine's resolved config (trtllm_config_*.yaml), not recipe deltas."
    )
    backends: Backends
    environment: dict | None = Field(None, description="Effective environment (recommended).")
    cluster_requirements: ClusterRequirements | None = Field(None, description="Recommended.")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


class Metrics(_Model):
    """Per-point measured metrics. Absolute latencies are the portable ones."""

    concurrency: int = Field(..., ge=1)
    ttft_ms: float | None = Field(None, description="Mean time-to-first-token, ms.")
    tpot_ms: float | None = Field(None, description="Mean time-per-output-token, ms.")
    output_tput_tok_s: float | None = Field(None, description="Output tokens/sec.")
    total_tput_tok_s: float | None = Field(None, description="Total (in+out) tokens/sec, if reported.")


class ResultPoint(_Model):
    """One reported number within this run's single, fully-pinned configuration.

    A manifest describes one configuration (topology + backends + effective_config),
    so ``point_id`` is the stable label identifying a point within that run (e.g. a
    concurrency sweep step). The validator enforces it is present and unique; it does
    not resolve it against a separate config table.
    """

    point_id: str = Field(..., description="Stable, unique label for this point, e.g. '2p1d-b32-conc282'.")
    metrics: Metrics


class Results(_Model):
    baseline: list[ResultPoint] = Field(
        default_factory=list, description="Native baseline points, each with a unique point_id within this run."
    )


# ---------------------------------------------------------------------------
# capabilities (Dynamo-side; authored + gated on the Dynamo team)
# ---------------------------------------------------------------------------


class CapabilityDecl(_Model):
    available: bool = Field(..., description="Whether this capability exists today. If false, it is a declared gap.")
    invocation: str | None = Field(None, description="How to invoke it (required when available is true).")
    notes: str | None = None


class Capabilities(_Model):
    worker_without_frontend: CapabilityDecl = Field(
        ..., description="A Dynamo worker that answers requests without the Dynamo frontend/router."
    )
    prefill_only_discovery: CapabilityDecl = Field(..., description="Dynamo-frontend prefill-only topology discovery.")


# ---------------------------------------------------------------------------
# top-level manifest
# ---------------------------------------------------------------------------


class BenchmarkManifest(_Model):
    """A complete, portable description of one benchmark run."""

    manifest_version: str = Field(
        ...,
        description=f"Schema version, e.g. '{MANIFEST_VERSION}'. Required; the validator rejects a major it cannot read.",
    )
    metadata: Metadata
    artifacts: Artifacts
    source: Source
    model: Model
    topology: Topology
    runtime: Runtime
    conversion: Conversion
    results: Results = Field(default_factory=Results)
    capabilities: Capabilities | None = Field(None, description="Dynamo capability declarations (recommended).")
