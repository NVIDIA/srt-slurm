# Deferred DCGM 4.7 runtime support

## Status and scope

This document records the work needed for srt-slurm to consume a user-supplied
DCGM 4.7 ARM64 runtime package. It is a design checklist, not an implemented
feature. The current implementation uses the host ACPI `power_meter` hwmon
interface for CPU-side power collection. No DCGM downloader, package
installer, or host mutation is included in this branch.

A target-system validation enumerated both CPU entities and returned valid
watched values for field 1130. It also established that a one-shot live query
is insufficient for this field and that a complete, version-matched runtime
must be loaded.

## Proposed configuration contract

Add an optional runtime block beneath `telemetry.cpu_power` only when the DCGM
runtime feature is implemented:

```yaml
telemetry:
  cpu_power:
    enabled: true
    source: dcgm
    dcgm_runtime:
      package_url: https://packages.example.invalid/path/to/dcgm-arm64.deb
      package_sha256: <sha256-of-package>
      expected_version: <expected-dcgm-package-version>
```

The checksum and expected version must be mandatory whenever `package_url` is
set. Authentication credentials must come from the existing secret or CI
credential mechanism and must never be written into the recipe, logs, manifest,
or generated Slurm script.

## Required implementation

1. **Acquire and stage the package safely.** Download the package once on an
   authenticated host, verify its SHA-256, and stage it into the job. Prefer an
   AIB-side download over giving compute nodes a private GitLab token. Use an
   atomic, lock-protected cache when packages are shared between jobs.
2. **Do not install onto the host.** Extract the outer local-repository `.deb`
   and its required ARM64 packages with `dpkg-deb -x` into a job-scoped runtime
   prefix. Do not run `apt install` or `dpkg -i` against the host filesystem.
3. **Discover the repository by contents.** Locate the directory containing
   `libdcgm_*_arm64.deb`; do not assume the directory name inside the outer
   package.
4. **Require one coherent version.** Extract exactly one matching package for
   each of `libdcgm`, `nv-hostengine`, `dcgmi`,
   `datacenter-gpu-manager-4-module-config`,
   `datacenter-gpu-manager-4-module-sysmon`, and
   `datacenter-gpu-manager-4-python3`. Reject missing, ambiguous, wrong-arch, or
   mixed-version inputs.
5. **Build a private runtime environment.** Assemble the DCGM and SysMon shared
   libraries in one runtime library directory and set only the collector's
   `PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH`. Verify that the mapped
   `libdcgm.so` and `libdcgmmodulesysmon.so` originate from this prefix.
6. **Initialize the matching library explicitly.** Call the packaged bindings'
   low-level `_dcgmInit(<runtime-library-directory>)` before `dcgmInit()` and
   `dcgmStartEmbedded()`. Do not rely on the incomplete high-level helper path
   observed in the diagnostic.
7. **Use watched CPU values.** Enumerate `DCGM_FE_CPU` entities, create a group
   and field group for field 1130, call `dcgmWatchFields`, wait for the first
   update, and read with `dcgmEntitiesGetLatestValues(..., flags=0)`. A live
   query returned DCGM status `-32` in the validated environment.
8. **Keep power semantics explicit.** DCGM field 1130 is CPU-rail power, not the
   full CPU-side socket total exposed by hwmon. Mark the manifest aggregate
   scope as `cpu_rail_only`; do not silently compare or merge it with hwmon
   `Total Power`.
9. **Clean up deterministically.** Unwatch fields, delete field/entity groups,
   shut down the embedded engine, and remove only the job-scoped extraction
   directory. Preserve the downloaded package identity and runtime provenance
   in the manifest.

## Validation gates

Before enabling the feature for benchmarks, require automated tests for schema
validation, credential redaction, checksum rejection, wrong architecture,
mixed package versions, content-based repository discovery, environment
construction, and cleanup. A target-system integration test must additionally
prove:

- all expected CPU entities are enumerated on every selected node;
- field 1130 returns positive status-0 watched values;
- the mapped DCGM and SysMon libraries match the staged 4.7 runtime;
- sampling occurs once per node rather than once per model process;
- required-mode failures prevent publication of incomplete power results; and
- no package, token, or runtime mutation escapes the job directory.

After those gates pass, `auto` may continue preferring hwmon for CPU-side total
power and use DCGM 4.7 only as the CPU-rail fallback.
