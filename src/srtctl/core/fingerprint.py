# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime environment fingerprinting and comparison.

Captures the actual state of a container environment (pip packages, GPU info,
framework versions) for reproducibility and debugging. Every operation is
fault-tolerant — probes that fail are logged and skipped, never fatal.

Three main capabilities:
- **capture**: Collect environment fingerprint inside a running container
- **diff**: Compare two fingerprints and produce a structured delta
- **check**: Verify current environment matches a reference fingerprint

Design principles:
- Every probe can fail independently (returns sentinel, never raises)
- All output is deterministically sorted for clean diffs
- Fast — total capture time is ~2 seconds
- Pure functions where possible for testability
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Timeout for external commands (seconds)
_CMD_TIMEOUT = 5

# Sentinel for probes that failed
UNAVAILABLE = "unavailable"


# ============================================================================
# Data Types
# ============================================================================


class CheckStatus(str, Enum):
    """Status of a single check in a comparison."""

    OK = "ok"
    MISMATCH = "mismatch"
    MISSING = "missing"
    EXTRA = "extra"
    ERROR = "error"


@dataclass(frozen=True)
class ProbeResult:
    """Result of a single environment probe."""

    value: Any
    ok: bool = True
    error: str | None = None

    @staticmethod
    def success(value: Any) -> ProbeResult:
        return ProbeResult(value=value, ok=True)

    @staticmethod
    def failure(error: str) -> ProbeResult:
        return ProbeResult(value=UNAVAILABLE, ok=False, error=error)


@dataclass(frozen=True)
class PackageDiff:
    """Diff of a single pip package between two fingerprints."""

    package: str
    status: CheckStatus
    version_a: str | None = None
    version_b: str | None = None


@dataclass(frozen=True)
class FingerprintDiff:
    """Structured diff between two fingerprints."""

    # Scalar field diffs: field_name -> (value_a, value_b)
    field_changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Package-level diffs (sorted by package name)
    package_diffs: list[PackageDiff] = field(default_factory=list)
    # Fields present in both with identical values
    matching_fields: list[str] = field(default_factory=list)
    # Package count summary
    packages_matched: int = 0
    packages_changed: int = 0
    packages_added: int = 0
    packages_removed: int = 0


# ============================================================================
# Fixed field order for deterministic output
# ============================================================================

# Keys are written in this order. Anything not in this list is appended
# alphabetically at the end.
_FIELD_ORDER = [
    # Identity
    "hostname",
    "timestamp",
    # Hardware + OS
    "arch",
    "os",
    "gpu",
    # Core versions
    "python_version",
    "cuda_version",
    "torch_version",
    "nccl_version",
    # Full package list (always last)
    "pip_packages",
]


def _ordered_fingerprint(data: dict[str, Any]) -> dict[str, Any]:
    """Reorder fingerprint dict to canonical field order."""
    ordered: dict[str, Any] = {}
    for key in _FIELD_ORDER:
        if key in data:
            ordered[key] = data[key]
    # Append any extra keys alphabetically
    for key in sorted(data.keys()):
        if key not in ordered:
            ordered[key] = data[key]
    return ordered


# ============================================================================
# Probes — each returns ProbeResult, never raises
# ============================================================================


def _run_cmd(cmd: str, timeout: int = _CMD_TIMEOUT) -> str | None:
    """Run a shell command, return stdout or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except subprocess.TimeoutExpired:
        logger.debug("Command timed out (%ds): %s", timeout, cmd)
        return None
    except Exception as e:
        logger.debug("Command failed: %s — %s", cmd, e)
        return None


def probe_hostname() -> ProbeResult:
    """Get the hostname."""
    try:
        import socket

        return ProbeResult.success(socket.gethostname())
    except Exception as e:
        return ProbeResult.failure(str(e))


def probe_timestamp() -> ProbeResult:
    """Get current UTC timestamp in ISO format."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return ProbeResult.success(ts)
    except Exception as e:
        return ProbeResult.failure(str(e))


def probe_arch() -> ProbeResult:
    """Get CPU architecture."""
    try:
        return ProbeResult.success(platform.machine())
    except Exception as e:
        return ProbeResult.failure(str(e))


def probe_os() -> ProbeResult:
    """Get OS description from /etc/os-release."""
    try:
        p = Path("/etc/os-release")
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return ProbeResult.success(line.split("=", 1)[1].strip('"'))
        return ProbeResult.success(platform.platform())
    except Exception as e:
        return ProbeResult.failure(str(e))


def probe_gpu() -> ProbeResult:
    """Get GPU info from nvidia-smi (with timeout)."""
    out = _run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    if out is None:
        return ProbeResult.failure("nvidia-smi unavailable or timed out")

    gpus = []
    driver = UNAVAILABLE
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
            driver = parts[1]

    return ProbeResult.success({"available": True, "driver": driver, "gpus": gpus})


def probe_python_version() -> ProbeResult:
    """Get Python version."""
    try:
        return ProbeResult.success(platform.python_version())
    except Exception as e:
        return ProbeResult.failure(str(e))


def probe_cuda_version() -> ProbeResult:
    """Get CUDA toolkit version from nvcc."""
    out = _run_cmd("nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ,")
    if out:
        return ProbeResult.success(out)
    return ProbeResult.failure("nvcc not found")


def probe_torch_version() -> ProbeResult:
    """Get PyTorch version."""
    out = _run_cmd('python3 -c "import torch; print(torch.__version__)"')
    if out:
        return ProbeResult.success(out)
    return ProbeResult.failure("torch not importable")


def probe_nccl_version() -> ProbeResult:
    """Get NCCL version via PyTorch."""
    out = _run_cmd('python3 -c "import torch; print(torch.cuda.nccl.version())"')
    if out:
        return ProbeResult.success(out)
    return ProbeResult.failure("nccl version unavailable")


def probe_pip_packages() -> ProbeResult:
    """Get installed pip packages, sorted alphabetically (case-insensitive)."""
    out = _run_cmd("pip freeze")
    if out is None:
        return ProbeResult.failure("pip freeze failed")

    packages = sorted(
        [line.strip() for line in out.splitlines() if line.strip()],
        key=lambda s: s.lower(),
    )
    return ProbeResult.success(packages)


# ============================================================================
# Capture — run all probes, return ordered dict
# ============================================================================

# All probes in execution order
_PROBES: dict[str, Any] = {
    "hostname": probe_hostname,
    "timestamp": probe_timestamp,
    "arch": probe_arch,
    "os": probe_os,
    "gpu": probe_gpu,
    "python_version": probe_python_version,
    "cuda_version": probe_cuda_version,
    "torch_version": probe_torch_version,
    "nccl_version": probe_nccl_version,
    "pip_packages": probe_pip_packages,
}


def capture_fingerprint(extra_probes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run all probes and return an ordered fingerprint dict.

    Every probe is independent — failure of one never affects others.
    Failed probes are included with their sentinel value so diffs can
    distinguish "not installed" from "probe failed".

    Args:
        extra_probes: Optional additional probe functions to run.

    Returns:
        Ordered dict with canonical field order, ready for JSON serialization.
    """
    probes = dict(_PROBES)
    if extra_probes:
        probes.update(extra_probes)

    data: dict[str, Any] = {}
    for name, probe_fn in probes.items():
        try:
            result = probe_fn()
            data[name] = result.value
            if not result.ok:
                logger.debug("Probe %s failed: %s", name, result.error)
        except Exception as e:
            # Belt-and-suspenders: even if ProbeResult contract is violated
            data[name] = UNAVAILABLE
            logger.debug("Probe %s raised unexpectedly: %s", name, e)

    return _ordered_fingerprint(data)


def write_fingerprint(path: Path, extra_probes: dict[str, Any] | None = None) -> bool:
    """Capture fingerprint and write to a JSON file.

    Returns True on success, False on any failure (never raises).
    """
    try:
        data = capture_fingerprint(extra_probes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        return True
    except Exception as e:
        logger.debug("Failed to write fingerprint to %s: %s", path, e)
        return False


def load_fingerprint(path: Path) -> dict[str, Any] | None:
    """Load a fingerprint from a JSON file. Returns None on failure."""
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.debug("Failed to load fingerprint from %s: %s", path, e)
        return None


# ============================================================================
# Diff — compare two fingerprints
# ============================================================================

# Scalar fields that are compared for equality
_DIFF_FIELDS = [
    "arch",
    "os",
    "python_version",
    "cuda_version",
    "torch_version",
    "nccl_version",
]


def _parse_pip_packages(packages: list[str]) -> dict[str, str]:
    """Parse pip freeze output into {package_name: version} dict.

    Handles both == and @ formats:
        torch==2.6.0  ->  {"torch": "2.6.0"}
        foo @ file:///... -> {"foo": "file:///..."}
    """
    result = {}
    for line in packages:
        if "==" in line:
            name, version = line.split("==", 1)
            result[name.lower()] = version
        elif " @ " in line:
            name, location = line.split(" @ ", 1)
            result[name.lower()] = f"@ {location}"
        else:
            # Fallback: treat whole line as package with unknown version
            result[line.lower()] = "?"
    return result


def diff_fingerprints(a: dict[str, Any], b: dict[str, Any]) -> FingerprintDiff:
    """Compare two fingerprints and return a structured diff.

    Args:
        a: "Before" or "reference" fingerprint
        b: "After" or "current" fingerprint

    Returns:
        FingerprintDiff with field changes and package-level diffs.
    """
    field_changes: dict[str, tuple[str, str]] = {}
    matching_fields: list[str] = []

    # Compare scalar fields
    for field_name in _DIFF_FIELDS:
        val_a = str(a.get(field_name, UNAVAILABLE))
        val_b = str(b.get(field_name, UNAVAILABLE))
        if val_a == val_b:
            matching_fields.append(field_name)
        else:
            field_changes[field_name] = (val_a, val_b)

    # Compare GPU driver separately (nested)
    driver_a = _extract_driver(a)
    driver_b = _extract_driver(b)
    if driver_a == driver_b:
        matching_fields.append("gpu.driver")
    else:
        field_changes["gpu.driver"] = (driver_a, driver_b)

    # Compare pip packages
    pkgs_a = _parse_pip_packages(a.get("pip_packages", []))
    pkgs_b = _parse_pip_packages(b.get("pip_packages", []))

    all_packages = sorted(set(pkgs_a.keys()) | set(pkgs_b.keys()))
    package_diffs: list[PackageDiff] = []
    matched = 0
    changed = 0
    added = 0
    removed = 0

    for pkg in all_packages:
        in_a = pkg in pkgs_a
        in_b = pkg in pkgs_b

        if in_a and in_b:
            if pkgs_a[pkg] == pkgs_b[pkg]:
                matched += 1
            else:
                changed += 1
                package_diffs.append(
                    PackageDiff(
                        package=pkg,
                        status=CheckStatus.MISMATCH,
                        version_a=pkgs_a[pkg],
                        version_b=pkgs_b[pkg],
                    )
                )
        elif in_a and not in_b:
            removed += 1
            package_diffs.append(PackageDiff(package=pkg, status=CheckStatus.MISSING, version_a=pkgs_a[pkg]))
        else:
            added += 1
            package_diffs.append(PackageDiff(package=pkg, status=CheckStatus.EXTRA, version_b=pkgs_b[pkg]))

    return FingerprintDiff(
        field_changes=field_changes,
        package_diffs=package_diffs,
        matching_fields=matching_fields,
        packages_matched=matched,
        packages_changed=changed,
        packages_added=added,
        packages_removed=removed,
    )


def _extract_driver(fingerprint: dict[str, Any]) -> str:
    """Extract GPU driver version from fingerprint, handling nested structure."""
    gpu = fingerprint.get("gpu", {})
    if isinstance(gpu, dict):
        return str(gpu.get("driver", UNAVAILABLE))
    return UNAVAILABLE


# ============================================================================
# Check — verify current environment matches a reference
# ============================================================================


@dataclass(frozen=True)
class CheckResult:
    """Result of checking one aspect of the environment."""

    field: str
    status: CheckStatus
    message: str
    expected: str | None = None
    actual: str | None = None


def check_against_fingerprint(
    reference: dict[str, Any],
    current: dict[str, Any] | None = None,
) -> list[CheckResult]:
    """Check current environment against a reference fingerprint.

    If current is None, captures a fresh fingerprint first.

    Returns:
        List of CheckResult, one per field/package that differs.
        Empty list means everything matches.
    """
    if current is None:
        current = capture_fingerprint()

    diff = diff_fingerprints(reference, current)
    results: list[CheckResult] = []

    # Scalar field mismatches
    for field_name, (expected, actual) in diff.field_changes.items():
        if expected == UNAVAILABLE or actual == UNAVAILABLE:
            results.append(
                CheckResult(
                    field=field_name,
                    status=CheckStatus.ERROR,
                    message=f"{field_name}: could not compare ({expected} vs {actual})",
                    expected=expected,
                    actual=actual,
                )
            )
        else:
            results.append(
                CheckResult(
                    field=field_name,
                    status=CheckStatus.MISMATCH,
                    message=f"{field_name}: {expected} -> {actual}",
                    expected=expected,
                    actual=actual,
                )
            )

    # Package diffs
    for pkg_diff in diff.package_diffs:
        if pkg_diff.status == CheckStatus.MISMATCH:
            results.append(
                CheckResult(
                    field=f"pip:{pkg_diff.package}",
                    status=CheckStatus.MISMATCH,
                    message=f"{pkg_diff.package}: {pkg_diff.version_a} -> {pkg_diff.version_b}",
                    expected=pkg_diff.version_a,
                    actual=pkg_diff.version_b,
                )
            )
        elif pkg_diff.status == CheckStatus.MISSING:
            results.append(
                CheckResult(
                    field=f"pip:{pkg_diff.package}",
                    status=CheckStatus.MISSING,
                    message=f"{pkg_diff.package}=={pkg_diff.version_a} not installed",
                    expected=pkg_diff.version_a,
                )
            )
        elif pkg_diff.status == CheckStatus.EXTRA:
            results.append(
                CheckResult(
                    field=f"pip:{pkg_diff.package}",
                    status=CheckStatus.EXTRA,
                    message=f"{pkg_diff.package}=={pkg_diff.version_b} is extra (not in reference)",
                    actual=pkg_diff.version_b,
                )
            )

    return results


# ============================================================================
# Bash preamble generation — for injection into worker startup
# ============================================================================


def generate_capture_script(output_path: str) -> str:
    """Generate a bash one-liner that captures a fingerprint inside a container.

    The generated command:
    - Runs the fingerprint capture as a Python script
    - Is wrapped in || true so it never blocks the worker
    - Takes ~2 seconds

    Args:
        output_path: Path inside the container where fingerprint JSON is written,
                     e.g. "/logs/fingerprint_prefill_w0.json"

    Returns:
        Bash command string safe for inclusion in a preamble chain.
    """
    # We write a temp Python script and execute it, rather than using
    # python3 -c with inline code. This avoids escaping nightmares when
    # the command passes through bash → srun → bash → python.
    script = f"""\
import json, subprocess, platform, socket, sys
from pathlib import Path
from datetime import datetime, timezone

def run(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def pip_pkgs():
    out = run('pip freeze')
    if not out: return []
    return sorted([l.strip() for l in out.splitlines() if l.strip()], key=lambda s: s.lower())

def gpu_info():
    out = run('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader')
    if not out: return {{'available': False}}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            gpus.append({{'name': parts[0], 'driver': parts[1], 'memory': parts[2]}})
    return {{'available': True, 'driver': gpus[0]['driver'] if gpus else 'unknown', 'gpus': gpus}}

fp = {{
    'hostname': socket.gethostname(),
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'arch': platform.machine(),
    'os': next((l.split('=',1)[1].strip('"') for l in Path('/etc/os-release').read_text().splitlines() if l.startswith('PRETTY_NAME=')), platform.platform()) if Path('/etc/os-release').exists() else platform.platform(),
    'gpu': gpu_info(),
    'python_version': platform.python_version(),
    'cuda_version': run('nvcc --version 2>/dev/null | grep release') or 'unavailable',
    'torch_version': run('python3 -c "import torch; print(torch.__version__)"') or 'unavailable',
    'nccl_version': run('python3 -c "import torch; print(torch.cuda.nccl.version())"') or 'unavailable',
    'pip_packages': pip_pkgs(),
}}
Path('{output_path}').parent.mkdir(parents=True, exist_ok=True)
Path('{output_path}').write_text(json.dumps(fp, indent=2) + '\\n')
"""
    # Use a heredoc to write the script to a temp file, then execute it.
    # This is immune to quoting/escaping issues in the bash → srun chain.
    return f"python3 <(cat <<'__FINGERPRINT_EOF__'\n{script}__FINGERPRINT_EOF__\n) || true"


# ============================================================================
# Formatting — human-readable output for CLI
# ============================================================================


def format_diff(diff: FingerprintDiff, verbose: bool = False) -> str:
    """Format a FingerprintDiff as human-readable text for terminal output.

    Args:
        diff: The diff to format.
        verbose: If True, show all package changes. If False, summarize.

    Returns:
        Formatted string ready to print.
    """
    lines: list[str] = []

    # Scalar field changes
    if diff.field_changes:
        lines.append("Runtime changes:")
        for field_name, (val_a, val_b) in diff.field_changes.items():
            lines.append(f"  {field_name:20s} {val_a}  ->  {val_b}")
        lines.append("")

    # Matching fields
    if diff.matching_fields:
        lines.append("Unchanged:")
        for field_name in diff.matching_fields:
            lines.append(f"  {field_name}")
        lines.append("")

    # Package summary
    lines.append(
        f"Packages: {diff.packages_matched} match, "
        f"{diff.packages_changed} changed, "
        f"{diff.packages_added} added, "
        f"{diff.packages_removed} removed"
    )

    # Package details
    changed = [d for d in diff.package_diffs if d.status == CheckStatus.MISMATCH]
    added = [d for d in diff.package_diffs if d.status == CheckStatus.EXTRA]
    removed = [d for d in diff.package_diffs if d.status == CheckStatus.MISSING]

    show_limit = None if verbose else 10

    if changed:
        lines.append("")
        lines.append(f"  Version changes ({len(changed)}):")
        for d in changed[:show_limit]:
            lines.append(f"    {d.package}: {d.version_a}  ->  {d.version_b}")
        if show_limit and len(changed) > show_limit:
            lines.append(f"    ... and {len(changed) - show_limit} more (use --verbose)")

    if added:
        lines.append("")
        lines.append(f"  Added ({len(added)}):")
        for d in added[:show_limit]:
            lines.append(f"    + {d.package}=={d.version_b}")
        if show_limit and len(added) > show_limit:
            lines.append(f"    ... and {len(added) - show_limit} more")

    if removed:
        lines.append("")
        lines.append(f"  Removed ({len(removed)}):")
        for d in removed[:show_limit]:
            lines.append(f"    - {d.package}=={d.version_a}")
        if show_limit and len(removed) > show_limit:
            lines.append(f"    ... and {len(removed) - show_limit} more")

    return "\n".join(lines)


def format_check_results(results: list[CheckResult]) -> str:
    """Format check results as human-readable text.

    Args:
        results: List of check results from check_against_fingerprint.

    Returns:
        Formatted string. Empty results produce "Environment matches fingerprint."
    """
    if not results:
        return "Environment matches fingerprint."

    lines = [f"{len(results)} mismatches found:", ""]

    # Group by type
    field_results = [r for r in results if not r.field.startswith("pip:")]
    pip_results = [r for r in results if r.field.startswith("pip:")]

    if field_results:
        lines.append("Runtime:")
        for r in field_results:
            icon = _status_icon(r.status)
            lines.append(f"  {icon} {r.message}")
        lines.append("")

    if pip_results:
        mismatches = [r for r in pip_results if r.status == CheckStatus.MISMATCH]
        missing = [r for r in pip_results if r.status == CheckStatus.MISSING]
        extra = [r for r in pip_results if r.status == CheckStatus.EXTRA]

        lines.append("Packages:")
        if mismatches:
            lines.append(f"  {len(mismatches)} version mismatches:")
            for r in mismatches:
                lines.append(f"    {r.field.removeprefix('pip:')}: {r.expected} -> {r.actual}")
        if missing:
            lines.append(f"  {len(missing)} missing:")
            for r in missing:
                lines.append(f"    - {r.field.removeprefix('pip:')}=={r.expected}")
        if extra:
            lines.append(f"  {len(extra)} extra:")
            for r in extra:
                lines.append(f"    + {r.field.removeprefix('pip:')}=={r.actual}")

    return "\n".join(lines)


def _status_icon(status: CheckStatus) -> str:
    match status:
        case CheckStatus.OK:
            return "ok"
        case CheckStatus.MISMATCH:
            return "MISMATCH"
        case CheckStatus.MISSING:
            return "MISSING"
        case CheckStatus.EXTRA:
            return "EXTRA"
        case CheckStatus.ERROR:
            return "ERROR"
