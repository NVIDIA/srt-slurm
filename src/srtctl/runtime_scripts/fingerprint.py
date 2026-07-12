#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture a worker container's runtime environment as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

COMMAND_TIMEOUT_SECONDS = 3
UNAVAILABLE = "unavailable"
DEFAULT_MODEL_PATH = Path("/model")
DEFAULT_CPUINFO_PATH = Path("/proc/cpuinfo")

FRAMEWORK_PACKAGES: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "tensorrt_llm": "tensorrt-llm",
    "dynamo": "ai-dynamo",
}

CPU_MODEL_KEYS = {"model name", "cpu model", "hardware"}
ARM_CPUINFO_KEYS = {
    "cpu implementer",
    "cpu architecture",
    "cpu variant",
    "cpu part",
    "cpu revision",
}
SLURM_CPU_ENV_KEYS = (
    "SLURM_CPUS_ON_NODE",
    "SLURM_CPUS_PER_GPU",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
)
ENV_PREFIXES = (
    "CUDA_",
    "TORCH_",
    "PYTORCH_",
    "NCCL_",
    "VLLM_",
    "SGLANG_",
    "SGL_",
    "TRTLLM_",
    "TRT_LLM_",
    "TENSORRT_",
    "HF_",
    "TRANSFORMERS_",
    "DYN_",
    "NVIDIA_",
    "OMPI_",
    "UCX_",
    "NVSHMEM_",
)
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
T = TypeVar("T")


def best_effort(probe: Callable[[], T], fallback: T) -> T:
    """Run one fingerprint probe without letting it abort the full capture."""
    try:
        return probe()
    except Exception:
        return fallback


def run(command: Sequence[str], timeout: int = COMMAND_TIMEOUT_SECONDS) -> str | None:
    """Run a probe command and return stripped stdout on success."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def find_python() -> str:
    """Prefer the Python environment used by common Dynamo containers."""
    for candidate in ("/opt/dynamo/venv/bin/python3", "/opt/venv/bin/python3"):
        if Path(candidate).exists():
            return candidate
    return "python3"


def pip_packages(python: str) -> dict[str, list[str]]:
    """Capture package inventories from every available package frontend."""
    result: dict[str, list[str]] = {}
    commands = (
        (python, [python, "-m", "pip", "freeze"]),
        ("python3", ["python3", "-m", "pip", "freeze"]),
        ("pip", ["pip", "freeze"]),
        ("uv", ["uv", "pip", "freeze"]),
    )
    for label, command in commands:
        output = run(command)
        if not output:
            continue
        packages = sorted(
            (line.strip() for line in output.splitlines() if line.strip() and not line.startswith("#")),
            key=str.lower,
        )
        if packages:
            result[label] = packages
    return result


def gpu_info() -> dict[str, object]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if not output:
        return {"available": False}

    gpus: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return {
        "available": True,
        "driver": gpus[0]["driver"] if gpus else "unknown",
        "gpus": gpus,
    }


def cpu_model_from_cpuinfo(cpuinfo: str) -> str | None:
    """Extract a useful CPU model from x86 or ARM ``/proc/cpuinfo`` text."""
    arm_fields: dict[str, str] = {}
    processor_label: str | None = None

    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if not normalized_value:
            continue
        if normalized_key in CPU_MODEL_KEYS:
            return normalized_value
        if normalized_key == "processor" and not normalized_value.isdecimal():
            processor_label = processor_label or normalized_value
        elif normalized_key in ARM_CPUINFO_KEYS:
            arm_fields.setdefault(normalized_key, normalized_value)

    if processor_label:
        return processor_label

    implementer = arm_fields.get("cpu implementer")
    part = arm_fields.get("cpu part")
    if not implementer and not part:
        return None

    pieces = ["ARM CPU"]
    if implementer:
        pieces.append(f"implementer {implementer}")
    if part:
        pieces.append(f"part {part}")

    details = (
        ("architecture", arm_fields.get("cpu architecture")),
        ("variant", arm_fields.get("cpu variant")),
        ("revision", arm_fields.get("cpu revision")),
    )
    detail_text = ", ".join(f"{label} {value}" for label, value in details if value)
    if detail_text:
        return f"{' '.join(pieces)} ({detail_text})"
    return " ".join(pieces)


def format_cpu_ids(cpu_ids: list[int]) -> str:
    if not cpu_ids:
        return ""

    ranges: list[str] = []
    start = previous = cpu_ids[0]
    for cpu_id in cpu_ids[1:]:
        if cpu_id == previous + 1:
            previous = cpu_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def cpu_info(
    cpuinfo_path: Path = DEFAULT_CPUINFO_PATH,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    model = UNAVAILABLE
    try:
        if cpuinfo_path.exists():
            model = cpu_model_from_cpuinfo(cpuinfo_path.read_text(errors="replace")) or UNAVAILABLE
    except OSError:
        pass

    try:
        affinity_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_ids = []

    runtime_env = os.environ if env is None else env
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(affinity_ids) if affinity_ids else None,
        "affinity_list": format_cpu_ids(affinity_ids) if affinity_ids else None,
        "slurm": {key: runtime_env[key] for key in SLURM_CPU_ENV_KEYS if key in runtime_env},
    }


def framework_versions(python: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, package in FRAMEWORK_PACKAGES.items():
        output = run(
            [
                python,
                "-c",
                f"import importlib.metadata; print(importlib.metadata.version({package!r}))",
            ]
        )
        if output:
            versions[name] = output
    return versions


def model_identity(model_path: Path) -> dict[str, str] | None:
    if not model_path.exists():
        return None

    info: dict[str, str] = {}
    for refs_path in (model_path / ".huggingface" / "refs" / "main", model_path / "refs" / "main"):
        if refs_path.exists():
            try:
                info["hf_revision"] = refs_path.read_text().strip()
                break
            except (OSError, UnicodeError):
                continue

    if "hf_revision" not in info:
        cache_download = model_path / ".cache" / "huggingface" / "download"
        if cache_download.is_dir():
            for metadata_file in sorted(cache_download.glob("*.metadata")):
                try:
                    first_line = metadata_file.read_text().splitlines()[0].strip()
                except (IndexError, OSError, UnicodeError):
                    continue
                if len(first_line) == 40 and all(character in "0123456789abcdef" for character in first_line):
                    info["hf_revision"] = first_line
                    break

    metadata_path = model_path / ".huggingface" / "download_metadata.json"
    if metadata_path.exists():
        metadata = read_json_object(metadata_path)
        revision = metadata.get("commit_hash")
        repo_id = metadata.get("repo_id")
        if isinstance(revision, str):
            info["hf_revision"] = revision
        if isinstance(repo_id, str):
            info["hf_repo"] = repo_id

    config_path = model_path / "config.json"
    if config_path.exists():
        config = read_json_object(config_path)
        model_id = config.get("_name_or_path")
        if isinstance(model_id, str):
            info["model_id"] = model_id

    return info or None


def read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object, treating malformed or differently shaped data as absent."""
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def environment_variables(env: Mapping[str, str] | None = None) -> dict[str, str]:
    runtime_env = os.environ if env is None else env
    result: dict[str, str] = {}
    for key, value in sorted(runtime_env.items()):
        if not any(key.startswith(prefix) for prefix in ENV_PREFIXES):
            continue
        result[key] = "***REDACTED***" if any(marker in key.upper() for marker in SECRET_MARKERS) else value
    return result


def os_description() -> str:
    os_release = Path("/etc/os-release")
    if os_release.exists():
        try:
            for line in os_release.read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip('"')
        except OSError:
            pass
    return platform.platform()


def cuda_version() -> str:
    output = run(["nvcc", "--version"])
    if output:
        for line in output.splitlines():
            if "release" in line:
                return line.strip()
    return UNAVAILABLE


def nccl_version(python: str) -> str:
    return run([python, "-c", "import torch; print(torch.cuda.nccl.version())"]) or UNAVAILABLE


def capture_fingerprint(
    model_path: Path = DEFAULT_MODEL_PATH,
    cpuinfo_path: Path = DEFAULT_CPUINFO_PATH,
) -> dict[str, object]:
    python = find_python()
    return {
        "hostname": best_effort(socket.gethostname, UNAVAILABLE),
        "timestamp": best_effort(
            lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            UNAVAILABLE,
        ),
        "arch": best_effort(platform.machine, UNAVAILABLE),
        "os": best_effort(os_description, UNAVAILABLE),
        "cpu": best_effort(
            lambda: cpu_info(cpuinfo_path),
            {
                "model": UNAVAILABLE,
                "logical_cpus": None,
                "affinity_cpus": None,
                "affinity_list": None,
                "slurm": {},
            },
        ),
        "gpu": best_effort(gpu_info, {"available": False}),
        "python_version": best_effort(platform.python_version, UNAVAILABLE),
        "cuda_version": best_effort(cuda_version, UNAVAILABLE),
        "nccl_version": best_effort(lambda: nccl_version(python), UNAVAILABLE),
        "frameworks": best_effort(lambda: framework_versions(python), {}),
        "model": best_effort(lambda: model_identity(model_path), None),
        "env": best_effort(environment_variables, {}),
        "pip_packages": best_effort(lambda: pip_packages(python), {}),
    }


def write_fingerprint(output_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(capture_fingerprint(model_path), indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fingerprint JSON output path")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Mounted model path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    write_fingerprint(args.output, args.model_path)


if __name__ == "__main__":
    main()
