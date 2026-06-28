#!/usr/bin/env python3
"""Apply vLLM PR #43729 to the Jun 27 nightly installed package."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import py_compile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

PR_COMMIT = "c4af560742ae17335d27e93b3ba167152a2b37b4"
NIGHTLY_BASE_COMMIT = "68ee8300a047db78fb52bac477daaaac7be11216"
NIGHTLY_SOURCE_SHA256 = "d3f16d2691eda49559b547303e89c9564ab60f783c6d8bee7c19fc2371cf390c"
PATCHED_SOURCE_SHA256 = "e489e64c572254890af42573e2731267aed5b5f554861036657458250e47c112"
MIN_FLASHINFER_VERSION = Version("0.6.12rc2")


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"PR #43729 patch expected one {description} marker, found {count}")
    return source.replace(old, new, 1)


def find_target() -> Path:
    override = os.environ.get("VLLM_PR43729_TARGET")
    if override:
        return Path(override).resolve()

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Unable to locate the installed vllm package")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "v1/attention/backends/mla/flashinfer_mla.py"


def verify_flashinfer() -> None:
    versions: dict[str, str] = {}
    for package in ("flashinfer-python", "flashinfer-cubin"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Required package {package} is not installed") from exc

    too_old = {
        package: installed for package, installed in versions.items() if Version(installed) < MIN_FLASHINFER_VERSION
    }
    if too_old:
        raise RuntimeError(f"vLLM PR #43729 requires FlashInfer >= 0.6.12rc2; found {too_old}")
    print(f"[vllm-pr43729] FlashInfer packages: {versions}")


def main() -> None:
    verify_flashinfer()
    target = find_target()
    if not target.is_file():
        raise RuntimeError(f"vLLM FlashInfer MLA source not found: {target}")

    source_bytes = target.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = source_bytes.decode()
    patched_markers = (
        "class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):\n    can_return_lse_for_decode: bool = True",
        "kernel_out = trtllm_batch_decode_with_kv_cache_mla(",
        "return_lse=return_lse,",
        "return o, lse",
    )
    marker_presence = tuple(marker in source for marker in patched_markers)
    if all(marker_presence):
        if source_hash != PATCHED_SOURCE_SHA256:
            raise RuntimeError(
                f"{target}: PR #43729 markers are present, but the source does not "
                f"match patched nightly commit {NIGHTLY_BASE_COMMIT}"
            )
        print(f"[vllm-pr43729] {target}: already patched ({PR_COMMIT})")
        py_compile.compile(str(target), doraise=True)
        return
    if any(marker_presence):
        raise RuntimeError(f"{target}: partial PR #43729 patch detected; refusing to continue")
    if source_hash != NIGHTLY_SOURCE_SHA256:
        raise RuntimeError(
            f"{target}: expected vLLM nightly commit {NIGHTLY_BASE_COMMIT} "
            f"(source sha256 {NIGHTLY_SOURCE_SHA256}), found {source_hash}"
        )

    source = replace_once(
        source,
        "class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):\n    def __init__(",
        "class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):\n"
        "    can_return_lse_for_decode: bool = True\n\n"
        "    def __init__(",
        "FlashInferMLAImpl class",
    )
    source = replace_once(
        source,
        "        o = trtllm_batch_decode_with_kv_cache_mla(\n",
        "        return_lse = self.need_to_return_lse_for_decode\n"
        "        kernel_out = trtllm_batch_decode_with_kv_cache_mla(\n",
        "decode call",
    )
    source = replace_once(
        source,
        "            bmm2_scale=self.bmm2_scale,\n        )\n\n        # Flatten the output for consistent shape",
        "            bmm2_scale=self.bmm2_scale,\n"
        "            return_lse=return_lse,\n"
        "        )\n"
        "        if return_lse:\n"
        "            o, lse = kernel_out\n"
        "        else:\n"
        "            o, lse = kernel_out, None\n\n"
        "        # Flatten the output for consistent shape",
        "decode result handling",
    )
    source = replace_once(
        source,
        "        # TODO: Return LSE pending support from Flashinfer API:\n"
        "        # https://github.com/flashinfer-ai/flashinfer/pull/1566\n"
        "        return o, None",
        "        return o, lse",
        "old LSE TODO",
    )

    patched_hash = hashlib.sha256(source.encode()).hexdigest()
    if patched_hash != PATCHED_SOURCE_SHA256:
        raise RuntimeError(f"PR #43729 patch produced unexpected source sha256 {patched_hash}")

    target.write_text(source)
    py_compile.compile(str(target), doraise=True)
    print(f"[vllm-pr43729] patched {target} with {PR_COMMIT}")


if __name__ == "__main__":
    main()
