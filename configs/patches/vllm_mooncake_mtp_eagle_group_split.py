#!/usr/bin/env python3
"""Split mixed eagle/non-eagle Mooncake attention groups.

DeepSeek V4 + MTP marks the physical KV group containing the MTP layer as an
eagle group. In c188b96, Mooncake's coordinator merges physical groups solely
by KVCacheSpec. That coalesces the eagle SWA group with three non-eagle SWA
groups, even though their reachable-block proofs differ by one peek block.
The merged pool requires every physical group to contain every candidate hash,
so the non-eagle groups remove the eagle peek block before eagle verification
can consume it. The resulting cross-attention-group intersection is always
empty and the Mooncake master reports Get=0.

Keep all upstream eagle semantics, but make eagle membership part of the
coordinator's grouping key. This produces separate same-spec eagle and
non-eagle attention groups whose proofs can converge correctly.

Runs inside the worker container after vLLM is installed (post_install_script).
It is deliberately anchored to the c188b96 source shape and fails closed when
that shape changes.

Transition only: vllm-internal PRs #275 (e5827fabc1) and #287 (89510b05dc)
supersede this workaround. Do not upstream this group-split patch. Remove it
when the runtime image contains both official fixes (89510b05dc or a
descendant).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

MARKER = "SRT patch(mtp-mooncake-eagle-group-split)"

MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator"

OLD_GROUPING = """            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls
                    group_ids.append(i)
                    break"""

NEW_GROUPING = """            # SRT patch(mtp-mooncake-eagle-group-split): groups with
            # identical specs can still require different reachable-block
            # proofs. Do not merge an eagle physical group with non-eagle
            # groups, or the shared block-pool intersection drops its peek.
            is_eagle_group = i in self.eagle_group_ids
            for existing_spec, group_ids, existing_cls in attention_groups:
                existing_is_eagle_group = group_ids[0] in self.eagle_group_ids
                if (
                    existing_spec == spec
                    and existing_is_eagle_group == is_eagle_group
                ):
                    assert manager_cls is existing_cls
                    group_ids.append(i)
                    break"""


def patch_text(text: str, source: str) -> str:
    """Return patched coordinator source, failing on an unexpected shape."""
    if MARKER in text:
        return text

    anchor_count = text.count(OLD_GROUPING)
    if anchor_count != 1:
        raise RuntimeError(
            f"expected exactly one Mooncake grouping anchor in {source}, "
            f"found {anchor_count}"
        )

    patched = text.replace(OLD_GROUPING, NEW_GROUPING, 1)
    compile(patched, source, "exec")

    required = (
        MARKER,
        "is_eagle_group = i in self.eagle_group_ids",
        "existing_is_eagle_group = group_ids[0] in self.eagle_group_ids",
        "existing_is_eagle_group == is_eagle_group",
    )
    missing = [needle for needle in required if needle not in patched]
    if missing:
        raise RuntimeError(f"patched source validation failed: missing {missing}")
    return patched


def resolve_installed_path() -> Path:
    module = importlib.import_module(MODULE)
    if module.__file__ is None:
        raise RuntimeError(f"module {MODULE} has no __file__")
    return Path(module.__file__)


def run_self_test() -> None:
    fixture = (
        "def split(self, attention_groups, spec, manager_cls, i):\n"
        "    if True:\n"
        "        if True:\n"
        f"{OLD_GROUPING}\n"
    )
    patched = patch_text(fixture, "<self-test>")
    assert patched.count(MARKER) == 1
    assert patched == patch_text(patched, "<self-test-idempotent>")
    print("Mooncake eagle-group split patch self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        help="coordinator.py to validate/patch instead of importing installed vLLM",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate transformed syntax without writing the target",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    path = args.path if args.path is not None else resolve_installed_path()
    text = path.read_text()
    patched = patch_text(text, str(path))

    if text == patched:
        print(f"Mooncake eagle-group split patch already applied -> {path}")
        return
    if args.dry_run:
        print(f"Mooncake eagle-group split patch dry-run passed -> {path}")
        return

    path.write_text(patched)
    if path.read_text() != patched:
        raise RuntimeError(f"failed to verify patched contents in {path}")
    print(f"Patched Mooncake coordinator eagle/non-eagle grouping -> {path}")


if __name__ == "__main__":
    main()
