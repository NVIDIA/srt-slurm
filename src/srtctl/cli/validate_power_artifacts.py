# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI wrapper for offline validation of retained dcgm-power artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from srtctl.core.power.validate_artifacts import validate_power_artifacts


def _expect_role(item: str) -> tuple[str, int]:
    """Parse and validate one ``ROLE=COUNT`` occurrence."""
    role, sep, count_text = item.partition("=")
    if not sep or not role:
        raise argparse.ArgumentTypeError(f"expected ROLE=COUNT, got {item!r}")
    try:
        count = int(count_text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"COUNT must be an integer in {item!r}") from None
    if count < 0:
        raise argparse.ArgumentTypeError(f"COUNT must be non-negative in {item!r}")
    return role, count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a retained dcgm-power artifact package")
    parser.add_argument("--power-dir", type=Path, required=True, help="directory holding manifest.json and samples.csv")
    parser.add_argument(
        "--result-root", type=Path, required=True, help="root that window result_path values resolve under"
    )
    parser.add_argument(
        "--expect-role",
        action="append",
        default=[],
        type=_expect_role,
        metavar="ROLE=COUNT",
        help="require exactly COUNT GPUs mapped to ROLE (repeatable)",
    )
    parser.add_argument(
        "--require-distinct-het-groups",
        action="store_true",
        help="require each worker role to occupy its own heterogeneous Slurm group",
    )
    args = parser.parse_args(argv)

    expected_roles: dict[str, int] | None = None
    if args.expect_role:
        expected_roles = {}
        for role, count in args.expect_role:
            if role in expected_roles:
                parser.error(f"duplicate --expect-role for {role!r}")
            expected_roles[role] = count

    report = validate_power_artifacts(
        power_dir=args.power_dir,
        result_root=args.result_root,
        expected_roles=expected_roles,
        require_distinct_het_groups=args.require_distinct_het_groups,
    )
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
