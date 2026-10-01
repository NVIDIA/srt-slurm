# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural checks for the Design Rules in CLAUDE.md.

Each rule walks the AST of ``src/srtctl``. Code that predates a rule is listed in
that rule's baseline; the baseline may only shrink. A new violation fails with the
rule and the fix, and a baseline entry that no longer matches any code fails so
the entry gets deleted.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "srtctl"
SKIP_DIRS = ("benchmarks/scripts/",)

# (path relative to src/srtctl, short key describing the site)
Site = tuple[str, str]


def _python_files() -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(SKIP_DIRS):
            continue
        yield rel, ast.parse(path.read_text(), filename=str(path))


def _dotted(node: ast.expr) -> str | None:
    """``self.config.frontend.type`` for an attribute chain of names, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_backend_or_frontend(node: ast.expr) -> str | None:
    dotted = _dotted(node)
    if dotted is None:
        return None
    last = dotted.rsplit(".", 1)[-1]
    return last if last in ("backend", "frontend") else None


def reflective_access(rel: str, tree: ast.Module) -> Iterator[Site]:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in ("getattr", "hasattr", "setattr") or len(node.args) < 2:
            continue
        owner = _is_backend_or_frontend(node.args[0])
        name = node.args[1]
        if owner and isinstance(name, ast.Constant) and isinstance(name.value, str):
            yield rel, f"{node.func.id}({owner}, {name.value!r})"


def _frontend_type_expr(node: ast.expr) -> bool:
    dotted = _dotted(node)
    return dotted is not None and (dotted.endswith("frontend.type") or dotted.rsplit(".", 1)[-1] == "frontend_type")


def frontend_name_branches(rel: str, tree: ast.Module) -> Iterator[Site]:
    if rel.startswith("frontends/"):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(_frontend_type_expr(o) for o in operands):
            continue
        for o in operands:
            if isinstance(o, ast.Constant) and isinstance(o.value, str) and o.value != "none":
                yield rel, f"frontend.type vs {o.value!r}"


_PORT_NAME = re.compile(r"(^|_)(port|PORT)(_BASE)?$")


def port_arithmetic(rel: str, tree: ast.Module) -> Iterator[Site]:
    if rel in ("ports.py", "core/topology.py"):
        return
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub)):
            continue
        for operand in (node.left, node.right):
            dotted = _dotted(operand)
            if dotted is not None and _PORT_NAME.search(dotted.rsplit(".", 1)[-1]):
                yield rel, ast.unparse(node)


@dataclass(frozen=True)
class Rule:
    name: str
    check: Callable[[str, ast.Module], Iterator[Site]]
    remedy: str
    baseline: frozenset[Site]


RULES = [
    Rule(
        "Backends answer through BackendProtocol, frontends through FrontendConfig/FrontendProtocol",
        reflective_access,
        "Read the typed field or protocol member directly (backend.failover, frontend.numa_bind). "
        "If a backend lacks the feature, add the member to BackendProtocol with a neutral default on "
        "every backend; if a frontend lacks a hook, add it to FrontendProtocol.",
        # Frontend field reads kept reflective because stage tests pass partial SimpleNamespace frontends.
        frozenset(
            {
                ("frontends/base.py", "getattr(frontend, 'numa_bind')"),
                ("frontends/dynamo.py", "getattr(frontend, 'worker_selection')"),
                ("frontends/static_router.py", "getattr(frontend, 'container_image')"),
                ("services/implicit.py", "getattr(frontend, 'type')"),
            }
        ),
    ),
    Rule(
        "Names go in tables, never in branches",
        frontend_name_branches,
        "Put the behavior on the frontend (a FrontendProtocol attribute or hook) and read it through "
        "get_frontend(config.frontend.type) instead of comparing the name outside src/srtctl/frontends/.",
        frozenset(
            {
                ("backends/atom.py", "frontend.type vs 'atomesh'"),
                ("benchmarks/router.py", "frontend.type vs 'sglang-router'"),
                ("cli/mixins/benchmark_stage.py", "frontend.type vs 'sglang-router'"),
                ("cli/mixins/frontend_stage.py", "frontend.type vs 'dynamo'"),
                ("cli/submit.py", "frontend.type vs 'dynamo'"),
                ("cli/submit.py", "frontend.type vs 'vllm'"),
                ("core/schema.py", "frontend.type vs 'dynamo'"),
                ("core/schema.py", "frontend.type vs 'vllm'"),
                ("core/schema.py", "frontend.type vs 'vllm-router'"),
            }
        ),
    ),
    Rule(
        "Every listener a process opens comes from the allocator",
        port_arithmetic,
        "Allocate the port with NodePortAllocator (a PortKind in ports.py, carried on Process) instead of "
        "deriving it from another port; a derived port collides when two processes share a node.",
        frozenset(
            {
                ("backends/vllm.py", "grpc_port + 1"),
                # KVBM_ZMQ_PORTS allocates a two-port block; the ACK port is the block's second port.
                ("cli/mixins/worker_stage.py", "leader.kvbm_zmq_port + 1"),
            }
        ),
    ),
]


def _violations(rule: Rule) -> set[Site]:
    return {site for rel, tree in _python_files() for site in rule.check(rel, tree)}


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.check.__name__)
def test_no_new_violations(rule: Rule):
    new = sorted(_violations(rule) - rule.baseline)
    assert not new, f"Design rule: {rule.name}.\n{rule.remedy}\nNew violations:\n" + "\n".join(
        f"  src/srtctl/{rel}: {key}" for rel, key in new
    )


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.check.__name__)
def test_baseline_has_no_fixed_entries(rule: Rule):
    fixed = sorted(rule.baseline - _violations(rule))
    assert not fixed, (
        "These sites were fixed; delete them from the baseline in tests/test_design_rules.py:\n"
        + "\n".join(f"  src/srtctl/{rel}: {key}" for rel, key in fixed)
    )
