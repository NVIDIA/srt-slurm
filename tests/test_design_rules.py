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


def _subscriber_destination_arithmetic(tree: ast.Module) -> set[ast.BinOp]:
    """Recognize outbound KV-event URLs through the Frontend subscriber contract.

    ``kv_events_subscriber`` returns a remote (host, port, topic), not a listener
    allocation. A publisher may undo its engine's rank offset in that destination.
    Follow single-assignment locals within one function, and exempt only the port
    expression in an ``endpoint`` URL using that same remote host. Reusing the
    port for a listener, or changing either local, must still trip the rule.
    """
    destinations: set[ast.BinOp] = set()
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        # Nested scopes have independent bindings and are considered separately.
        nodes: list[ast.AST] = []
        pending: list[ast.AST] = list(scope.body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                continue
            nodes.append(node)
            pending.extend(ast.iter_child_nodes(node))
        stores: dict[str, int] = {}
        assignments: dict[str, ast.expr] = {}
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stores[node.id] = stores.get(node.id, 0) + 1
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assignments[node.targets[0].id] = node.value
        addresses: set[tuple[str, str]] = set()
        for node in nodes:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Tuple) and len(target.elts) == 3):
                continue
            host, port, _ = target.elts
            if not (isinstance(host, ast.Name) and isinstance(port, ast.Name)):
                continue
            if stores.get(host.id) != 1 or stores.get(port.id) != 1:
                continue
            value = node.value
            if isinstance(value, ast.Name) and stores.get(value.id) == 1:
                value = assignments.get(value.id, value)
            if isinstance(value, ast.IfExp) and isinstance(value.orelse, ast.Constant) and value.orelse.value is None:
                value = value.body
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "kv_events_subscriber"
            ):
                addresses.add((host.id, port.id))
        for node in nodes:
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if not (isinstance(key, ast.Constant) and key.value == "endpoint" and isinstance(value, ast.JoinedStr)):
                    continue
                match value.values:
                    case [
                        ast.Constant(value="tcp://"),
                        ast.FormattedValue(value=ast.Name(id=host)),
                        ast.Constant(value=":"),
                        ast.FormattedValue(value=ast.BinOp(left=ast.Name(id=port)) as arithmetic),
                    ] if (host, port) in addresses:
                        destinations.add(arithmetic)
    return destinations


def port_arithmetic(rel: str, tree: ast.Module) -> Iterator[Site]:
    if rel in ("ports.py", "core/topology.py"):
        return
    destinations = _subscriber_destination_arithmetic(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub)):
            continue
        if node in destinations:
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
        "Backends answer through Backend, frontends through FrontendConfig/Frontend",
        reflective_access,
        "Read the typed field or protocol member directly (backend.failover, frontend.numa_bind). "
        "If a backend lacks the feature, add the member to Backend with a neutral default on "
        "every backend; if a frontend lacks a hook, add it to Frontend.",
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
        "Put the behavior on the frontend (a Frontend attribute or hook) and read it through "
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
