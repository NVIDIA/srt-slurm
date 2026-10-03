"""Generated documentation artifacts.

``srtctl schema-docs`` writes every file in :func:`generated_artifacts` and ``srtctl schema-docs --check``
fails when a checked-in copy differs, so the field reference, the published JSON Schemas and the CLI
reference cannot drift from the code they are rendered from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"

CLI_NOTICE = (
    "<!-- GENERATED FILE. Do not edit by hand. Rendered from the argparse parser in "
    "src/srtctl/cli/submit.py by `srtctl schema-docs`; CI fails when this file is stale. -->"
)


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _flag(action: argparse.Action) -> str:
    if not action.option_strings:
        return f"`{action.metavar or action.dest}`"
    names = ", ".join(action.option_strings)
    if action.nargs == 0:
        return f"`{names}`"
    metavar = action.metavar or action.dest.upper()
    if isinstance(metavar, tuple):
        metavar = " ".join(metavar)
    return f"`{names} {metavar}`"


def _help(action: argparse.Action) -> str:
    text = action.help or ""
    if "%(" in text:
        params = {k: v for k, v in vars(action).items() if v is not argparse.SUPPRESS}
        text = text % params
    if action.choices is not None and not isinstance(action.choices, dict):
        text = f"{text} One of: {', '.join(f'`{c}`' for c in action.choices)}.".strip()
    return text


def _default(action: argparse.Action) -> str:
    if action.required:
        return "required"
    value = action.default
    if value is None or value is False or value == [] or value is argparse.SUPPRESS or action.nargs == 0:
        return ""
    return f"`{value}`"


def _subparsers(parser: argparse.ArgumentParser) -> list[tuple[str, list[str], str, argparse.ArgumentParser]]:
    """(name, aliases, help, parser) for each subcommand, in registration order."""
    out: list[tuple[str, list[str], str, argparse.ArgumentParser]] = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        helps = {choice.dest: choice.help or "" for choice in action._choices_actions}
        by_parser: dict[int, tuple[str, list[str], str, argparse.ArgumentParser]] = {}
        for name, sub in action.choices.items():
            if id(sub) in by_parser:
                by_parser[id(sub)][1].append(name)
                continue
            entry = (name, [], helps.get(name, ""), sub)
            by_parser[id(sub)] = entry
            out.append(entry)
    return out


def _options_table(parser: argparse.ArgumentParser) -> list[str]:
    rows = [
        action
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction | argparse._SubParsersAction | argparse._VersionAction)
        and action.help is not argparse.SUPPRESS
    ]
    if not rows:
        return []
    lines = ["| Argument | Default | Description |", "| --- | --- | --- |"]
    lines += [f"| {_flag(a)} | {_default(a)} | {_cell(_help(a))} |" for a in rows]
    return lines + [""]


def _render_command(path: str, aliases: list[str], help_text: str, parser: argparse.ArgumentParser, level: int):
    lines = [f"{'#' * level} `{path}`", ""]
    summary = parser.description or help_text
    if summary:
        lines += [_cell(summary), ""]
    if aliases:
        lines += ["Aliases: " + ", ".join(f"`{a}`" for a in aliases), ""]
    lines += _options_table(parser)
    for name, sub_aliases, sub_help, sub in _subparsers(parser):
        lines += _render_command(f"{path} {name}", sub_aliases, sub_help, sub, min(level + 1, 4))
    return lines


def render_cli_reference(parser: argparse.ArgumentParser | None = None) -> str:
    """Every `srtctl` subcommand and argument, rendered from the argparse definitions."""
    if parser is None:
        from srtctl.cli.submit import build_parser

        parser = build_parser()
    lines = [
        "# CLI Reference",
        "",
        CLI_NOTICE,
        "",
        (
            "Every `srtctl` subcommand and argument, generated from the parser itself. "
            "Workflows, examples, and what each command does are in the [CLI Guide](cli.md). "
            "Running `srtctl` with no arguments starts the interactive mode."
        ),
        "",
        "```text",
        (parser.epilog or "").rstrip(),
        "```",
        "",
    ]
    for name, aliases, help_text, sub in _subparsers(parser):
        lines += _render_command(f"srtctl {name}", aliases, help_text, sub, level=2)
    return "\n".join(lines).rstrip() + "\n"


def _json(data: object) -> str:
    return json.dumps(data, indent=2) + "\n"


def generated_artifacts(docs_dir: Path = DOCS_DIR) -> dict[Path, str]:
    """Path -> rendered content for every generated file under ``docs_dir``."""
    from srtctl.core.schema import ClusterConfig, SrtConfig
    from srtctl.core.schema_docs import json_schema, render_schema_reference

    return {
        docs_dir / "schema-reference.md": render_schema_reference(),
        docs_dir / "schema" / "recipe.schema.json": _json(json_schema(SrtConfig)),
        docs_dir / "schema" / "cluster.schema.json": _json(json_schema(ClusterConfig)),
        docs_dir / "cli-reference.md": render_cli_reference(),
    }


def write_generated(docs_dir: Path = DOCS_DIR) -> list[Path]:
    written = []
    for path, content in generated_artifacts(docs_dir).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def stale_generated(docs_dir: Path = DOCS_DIR) -> list[Path]:
    """Generated files that are missing or differ from what the code renders."""
    return [
        path
        for path, content in generated_artifacts(docs_dir).items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
