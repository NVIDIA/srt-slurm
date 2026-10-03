# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hook: write ``llms.txt`` into the built site.

One line per page in the ``nav`` (title, URL, first paragraph), grouped by nav section, plus the
machine-readable JSON Schemas. It is rebuilt from ``mkdocs.yml`` and the pages on every build, so it
cannot fall out of sync with the site. Stdlib only: the docs workflow installs nothing but mkdocs-material.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SUMMARY_LIMIT = 240
_SKIP_PREFIXES = ("#", "-", "*", "|", "<", "!", ">", "[TOC", "---", "===")


def page_summary(markdown: str, limit: int = SUMMARY_LIMIT) -> str:
    """The first prose paragraph of a page, links reduced to their text, cut at a sentence end."""
    paragraph: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        text = line.strip()
        if text.startswith(("```", "~~~")):
            in_code = not in_code
            if paragraph:
                break
            continue
        if in_code:
            continue
        if not text or text.startswith(_SKIP_PREFIXES) or re.match(r"\d+\.\s", text):
            if paragraph:
                break
            continue
        paragraph.append(text)
    summary = " ".join(paragraph)
    summary = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", summary)
    summary = re.sub(r"\*\*([^*]+)\*\*", r"\1", summary)
    if len(summary) > limit:
        cut = summary.rfind(". ", 0, limit)
        summary = summary[: cut + 1] if cut > limit // 3 else summary[:limit].rsplit(" ", 1)[0] + " ..."
    return summary


def page_url(site_url: str, src: str) -> str:
    """URL of a page under MkDocs' default ``use_directory_urls``."""
    base = site_url.rstrip("/") + "/"
    if src == "README.md" or src.endswith("/README.md") or src == "index.md":
        return base + src[: -len(Path(src).name)]
    return base + src[: -len(".md")] + "/"


def _entries(items: list[Any], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str):
            out.append((prefix.rstrip(": ") or item, item))
            continue
        ((label, value),) = item.items()
        if isinstance(value, str):
            out.append((f"{prefix}{label}", value))
        else:
            out.extend(_entries(value, f"{prefix}{label}: "))
    return out


def render_llms_txt(nav: list[Any], docs_dir: Path, site_url: str, site_name: str) -> str:
    home = docs_dir / "README.md"
    lines = [f"# {site_name}", ""]
    if home.exists():
        lines += [f"> {page_summary(home.read_text(encoding='utf-8'))}", ""]
    lines += [
        (
            "Field-level facts (keys, types, defaults, allowed values) are generated from the code: start with the "
            "Schema Reference or the JSON Schemas below; the other pages explain behavior and workflows."
        ),
        "",
    ]
    for item in nav:
        ((section, value),) = item.items() if isinstance(item, dict) else ((item, item),)
        lines += [f"## {section}", ""]
        entries = [(section, value)] if isinstance(value, str) else _entries(value)
        for title, src in entries:
            summary = page_summary((docs_dir / src).read_text(encoding="utf-8"))
            lines.append(f"- [{title}]({page_url(site_url, src)})" + (f": {summary}" if summary else ""))
        lines.append("")
    schemas = site_url.rstrip("/") + "/schema/"
    lines += [
        "## Machine-readable",
        "",
        (
            f"- [Recipe JSON Schema]({schemas}recipe.schema.json): "
            "draft 2020-12 schema for recipes and override files (`srtctl schema`); use with "
            "`# yaml-language-server: $schema=<this URL>`."
        ),
        (
            f"- [Cluster config JSON Schema]({schemas}cluster.schema.json): "
            "schema for `srtslurm.yaml` (`srtctl schema --cluster`)."
        ),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def on_post_build(config: Any, **kwargs: Any) -> None:
    text = render_llms_txt(config["nav"], Path(config["docs_dir"]), config["site_url"] or "/", config["site_name"])
    Path(config["site_dir"], "llms.txt").write_text(text, encoding="utf-8")
