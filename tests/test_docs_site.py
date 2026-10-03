# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MkDocs nav and the llms.txt hook, checked without installing MkDocs (CI also runs `mkdocs build --strict`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"


class _MkdocsLoader(yaml.SafeLoader):
    """Safe loading that leaves mkdocs.yml's ``!!python/name:`` tags (the mermaid fence formatter) unresolved."""


_MkdocsLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: suffix)
MKDOCS = yaml.load((REPO_ROOT / "mkdocs.yml").read_text(), Loader=_MkdocsLoader)

_spec = importlib.util.spec_from_file_location("mkdocs_llms_txt", REPO_ROOT / "tools" / "mkdocs_llms_txt.py")
assert _spec and _spec.loader
llms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llms)


def _nav_pages(items) -> list[str]:
    out = []
    for item in items:
        value = item if isinstance(item, str) else next(iter(item.values()))
        out += [value] if isinstance(value, str) else _nav_pages(value)
    return out


def _excluded(rel: Path) -> bool:
    patterns = [p.strip() for p in MKDOCS["exclude_docs"].splitlines() if p.strip()]
    return any(str(rel) == p or (p.endswith("/") and str(rel).startswith(p)) for p in patterns)


def test_every_published_page_is_in_the_nav_once() -> None:
    nav = _nav_pages(MKDOCS["nav"])
    assert len(nav) == len(set(nav)), "a page is listed twice in mkdocs.yml nav"
    published = {str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md") if not _excluded(p.relative_to(DOCS))}
    assert published - set(nav) == set(), "add these pages to the mkdocs.yml nav (or exclude_docs)"
    assert set(nav) - published == set(), "nav lists pages that do not exist or are excluded"


def test_llms_txt_lists_every_nav_page_with_a_summary() -> None:
    site_url = MKDOCS["site_url"]
    text = llms.render_llms_txt(MKDOCS["nav"], DOCS, site_url, MKDOCS["site_name"])
    for src in _nav_pages(MKDOCS["nav"]):
        url = llms.page_url(site_url, src)
        line = next((ln for ln in text.splitlines() if f"]({url})" in ln), None)
        assert line is not None, src
        assert ": " in line.split(f"]({url})", 1)[1], f"{src} has no summary paragraph"
    assert f"{site_url}schema/recipe.schema.json" in text
    assert "design/" not in text and "AGENTS" not in text


def test_page_summary_skips_lists_code_and_headings() -> None:
    md = "# Title\n\n- [toc](#a)\n\n```yaml\nx: 1\n```\n\nFirst **real** [paragraph](x.md).\nStill it.\n\nNot this.\n"
    assert llms.page_summary(md) == "First real paragraph. Still it."
    assert llms.page_url("https://h/s/", "README.md") == "https://h/s/"
    assert llms.page_url("https://h/s/", "dsight.md") == "https://h/s/dsight/"
