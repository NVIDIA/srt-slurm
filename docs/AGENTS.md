# Documentation guidance

Documentation in this folder describes the current implementation, including
behavior implemented by the same PR. Keep diagrams, examples, file formats and
UI names consistent with the code.

Do not include proposed features, roadmaps or future behavior. Keep proposals in
issues or PR discussions. Document current limitations and missing-data behavior
as they exist.

## Diagrams

Draw every diagram as a fenced `mermaid` block (`flowchart` for structure, `sequenceDiagram` for request and lifecycle flows); MkDocs renders them. Do not hand-draw boxes and arrows in a plain code block: `tests/test_docs_diagrams.py` fails on ASCII boxes (`+----+`), arrow lines (`|` / `v`, `-->`, `──▶`) and joined box-drawing boxes outside a `mermaid` block. Directory trees and captured terminal output are not diagrams and are fine as `text` blocks.
