#!/usr/bin/env python3
"""Check all documentation pages, tutorial notebooks, local links, and site assets."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
NOTEBOOKS = DOCS / "notebooks"
PAGES = {
    Path("index.md"),
    Path("core-concepts.md"),
    Path("finite-difference-api.md"),
    Path("fitting-api.md"),
    Path("ewald-api.md"),
    Path("harmonic-api.md"),
    Path("scph.md"),
    Path("Q&A.md"),
    Path("notebooks/index.md"),
    Path("development/numerical-contracts.md"),
    Path("development/long-range-dynamics.md"),
    Path("theory/index.md"),
    Path("theory/notation.md"),
    Path("theory/glossary.md"),
    Path("theory/limitations.md"),
    Path("theory/periodic-force-constants.md"),
    Path("theory/symmetry-parameterization.md"),
    Path("theory/invariance-constraints.md"),
    Path("theory/supercells-identifiability.md"),
    Path("theory/reconstruction.md"),
    Path("theory/harmonic-dynamics.md"),
    Path("theory/reciprocal-symmetry.md"),
    Path("theory/long-range-electrostatics.md"),
    Path("theory/advanced/exact-linear-algebra.md"),
    Path("theory/advanced/self-consistent-phonons.md"),
}
TUTORIAL_NOTEBOOKS = {
    "rotational-sum-rules.ipynb",
    "si-finite-difference.ipynb",
    "nacl-long-range.ipynb",
    "ba8ga16ge30.ipynb",
    "si-fitting.ipynb",
    "k4as4pt2.ipynb",
}
LEGACY_MATH = re.compile(r"(?<!\\)\\(?:\(|\)|\[|\])")
LINK = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")
FENCE = re.compile(r"^\s*(?:```|~~~)")


def _content_without_fences(text: str) -> str:
    lines: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return "\n".join(lines)


def _check_local_links(path: Path, text: str) -> list[str]:
    errors = []
    for match in LINK.finditer(_content_without_fences(text)):
        target = match.group(1)
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(ROOT):
            errors.append(f"{path.relative_to(ROOT)}: link escapes repository: {target}")
        elif not resolved.exists():
            errors.append(f"{path.relative_to(ROOT)}: broken local link: {target}")
    return errors


def _check_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if LEGACY_MATH.search(source):
            errors.append(
                f"legacy Markdown math delimiter in "
                f"{path.relative_to(ROOT)} (markdown cell {index})"
            )
        errors.extend(_check_local_links(path, source))
    return errors


def main() -> int:
    errors: list[str] = []
    actual = {path.relative_to(DOCS) for path in DOCS.rglob("*.md")}
    missing = sorted(PAGES - actual)
    errors.extend(f"missing required documentation page: docs/{path}" for path in missing)

    notebook_names = {path.name for path in NOTEBOOKS.glob("*.ipynb")}
    missing_notebooks = sorted(TUTORIAL_NOTEBOOKS - notebook_names)
    errors.extend(
        f"missing tutorial notebook: docs/notebooks/{name}" for name in missing_notebooks
    )

    if not (DOCS / "assets/images/logo.png").is_file():
        errors.append("missing docs/assets/images/logo.png")
    if not (DOCS / "assets/stylesheets/extra.css").is_file():
        errors.append("missing docs/assets/stylesheets/extra.css")
    if (ROOT / "README.zh-CN.md").exists():
        errors.append("legacy README.zh-CN.md must not exist; the README is Chinese-only")

    for relative in sorted(actual):
        path = DOCS / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if LEGACY_MATH.search(text):
            errors.append(f"legacy Markdown math delimiter in {path.relative_to(ROOT)}")
        errors.extend(_check_local_links(path, text))

    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        errors.extend(_check_notebook(path))

    readme = ROOT / "README.md"
    if readme.exists() and LEGACY_MATH.search(readme.read_text(encoding="utf-8")):
        errors.append("legacy Markdown math delimiter in README.md")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"documentation set is valid: {len(actual & PAGES)} pages, "
        f"{len(notebook_names)} notebooks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
