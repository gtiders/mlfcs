#!/usr/bin/env python3
"""Check the deliberately small public documentation set and site assets."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = {
    Path("index.md"),
    Path("en/index.md"),
    Path("en/Q&A.md"),
    Path("en/api/finite-difference-api.md"),
    Path("en/api/fitting-api.md"),
    Path("zh/index.md"),
    Path("zh/Q&A.md"),
    Path("zh/api/finite-difference-api.md"),
    Path("zh/api/fitting-api.md"),
}
LEGACY_MATH = re.compile(r"(?<!\\)\\(?:\(|\)|\[|\])")
LINK = re.compile(r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")
FENCE = re.compile(r"^\s*(?:```|~~~)")
MIRRORED = (
    Path("index.md"),
    Path("Q&A.md"),
    Path("api/finite-difference-api.md"),
    Path("api/fitting-api.md"),
)


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


def main() -> int:
    errors: list[str] = []
    actual = {path.relative_to(DOCS) for path in DOCS.rglob("*.md")}
    unexpected = sorted(actual - PAGES)
    missing = sorted(PAGES - actual)
    errors.extend(f"unexpected documentation page: docs/{path}" for path in unexpected)
    errors.extend(f"missing documentation page: docs/{path}" for path in missing)

    if not (DOCS / "assets/images/logo.png").is_file():
        errors.append("missing docs/assets/images/logo.png")
    if not (DOCS / "assets/stylesheets/extra.css").is_file():
        errors.append("missing docs/assets/stylesheets/extra.css")
    if (ROOT / "README_ZH.md").exists():
        errors.append("legacy README_ZH.md must not exist")

    for relative in sorted(PAGES):
        path = DOCS / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if LEGACY_MATH.search(text):
            errors.append(f"legacy Markdown math delimiter in {path.relative_to(ROOT)}")
        errors.extend(_check_local_links(path, text))

    for relative in MIRRORED:
        english = DOCS / "en" / relative
        chinese = DOCS / "zh" / relative
        if not english.is_file() or not chinese.is_file():
            continue
        if relative == Path("index.md"):
            continue
        en_text = _content_without_fences(english.read_text(encoding="utf-8"))
        zh_text = _content_without_fences(chinese.read_text(encoding="utf-8"))
        if not en_text.strip() or not zh_text.strip():
            errors.append(f"empty bilingual page pair: {relative}")

    for path in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
        if path.exists() and LEGACY_MATH.search(path.read_text(encoding="utf-8")):
            errors.append(f"legacy Markdown math delimiter in {path.relative_to(ROOT)}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"documentation set is valid: {len(PAGES)} pages, {len(MIRRORED)} bilingual pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
