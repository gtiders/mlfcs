#!/usr/bin/env python3
"""Validate bilingual documentation structure and repository documentation rules."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LEGACY_MATH = re.compile(r"(?<!\\)\\(?:\(|\)|\[|\])")
VALID_STATUS = {"stable", "experimental", "planned", "research", "deprecated"}
MIN_PAGE_CONTENT_CHARACTERS = 220


def _markdown_paths(language: str) -> set[Path]:
    base = DOCS / language
    return {path.relative_to(base) for path in base.rglob("*.md")}


def _chinese_markdown_paths() -> set[Path]:
    base = DOCS / "zh"
    return {path.relative_to(base) for path in base.rglob("*.md")}


def _front_matter(text: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    return None if end < 0 else text[4:end]


def _field(front_matter: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}:\s*(\S+)\s*$", front_matter, re.MULTILINE)
    return None if match is None else match.group(1)


def main() -> int:
    errors: list[str] = []
    english = _markdown_paths("en")
    chinese = _chinese_markdown_paths()
    localized_only = []
    for path in sorted(chinese):
        front_matter = _front_matter((DOCS / "zh" / path).read_text(encoding="utf-8"))
        if front_matter is not None and _field(front_matter, "localized_only") == "true":
            localized_only.append(path)
    if len(chinese) != len(english) + len(localized_only):
        errors.append(
            f"English and Chinese documentation page counts differ after localized-only pages: "
            f"{len(english)} != {len(chinese) - len(localized_only)}"
        )

    if (ROOT / "README_ZH.md").exists():
        errors.append("legacy README_ZH.md must not exist")

    for path in [*DOCS.rglob("*.md"), ROOT / "README.md", ROOT / "README.zh-CN.md"]:
        text = path.read_text(encoding="utf-8")
        if LEGACY_MATH.search(text):
            errors.append(f"legacy Markdown math delimiter in {path.relative_to(ROOT)}")
        if "README_ZH.md" in text:
            errors.append(f"legacy README link in {path.relative_to(ROOT)}")

    for language in ("en", "zh"):
        for path in (DOCS / language).rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            front_matter = _front_matter(text)
            if front_matter is None:
                errors.append(f"missing front matter: {path.relative_to(ROOT)}")
                continue
            status = _field(front_matter, "status")
            if status not in VALID_STATUS:
                errors.append(f"invalid or missing status: {path.relative_to(ROOT)}")
            if not re.search(r"^audience:\s*$", front_matter, re.MULTILINE):
                errors.append(f"missing audience: {path.relative_to(ROOT)}")
            if not re.search(r"^code_verified:\s*\S+\s*$", front_matter, re.MULTILINE):
                errors.append(f"missing code_verified: {path.relative_to(ROOT)}")
            body = text[text.find("\n---\n", 4) + 5 :]
            content_size = len("".join(body.split()))
            if content_size < MIN_PAGE_CONTENT_CHARACTERS:
                errors.append(
                    "documentation page is still a short stub "
                    f"({content_size} < {MIN_PAGE_CONTENT_CHARACTERS} characters): "
                    f"{path.relative_to(ROOT)}"
                )

    for relative in sorted(english):
        chinese_path = DOCS / "zh" / relative
        if not chinese_path.exists():
            continue
        en_front = _front_matter((DOCS / "en" / relative).read_text(encoding="utf-8"))
        zh_front = _front_matter(chinese_path.read_text(encoding="utf-8"))
        if en_front is None or zh_front is None:
            continue
        for field in ("status", "code_verified"):
            if _field(en_front, field) != _field(zh_front, field):
                errors.append(f"bilingual {field} mismatch: {relative}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"documentation mirrors agree: {len(english)} pages per language")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
