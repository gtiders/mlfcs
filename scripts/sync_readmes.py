#!/usr/bin/env python3
"""Synchronize the GitHub README generated regions from documentation home pages."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    Path("docs/en/index.md"): Path("README.md"),
    Path("docs/zh/index.md"): Path("README.zh-CN.md"),
}
H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)
LINK_RE = re.compile(r"(!?\[[^\]]*\]\()(<[^>]+>|[^)\s]+)([^)]*)(\))")


class SyncError(RuntimeError):
    """Raised when a README cannot be synchronized safely."""


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SyncError("unterminated YAML front matter")
    return text[end + 5 :]


def _without_h1(text: str, source: Path) -> str:
    matches = list(H1_RE.finditer(text))
    if len(matches) != 1:
        raise SyncError(f"{source}: expected exactly one level-one heading, found {len(matches)}")
    match = matches[0]
    return (text[: match.start()] + text[match.end() :]).strip()


def _rewrite_target(raw_target: str, source: Path) -> str:
    wrapped = raw_target.startswith("<") and raw_target.endswith(">")
    target = raw_target[1:-1] if wrapped else raw_target
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or target.startswith("#"):
        return raw_target

    resolved = (ROOT / source.parent / parts.path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SyncError(f"{source}: link escapes repository root: {target}") from exc

    rewritten = urlunsplit(("", "", relative.as_posix(), parts.query, parts.fragment))
    return f"<{rewritten}>" if wrapped else rewritten


def _rewrite_links(text: str, source: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        target = _rewrite_target(match.group(2), source)
        return f"{match.group(1)}{target}{match.group(3)}{match.group(4)}"

    return LINK_RE.sub(replace, text)


def rendered_source(source: Path) -> str:
    text = (ROOT / source).read_text(encoding="utf-8").replace("\r\n", "\n")
    text = _strip_front_matter(text)
    text = _without_h1(text, source)
    return _rewrite_links(text, source).strip()


def expected_readme(source: Path, target: Path) -> str:
    path = ROOT / target
    current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    begin = f"<!-- BEGIN GENERATED: {source.as_posix()} -->"
    end = f"<!-- END GENERATED: {source.as_posix()} -->"
    if current.count(begin) != 1 or current.count(end) != 1:
        raise SyncError(f"{target}: expected exactly one generated marker pair for {source}")
    start = current.index(begin) + len(begin)
    stop = current.index(end)
    if start > stop:
        raise SyncError(f"{target}: generated markers are reversed")
    generated = f"\n\n{rendered_source(source)}\n\n"
    return (current[:start] + generated + current[stop:]).rstrip() + "\n"


def synchronize(*, check: bool) -> int:
    failed = False
    for source, target in PAIRS.items():
        expected = expected_readme(source, target)
        path = ROOT / target
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if current == expected:
            continue
        if check:
            failed = True
            print(f"{target} is not synchronized with {source}", file=sys.stderr)
            print(
                "".join(
                    difflib.unified_diff(
                        current.splitlines(keepends=True),
                        expected.splitlines(keepends=True),
                        fromfile=str(target),
                        tofile=f"{target} (expected)",
                    )
                ),
                file=sys.stderr,
            )
        else:
            path.write_text(expected, encoding="utf-8", newline="\n")
            print(f"updated {target} from {source}")
    if failed:
        print("Run: uv run python scripts/sync_readmes.py", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report drift without writing files")
    args = parser.parse_args()
    try:
        return synchronize(check=args.check)
    except SyncError as exc:
        print(f"README synchronization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
