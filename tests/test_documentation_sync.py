from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("sync_readmes", ROOT / "scripts" / "sync_readmes.py")
assert SPEC is not None and SPEC.loader is not None
SYNC_READMES = module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC_READMES)
SyncError = SYNC_READMES.SyncError
_rewrite_target = SYNC_READMES._rewrite_target
rendered_source = SYNC_READMES.rendered_source


def test_readme_sources_render_without_front_matter_or_h1():
    for source in (Path("docs/en/index.md"), Path("docs/zh/首页.md")):
        rendered = rendered_source(source)
        assert not rendered.startswith("---")
        assert not rendered.startswith("# MLFCS")
        expected_theory = "docs/en/theory/index.md" if source.parts[1] == "en" else "docs/zh/理论/理论.md"
        assert expected_theory in rendered


def test_external_and_anchor_links_are_unchanged():
    source = Path("docs/en/首页.md")
    assert _rewrite_target("https://example.com/a", source) == "https://example.com/a"
    assert _rewrite_target("#section", source) == "#section"


def test_link_outside_repository_is_rejected():
    with pytest.raises(SyncError, match="escapes repository root"):
        _rewrite_target("../../../../outside.md", Path("docs/en/首页.md"))


def test_readmes_are_synchronized():
    completed = subprocess.run(
        [sys.executable, "scripts/sync_readmes.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
