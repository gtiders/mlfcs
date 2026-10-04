"""Dependency-direction guards for the tools package.

Tool correctness is validated by one-off checks at implementation time
(AGENTS.md rule 3) and has no standing tests; the layering rules below are
the only contract the suite keeps.
"""

import ast
from pathlib import Path

from _architecture_helpers import internal_dependencies


def test_tools_modules_have_no_phonopy_runtime_or_reverse_dependency() -> None:
    root = Path(__file__).parents[1]
    sources = sorted(
        path for path in (root / "src/mlfcs/tools").glob("*.py") if path.name != "__init__.py"
    )
    assert sources, "the tools package is expected to contain modules"
    for source in sources:
        tree = ast.parse(source.read_text())
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )

        assert not any(name == "phonopy" or name.startswith("phonopy.") for name in imports), (
            f"{source.name} must not import phonopy"
        )
        assert not any(name.startswith("mlfcs.") for name in imports), (
            f"{source.name} must not import mlfcs packages"
        )
    assert internal_dependencies("tools") == set()
    packages = sorted(
        path.name
        for path in (root / "src/mlfcs").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file() and path.name != "tools"
    )
    for package in packages:
        assert "tools" not in internal_dependencies(package), (
            f"production package {package} must not import tools"
        )
