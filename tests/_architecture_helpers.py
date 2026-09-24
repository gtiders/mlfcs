"""Static import inspection for the rewritten package layers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "mlfcs"


def internal_dependencies(package: str) -> set[str]:
    dependencies: set[str] = set()
    for path in (ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            if module.startswith("mlfcs."):
                dependency = module.split(".", 2)[1]
                if dependency != package:
                    dependencies.add(dependency)
    return dependencies


__all__ = ["internal_dependencies"]
