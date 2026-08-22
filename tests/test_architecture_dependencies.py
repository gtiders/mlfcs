"""Lock the package dependency graph after the responsibility refactor."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "mlfcs"

ALLOWED = {
    "structure": set(),
    "interactions": {"structure"},
    "basis": {"interactions"},
    "force_constants": {"interactions", "structure"},
    "constraints": {"force_constants", "interactions", "structure"},
    "finite_difference": {"basis", "constraints", "force_constants", "interactions", "structure"},
    "fitting": {"basis", "constraints", "force_constants", "interactions", "structure"},
    "physics": {"constraints", "fitting", "force_constants", "interactions", "structure"},
    "io": {"force_constants", "structure"},
}


def _internal_dependencies(package: str) -> set[str]:
    dependencies: set[str] = set()
    for path in (ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
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


def test_package_dependencies_follow_the_locked_dag():
    for package, allowed in ALLOWED.items():
        unexpected = _internal_dependencies(package) - allowed
        assert not unexpected, f"{package} has forbidden dependencies: {sorted(unexpected)}"


def test_historical_ambiguous_packages_are_removed():
    for package in ("core", "ifc", "anharmonic", "public"):
        assert not (ROOT / package).exists()
