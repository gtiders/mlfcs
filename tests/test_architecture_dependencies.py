"""Lock the package dependency graph after the responsibility refactor."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "mlfcs"

ALLOWED = {
    "structure": set(),
    "interactions": {"exceptions", "structure"},
    "force_constants": {"interactions", "structure"},
    "constraints": {"force_constants", "interactions", "structure"},
    "finite_difference": {"constraints", "force_constants", "interactions", "structure"},
    "fitting": {"constraints", "force_constants", "interactions", "structure"},
    "sampling": {"force_constants", "structure"},
    "physics": {
        "constraints",
        "fitting",
        "force_constants",
        "interactions",
        "sampling",
        "structure",
    },
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
    for package in ("core", "ifc", "anharmonic", "basis", "public"):
        assert not (ROOT / package).exists()


def test_concrete_fitting_backends_do_not_leak_into_generic_modules():
    backend_root = ROOT / "fitting" / "backends"
    for path in ROOT.rglob("*.py"):
        if path.is_relative_to(backend_root):
            continue
        source = path.read_text()
        assert "mlfcs.fitting.backends.wick" not in source, path
        assert "mlfcs.fitting.backends.taylor" not in source, path


def test_production_code_has_no_print_or_legacy_diagnostics_types():
    for path in ROOT.rglob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        ), path
        assert "Diagnostics" not in source, path
        assert "reporter" not in source, path
        assert "verbose" not in source, path
        assert "log_level" not in source, path
