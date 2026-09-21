"""One-way package boundary around the reciprocal-space workflows.

The reciprocal package owns every q-grid consumer: exact grids, symmetry reduction,
harmonic modes, sampling, SCPH and SSCHA.  Two properties make that ownership real rather
than a directory name:

1. no other production module may import :mod:`mlfcs.reciprocal`, so mainline fitting,
   force constants and IO stay independent of reciprocal space;
2. the reciprocal package may only reach downwards -- structure, force constants, the
   Gaussian sampler and the fitter -- so it cannot become a hidden dependency of a
   higher-level workflow.

The old paths are gone without shims: `mlfcs.phonon.*` and `mlfcs.structure.reciprocal`
neither exist on disk nor resolve at import time, and no `sys.modules` alias recreates
them.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "mlfcs"
RECIPROCAL = SOURCE / "reciprocal"

# Packages the reciprocal package is allowed to reach.  `fitting` is here because SSCHA
# fits its own FC2 from sampled forces with the one fitter implementation the project
# has; duplicating least squares inside the reciprocal package or making the documented
# `SSCHA(primitive, reference=..., cutoff=...)` constructor take an injected fitter would
# both be worse than one explicit edge.  Nothing above the fitter may be imported.
RECIPROCAL_ALLOWED = {"exceptions", "fitting", "force_constants", "sampling", "structure"}
SUBPACKAGES = (
    "mlfcs.reciprocal",
    "mlfcs.reciprocal.grid",
    "mlfcs.reciprocal.statistics",
    "mlfcs.reciprocal.temperature",
    "mlfcs.reciprocal.sampling",
    "mlfcs.reciprocal.scph",
    "mlfcs.reciprocal.sscha",
)


def _imported_mlfcs_packages(source: str) -> set[str]:
    """Return the `mlfcs.<package>` names imported by one source text.

    Deferred imports inside functions count: a lazy import inside a method is still a
    dependency of the module, and hiding it behind `importlib` would only move the
    coupling out of sight.
    """
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if name in {"import_module", "__import__"} and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    modules.append(argument.value)
    return {
        module.split(".", 2)[1]
        for module in modules
        if module.startswith("mlfcs.") and len(module.split(".")) > 1
    }


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py"))


def test_no_production_module_imports_reciprocal():
    """The reciprocal package is a leaf of the production dependency graph."""
    offenders = []
    for path in _python_files(SOURCE):
        if path.is_relative_to(RECIPROCAL):
            continue
        if "reciprocal" in _imported_mlfcs_packages(path.read_text()):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"mainline modules import mlfcs.reciprocal: {offenders}"


def test_reciprocal_only_reaches_downwards():
    """The reciprocal package may not import workflow or writer packages."""
    offenders = {}
    for path in _python_files(RECIPROCAL):
        imported = _imported_mlfcs_packages(path.read_text()) - {"reciprocal"}
        unexpected = imported - RECIPROCAL_ALLOWED
        if unexpected:
            offenders[path.relative_to(ROOT).as_posix()] = sorted(unexpected)
    assert not offenders, f"reciprocal imports higher-level packages: {offenders}"


def test_base_package_import_closures_exclude_reciprocal():
    """Structure, force constants, interactions, IO and fitting never reach reciprocal."""
    checked = (
        "mlfcs.structure",
        "mlfcs.force_constants",
        "mlfcs.interactions",
        "mlfcs.constraints",
        "mlfcs.finite_difference",
        "mlfcs.fitting",
        "mlfcs.io",
        "mlfcs.sampling",
        "mlfcs.calculators",
    )
    for package in checked:
        root = SOURCE / Path(*package.split(".")[1:])
        assert root.exists(), package
        closure = set()
        for path in _python_files(root):
            closure |= {
                name
                for name in _imported_mlfcs_packages(path.read_text())
                if name not in {"exceptions"}
            }
        assert "reciprocal" not in closure, f"{package} imports mlfcs.reciprocal"


def test_legacy_reciprocal_paths_are_gone_without_shims():
    """No old module survives as a file, a package or a `sys.modules` alias."""
    assert not (SOURCE / "phonon").exists()
    assert not (SOURCE / "structure" / "reciprocal.py").exists()
    assert importlib.util.find_spec("mlfcs.phonon") is None
    assert importlib.util.find_spec("mlfcs.structure.reciprocal") is None
    import mlfcs  # noqa: F401  (import for the side effect of populating sys.modules)

    aliases = [name for name in sys.modules if name.startswith(("mlfcs.phonon", "mlfcs.structure.reciprocal"))]
    assert not aliases, f"legacy paths are aliased in sys.modules: {aliases}"


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_public_subpackages_import_standalone(module: str):
    """Every public subpackage imports on its own, without another one priming it."""
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
