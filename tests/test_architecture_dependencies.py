"""Lock the package dependency graph after the responsibility refactor."""

from __future__ import annotations

import ast
import subprocess
import sys

from _architecture_helpers import ROOT, internal_dependencies

ALLOWED = {
    "structure": set(),
    # Independent Cartesian Gaussian perturbation; it is the only sampling entry point the
    # root namespace keeps, and it is a base layer the reciprocal package may use.
    "sampling": {"structure"},
    "interactions": {"exceptions", "structure"},
    "force_constants": {"interactions", "structure"},
    "constraints": {"force_constants", "interactions", "structure"},
    "finite_difference": {"constraints", "force_constants", "interactions", "structure"},
    "fitting": {"calculators", "constraints", "force_constants", "interactions", "structure"},
    # ``calculators`` is a leaf over force constants, io and structure; the fitter imports
    # the Taylor calculator *inside* the error-reporting method, which is why the layer
    # below can be named here without creating an import cycle.
    "calculators": {"force_constants", "io", "structure"},
    # The reciprocal package owns every q-grid consumer.  It reaches downwards only:
    # structure, force constants, the Gaussian sampler and the fitter that SSCHA uses to
    # refit FC2 from sampled forces.
    "reciprocal": {"exceptions", "fitting", "force_constants", "sampling", "structure"},
    "io": {"force_constants", "structure"},
}


def test_package_dependencies_follow_the_locked_dag():
    for package, allowed in ALLOWED.items():
        unexpected = internal_dependencies(package) - allowed
        assert not unexpected, f"{package} has forbidden dependencies: {sorted(unexpected)}"


def test_mainline_packages_do_not_depend_on_reciprocal_workflows():
    for package in (
        "structure",
        "interactions",
        "force_constants",
        "constraints",
        "finite_difference",
        "fitting",
        "io",
        "sampling",
        "calculators",
    ):
        assert "reciprocal" not in internal_dependencies(package), package


def test_no_module_outside_the_reciprocal_package_imports_it():
    """A file-level scan, because the root namespace is not a package of its own.

    ``internal_dependencies`` walks one package at a time, so a stray import in
    ``src/mlfcs/__init__.py`` or in any other module outside the reciprocal tree would slip
    through.  The plan makes this boundary the merge gate, so it is checked directly.
    """
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if "reciprocal" in path.relative_to(ROOT).parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            if any(
                name == "mlfcs.reciprocal" or name.startswith("mlfcs.reciprocal.")
                for name in imported
                if name
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


def test_every_public_subpackage_imports_on_its_own():
    """A fresh interpreter per subpackage catches import cycles and missing optionals.

    Importing the package in a subprocess is the only way to see a cycle: an in-process
    import would already be satisfied by whatever the test session imported earlier.
    """
    modules = (
        "mlfcs.reciprocal",
        "mlfcs.reciprocal.grid",
        "mlfcs.reciprocal.fourier",
        "mlfcs.reciprocal.symmetry",
        "mlfcs.reciprocal.sampling",
        "mlfcs.reciprocal.scph",
        "mlfcs.reciprocal.sscha",
        "mlfcs.sampling",
        "mlfcs.structure",
        "mlfcs.interactions",
        "mlfcs.force_constants",
        "mlfcs.constraints",
        "mlfcs.finite_difference",
        "mlfcs.fitting",
        "mlfcs.io",
        "mlfcs.calculators",
    )
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{module}: {completed.stderr.strip()}"


def test_no_mainline_import_closure_reaches_the_reciprocal_package():
    """The closure, not only the direct imports: a transitive edge would show up here.

    A fresh interpreter per package is the only honest way to see this, because the test
    session has already imported most of the package graph by the time this runs.
    """
    packages = (
        "mlfcs.structure",
        "mlfcs.interactions",
        "mlfcs.force_constants",
        "mlfcs.constraints",
        "mlfcs.finite_difference",
        "mlfcs.fitting",
        "mlfcs.io",
        "mlfcs.calculators",
        "mlfcs.sampling",
    )
    probe = (
        "import sys; import {module}; "
        "leaked = sorted(name for name in sys.modules "
        "if name == 'mlfcs.reciprocal' or name.startswith('mlfcs.reciprocal.')); "
        "assert not leaked, leaked"
    )
    for package in packages:
        completed = subprocess.run(
            [sys.executable, "-c", probe.format(module=package)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{package}: {completed.stderr.strip()}"


def test_legacy_phonon_package_is_removed():
    assert not (ROOT / "phonon").exists()
    assert not (ROOT / "structure" / "reciprocal.py").exists()


def test_historical_ambiguous_packages_are_removed():
    for package in ("core", "ifc", "anharmonic", "basis", "public"):
        assert not (ROOT / package).exists()


def test_taylor_model_does_not_leak_into_generic_modules():
    taylor_root = ROOT / "fitting" / "taylor"
    for path in ROOT.rglob("*.py"):
        if path.is_relative_to(taylor_root):
            continue
        source = path.read_text()
        assert "mlfcs.fitting.backends" not in source, path


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
