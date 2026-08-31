"""Lock the package dependency graph after the responsibility refactor."""

from __future__ import annotations

import ast

from _architecture_helpers import ROOT, internal_dependencies

ALLOWED = {
    "structure": set(),
    "interactions": {"exceptions", "structure"},
    "force_constants": {"interactions", "structure"},
    "constraints": {"force_constants", "interactions", "structure"},
    "finite_difference": {"constraints", "force_constants", "interactions", "structure"},
    "fitting": {"constraints", "force_constants", "interactions", "structure"},
    # SCPH/SSCHA consume force constants and the fitter, while fitting itself
    # remains independent of phonon workflows.
    "phonon": {"fitting", "force_constants", "structure"},
    "io": {"force_constants", "structure"},
}


def test_package_dependencies_follow_the_locked_dag():
    for package, allowed in ALLOWED.items():
        unexpected = internal_dependencies(package) - allowed
        assert not unexpected, f"{package} has forbidden dependencies: {sorted(unexpected)}"


def test_mainline_packages_do_not_depend_on_phonon_workflows():
    for package in ("structure", "interactions", "force_constants", "constraints", "finite_difference", "fitting"):
        assert "phonon" not in internal_dependencies(package), package


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
