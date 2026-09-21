"""Lock the package dependency graph after the responsibility refactor."""

from __future__ import annotations

import ast

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
