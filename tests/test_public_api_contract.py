"""Public entry points must expose usable signatures without wrapper spelunking."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mlfcs.public import (
    SSCHA,
    ForceConstantCalculation,
    ForceConstantFitter,
    LoopSCPH,
    align_structures,
    build_supercell,
    enforce_harmonic_constraints,
    harmonic_frequencies,
    read_hdf5,
    write_force_constants,
)


def test_public_callables_have_explicit_documented_signatures():
    callables = (
        ForceConstantCalculation,
        ForceConstantFitter,
        LoopSCPH,
        SSCHA,
        align_structures,
        build_supercell,
        enforce_harmonic_constraints,
        harmonic_frequencies,
        read_hdf5,
        write_force_constants,
    )
    for callable_ in callables:
        signature = inspect.signature(callable_)
        assert inspect.getdoc(callable_)
        assert all(
            parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in signature.parameters.values()
        ), callable_.__qualname__


def _imports(module: str) -> set[str]:
    path = Path(__file__).parents[1] / "src" / "mlfcs" / Path(*module.split("."))
    source = path.with_suffix(".py").read_text()
    tree = ast.parse(source)
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def test_low_level_packages_do_not_depend_on_workflow_or_writer_modules():
    for module in (
        "structure.geometry",
        "clusters.orbits",
        "constraints.harmonic",
        "constraints.translational",
    ):
        imports = _imports(module)
        assert not any(
            value.startswith(
                ("mlfcs.io", "mlfcs.fitting", "mlfcs.anharmonic.scph", "mlfcs.anharmonic.sscha")
            )
            for value in imports
        ), module

    # ForceConstants.write is a retained public compatibility adapter.  The
    # data model otherwise has no workflow dependency.
    ifc_imports = _imports("ifc.model")
    assert not any(
        value.startswith(("mlfcs.fitting", "mlfcs.anharmonic.scph", "mlfcs.anharmonic.sscha"))
        for value in ifc_imports
    )

    for module in ("io.alamode", "io.hdf5", "io.phonon_hdf5", "io.phonopy", "io.shengbte"):
        imports = _imports(module)
        assert not any(
            value.startswith(
                ("mlfcs.api", "mlfcs.fitting", "mlfcs.anharmonic.sscha", "mlfcs.anharmonic.scph")
            )
            for value in imports
        ), module
