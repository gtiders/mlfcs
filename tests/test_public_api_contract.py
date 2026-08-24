"""Public entry points must expose usable signatures without wrapper spelunking."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mlfcs import (
    SSCHA,
    FiniteDifferenceCalculation,
    ForceConstantFitter,
    LoopSCPH,
    MLFCSCalculator,
    build_supercell,
    enforce_rotational_sum_rules,
    perturb_structures,
    read_hdf5,
    realize_force_constants,
    write_force_constants,
)


def test_public_callables_have_explicit_documented_signatures():
    callables = (
        FiniteDifferenceCalculation,
        ForceConstantFitter,
        LoopSCPH,
        MLFCSCalculator,
        SSCHA,
        build_supercell,
        perturb_structures,
        enforce_rotational_sum_rules,
        realize_force_constants,
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


def test_top_level_api_is_the_locked_whitelist():
    import mlfcs

    assert set(mlfcs.__all__) == {
        "FiniteDifferenceCalculation",
        "ForceConstantFitter",
        "ForceConstants",
        "LoopSCPH",
        "MLFCSCalculator",
        "SSCHA",
        "build_supercell",
        "perturb_structures",
        "read_hdf5",
        "write_force_constants",
        "realize_force_constants",
        "enforce_rotational_sum_rules",
    }


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
        "structure.periodic_geometry",
        "structure.supercell_mapping",
        "interactions.tensors",
        "interactions.orbits",
        "constraints.rotational",
        "constraints.translational",
    ):
        imports = _imports(module)
        assert not any(
            value.startswith(
                (
                    "mlfcs.io",
                    "mlfcs.fitting",
                    "mlfcs.physics.scph.solver",
                    "mlfcs.physics.sscha.solver",
                )
            )
            for value in imports
        ), module

    force_constant_imports = _imports("force_constants.representation")
    assert not any(
        value.startswith(("mlfcs.fitting", "mlfcs.physics", "mlfcs.io", "mlfcs.constraints"))
        for value in force_constant_imports
    )

    for module in ("io.alamode", "io.hdf5", "io.phonon_hdf5", "io.phonopy", "io.shengbte"):
        imports = _imports(module)
        assert not any(
            value.startswith(
                (
                    "mlfcs.finite_difference.calculation",
                    "mlfcs.fitting",
                    "mlfcs.physics.sscha.solver",
                    "mlfcs.physics.scph.solver",
                )
            )
            for value in imports
        ), module
