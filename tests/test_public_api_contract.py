"""Public entry points must expose usable signatures without wrapper spelunking."""

from __future__ import annotations

import inspect

from _architecture_helpers import module_imports

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
        "InteractionSpace",
        "LoopSCPH",
        "MLFCSCalculator",
        "PrimitiveInteractionSpace",
        "RealizedInteractionSpace",
        "ReferenceFrame",
        "SSCHA",
        "build_supercell",
        "perturb_structures",
        "read_hdf5",
        "write_force_constants",
        "realize_force_constants",
        "enforce_rotational_sum_rules",
    }


def test_low_level_packages_do_not_depend_on_workflow_or_writer_modules():
    for module in (
        "structure.periodic_geometry",
        "structure.supercell_mapping",
        "interactions.algebra.actions",
        "interactions.models",
        "constraints.rotational",
        "constraints.translational",
    ):
        imports = module_imports(module)
        assert not any(
            value.startswith(
                (
                    "mlfcs.io",
                    "mlfcs.fitting",
                    "mlfcs.phonon.scph.solver",
                    "mlfcs.phonon.sscha.solver",
                )
            )
            for value in imports
        ), module

    force_constant_imports = module_imports("force_constants.representation")
    assert not any(
        value.startswith(("mlfcs.fitting", "mlfcs.phonon", "mlfcs.io", "mlfcs.constraints"))
        for value in force_constant_imports
    )

    for module in ("io.alamode", "io.hdf5", "io.phonon_hdf5", "io.phonopy", "io.shengbte"):
        imports = module_imports(module)
        assert not any(
            value.startswith(
                (
                    "mlfcs.finite_difference.calculation",
                    "mlfcs.fitting",
                    "mlfcs.phonon.sscha.solver",
                    "mlfcs.phonon.scph.solver",
                )
            )
            for value in imports
        ), module
