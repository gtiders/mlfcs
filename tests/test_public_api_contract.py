"""Public entry points must expose usable signatures without wrapper spelunking."""

from __future__ import annotations

import inspect

from _architecture_helpers import module_imports

from mlfcs import (
    FiniteDifferenceCalculation,
    ForceConstantFitter,
    MLFCSCalculator,
    enforce_rotational_sum_rules,
    perturb_structures,
    read_hdf5,
    realize_force_constants,
    write_force_constants,
)
from mlfcs.reciprocal import SSCHA, LoopSCPH


def test_public_callables_have_explicit_documented_signatures():
    callables = (
        FiniteDifferenceCalculation,
        ForceConstantFitter,
        LoopSCPH,
        MLFCSCalculator,
        SSCHA,
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
        "MLFCSCalculator",
        "PrimitiveInteractionSpace",
        "RealizedInteractionSpace",
        "ReferenceFrame",
        "SSCHA",
        "perturb_structures",
        "read_hdf5",
        "write_force_constants",
        "realize_force_constants",
        "enforce_rotational_sum_rules",
    }


def test_reciprocal_workflows_are_not_root_exports():
    """Harmonic sampling, SCPH and SSCHA belong to `mlfcs.reciprocal`.

    Importing the reciprocal workflows from the root namespace would make every mainline
    consumer look like a reciprocal-space user, which is exactly what the one-way
    dependency rule forbids.
    """
    import mlfcs

    for name in ("LoopSCPH", "SSCHA", "HarmonicSampler", "harmonic_frequencies"):
        assert not hasattr(mlfcs, name), name
        assert name not in mlfcs.__all__


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
                    "mlfcs.reciprocal.scph.solver",
                    "mlfcs.reciprocal.sscha.solver",
                )
            )
            for value in imports
        ), module

    force_constant_imports = module_imports("force_constants.representation")
    assert not any(
        value.startswith(("mlfcs.fitting", "mlfcs.reciprocal", "mlfcs.io", "mlfcs.constraints"))
        for value in force_constant_imports
    )

    for module in ("io.alamode", "io.hdf5", "io.phonon_hdf5", "io.phonopy", "io.shengbte"):
        imports = module_imports(module)
        assert not any(
            value.startswith(
                (
                    "mlfcs.finite_difference.calculation",
                    "mlfcs.fitting",
                    "mlfcs.reciprocal.sscha.solver",
                    "mlfcs.reciprocal.scph.solver",
                )
            )
            for value in imports
        ), module
