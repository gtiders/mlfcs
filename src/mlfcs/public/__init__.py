"""Stable, user-facing MLFCS entry points.

Workflow-heavy modules are loaded lazily so importing finite-difference and
IO helpers does not initialize JAX or the SSCHA machinery.
"""

from importlib import import_module

from mlfcs.anharmonic.common.schedule import TemperatureSeriesResult
from mlfcs.anharmonic.scph import LoopSCPH, LoopSCPHResult, harmonic_frequencies
from mlfcs.core.supercell import build_supercell
from mlfcs.public.constraints import (
    RotationalSumRuleDiagnostics,
    RotationalSumRuleResult,
    enforce_rotational_sum_rules,
)
from mlfcs.public.finite_difference import Calculation, ForceConstantCalculation
from mlfcs.public.io import read_hdf5, write_force_constants
from mlfcs.public.structure import (
    PeriodicIndex,
    StructureRelation,
    align_structures,
)

__all__ = [
    "SSCHA",
    "Calculation",
    "EnsembleDiagnostics",
    "FitDataset",
    "FittingDiagnostics",
    "FittingResult",
    "ForceConstantCalculation",
    "ForceConstantFitter",
    "HarmonicEnsemble",
    "LoopSCPH",
    "LoopSCPHResult",
    "PeriodicIndex",
    "RotationalSumRuleDiagnostics",
    "RotationalSumRuleResult",
    "SSCHAIteration",
    "SSCHAResult",
    "StructureRelation",
    "TemperatureSeriesResult",
    "align_structures",
    "build_supercell",
    "enforce_rotational_sum_rules",
    "harmonic_frequencies",
    "read_hdf5",
    "write_force_constants",
]


def __getattr__(name: str):
    """Load fitting and SSCHA APIs only when explicitly requested."""
    if name in {"FitDataset", "FittingDiagnostics", "FittingResult", "ForceConstantFitter"}:
        return getattr(import_module("mlfcs.public.fitting"), name)
    if name in {"EnsembleDiagnostics", "HarmonicEnsemble", "SSCHA", "SSCHAIteration", "SSCHAResult"}:
        return getattr(import_module("mlfcs.anharmonic.sscha"), name)
    raise AttributeError(name)
