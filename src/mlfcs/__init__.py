"""ASE-first anharmonic force-constant tools."""

from importlib import import_module

from mlfcs.anharmonic.scph import LoopSCPH, LoopSCPHResult, harmonic_frequencies
from mlfcs.anharmonic.common.schedule import TemperatureSeriesResult
from mlfcs.api import Calculation, ForceConstantCalculation
from mlfcs.constraints.rotational_sum_rules import (
    RotationalSumRuleDiagnostics,
    RotationalSumRuleResult,
    enforce_rotational_sum_rules,
)
from mlfcs.core.geometry import (
    PeriodicIndex,
    StructureRelation,
    align_structures,
)
from mlfcs.core.supercell import build_supercell
from mlfcs.finite_difference.stencil import CentralDifferenceStencil
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.public.io import read_hdf5, write_force_constants

__all__ = [
    "SSCHA",
    "Calculation",
    "CentralDifferenceStencil",
    "EnsembleDiagnostics",
    "FitDataset",
    "FittingDiagnostics",
    "FittingResult",
    "ForceConstantCalculation",
    "ForceConstantFitter",
    "ForceConstants",
    "RotationalSumRuleDiagnostics",
    "RotationalSumRuleResult",
    "HarmonicEnsemble",
    "LoopSCPH",
    "LoopSCPHResult",
    "PeriodicIndex",
    "SSCHAIteration",
    "SparseOrderForceConstants",
    "StructureRelation",
    "TemperatureSeriesResult",
    "align_structures",
    "build_supercell",
    "enforce_rotational_sum_rules",
    "harmonic_frequencies",
    "read_hdf5",
    "write_force_constants",
]

__version__ = "4.0.0a3"


def __getattr__(name: str):
    """Load fitting and SSCHA APIs only when explicitly requested."""
    if name in {"FitDataset", "FittingDiagnostics", "FittingResult", "ForceConstantFitter"}:
        return getattr(import_module("mlfcs.fitting"), name)
    if name in {"EnsembleDiagnostics", "HarmonicEnsemble", "SSCHA", "SSCHAIteration"}:
        return getattr(import_module("mlfcs.anharmonic.sscha"), name)
    raise AttributeError(name)
