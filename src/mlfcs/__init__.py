"""ASE-first anharmonic force-constant tools."""

from importlib import import_module

from mlfcs.constraints.rotational import (
    RotationalSumRuleDiagnostics,
    RotationalSumRuleResult,
    enforce_rotational_sum_rules,
)
from mlfcs.finite_difference.calculation import Calculation, ForceConstantCalculation
from mlfcs.finite_difference.stencil import CentralDifferenceStencil
from mlfcs.force_constants.data import ForceConstants, SparseOrderForceConstants
from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.io.hdf5 import read_hdf5
from mlfcs.io.write import write_force_constants
from mlfcs.physics.scph.fourier import harmonic_frequencies
from mlfcs.physics.scph.solver import LoopSCPH, LoopSCPHResult
from mlfcs.physics.temperature import TemperatureSeriesResult
from mlfcs.structure.relation import StructureRelation, align_structures
from mlfcs.structure.supercell import build_supercell
from mlfcs.structure.supercell_mapping import PeriodicIndex

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
    "HarmonicEnsemble",
    "LoopSCPH",
    "LoopSCPHResult",
    "PeriodicIndex",
    "RotationalSumRuleDiagnostics",
    "RotationalSumRuleResult",
    "SSCHAIteration",
    "SSCHAResult",
    "SparseOrderForceConstants",
    "StructureRelation",
    "TemperatureSeriesResult",
    "align_structures",
    "build_supercell",
    "enforce_rotational_sum_rules",
    "harmonic_frequencies",
    "read_hdf5",
    "realize_force_constants",
    "write_force_constants",
]

__version__ = "4.0.0a4"


def __getattr__(name: str):
    """Load fitting and SSCHA APIs only when explicitly requested."""
    if name in {"FitDataset", "FittingDiagnostics", "FittingResult", "ForceConstantFitter"}:
        return getattr(import_module("mlfcs.fitting"), name)
    if name in {
        "EnsembleDiagnostics",
        "HarmonicEnsemble",
        "SSCHA",
        "SSCHAIteration",
        "SSCHAResult",
    }:
        return getattr(import_module("mlfcs.physics.sscha.solver"), name)
    raise AttributeError(name)
