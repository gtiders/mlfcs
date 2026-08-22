"""ASE-first anharmonic force-constant tools."""

from importlib import import_module

from mlfcs.anharmonic import LoopSCPH, LoopSCPHResult, harmonic_frequencies
from mlfcs.api import Calculation, ForceConstantCalculation
from mlfcs.constraints.harmonic import (
    HarmonicConstraintDiagnostics,
    HarmonicConstraintResult,
    enforce_harmonic_constraints,
)
from mlfcs.finite_difference.stencil import CentralDifferenceStencil
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.public.io import read_hdf5, write_force_constants
from mlfcs.structure.geometry import (
    PeriodicIndex,
    StructureRelation,
    align_structures,
    build_supercell,
)

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
    "HarmonicConstraintDiagnostics",
    "HarmonicConstraintResult",
    "HarmonicEnsemble",
    "LoopSCPH",
    "LoopSCPHResult",
    "PeriodicIndex",
    "SSCHAIteration",
    "SparseOrderForceConstants",
    "StructureRelation",
    "align_structures",
    "build_supercell",
    "enforce_harmonic_constraints",
    "harmonic_frequencies",
    "read_hdf5",
    "write_force_constants",
]

__version__ = "4.0.0a2"


def __getattr__(name: str):
    """Load fitting and SSCHA APIs only when explicitly requested."""
    if name in {"FitDataset", "FittingDiagnostics", "FittingResult", "ForceConstantFitter"}:
        return getattr(import_module("mlfcs.fitting"), name)
    if name in {"EnsembleDiagnostics", "HarmonicEnsemble", "SSCHA", "SSCHAIteration"}:
        return getattr(import_module("mlfcs.sscha"), name)
    raise AttributeError(name)
