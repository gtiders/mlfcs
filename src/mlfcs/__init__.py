"""ASE-first anharmonic force-constant tools."""

from pathlib import Path

from mlfcs.anharmonic import LoopSCPH, LoopSCPHResult, harmonic_frequencies
from mlfcs.api import Calculation, ForceConstantCalculation
from mlfcs.core.geometry import PeriodicIndex, StructureRelation, align_structures, build_supercell
from mlfcs.finite_difference.stencil import CentralDifferenceStencil
from mlfcs.harmonic_constraints import (
    HarmonicConstraintDiagnostics,
    HarmonicConstraintResult,
    enforce_harmonic_constraints,
)
from mlfcs.model import ForceConstants, SparseOrderForceConstants

# SSCHA remains an explicit submodule so the base namespace stays compact.

__all__ = [
    "Calculation",
    "CentralDifferenceStencil",
    "ForceConstantCalculation",
    "ForceConstants",
    "HarmonicConstraintDiagnostics",
    "HarmonicConstraintResult",
    "LoopSCPH",
    "LoopSCPHResult",
    "PeriodicIndex",
    "SparseOrderForceConstants",
    "StructureRelation",
    "align_structures",
    "build_supercell",
    "enforce_harmonic_constraints",
    "harmonic_frequencies",
    "read_hdf5",
]

__version__ = "4.0.0a2"


def read_hdf5(source: str | Path) -> ForceConstants:
    """Read native MLFCS HDF5 schema v2 force constants.

    Older native schemas are intentionally rejected because their atom-order
    semantics are not recoverable without guessing.
    """
    from mlfcs.io.hdf5 import read_hdf5 as _read_hdf5

    return _read_hdf5(source)
