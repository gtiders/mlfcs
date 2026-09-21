"""Reciprocal-space models: exact q grids, symmetry reduction and harmonic workflows.

The package owns everything that needs a reciprocal grid:

* :mod:`mlfcs.reciprocal.grid` builds the exact quotient labels of a supercell translation
  group and the symmetry decomposition of that grid;
* :mod:`mlfcs.reciprocal.symmetry` and :mod:`mlfcs.reciprocal.fourier` carry the Fourier
  gauge and the space-group representation of the mass-weighted displacement space;
* :mod:`mlfcs.reciprocal.modes`, :mod:`mlfcs.reciprocal.sampling`,
  :mod:`mlfcs.reciprocal.scph` and :mod:`mlfcs.reciprocal.sscha` are the consumers, and
  none of them re-derives q-point symmetry on its own.

Dependencies point one way: this package may import :mod:`mlfcs.structure`,
:mod:`mlfcs.force_constants`, :mod:`mlfcs.sampling` and the fitter, while no other
production package may import :mod:`mlfcs.reciprocal`.  That rule is enforced by
``tests/test_reciprocal_boundaries.py``.
"""

from mlfcs.reciprocal.grid import (
    IrreducibleReciprocalGrid,
    ReciprocalGridSymmetry,
    ReciprocalQuotientGrid,
    ReciprocalStar,
    irreducible_reciprocal_grid,
    quotient_qpoints,
    reciprocal_grid_symmetry,
    reciprocal_quotient_grid,
)
from mlfcs.reciprocal.sampling.harmonic import HarmonicSampler, SamplingState
from mlfcs.reciprocal.sampling.structures import SamplingBatch, perturb_structures
from mlfcs.reciprocal.scph.fourier import HarmonicMeshResult, harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH, LoopSCPHIteration, LoopSCPHResult
from mlfcs.reciprocal.sscha.solver import SSCHA, SSCHAIteration, SSCHAResult

__all__ = [
    "SSCHA",
    "HarmonicMeshResult",
    "HarmonicSampler",
    "IrreducibleReciprocalGrid",
    "LoopSCPH",
    "LoopSCPHIteration",
    "LoopSCPHResult",
    "ReciprocalGridSymmetry",
    "ReciprocalQuotientGrid",
    "ReciprocalStar",
    "SSCHAIteration",
    "SSCHAResult",
    "SamplingBatch",
    "SamplingState",
    "harmonic_frequencies",
    "irreducible_reciprocal_grid",
    "perturb_structures",
    "quotient_qpoints",
    "reciprocal_grid_symmetry",
    "reciprocal_quotient_grid",
]
