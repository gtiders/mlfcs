"""Fourier transforms and dynamical matrices for SCPH."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.reciprocal.fourier import (
    dynamical_matrices,
    fourier_terms,
)
from mlfcs.reciprocal.grid import (
    IrreducibleReciprocalGrid,
    irreducible_reciprocal_grid,
)
from mlfcs.reciprocal.statistics import OMEGA_TO_THZ as _OMEGA_TO_THZ
from mlfcs.reciprocal.symmetry import (
    expand_star_values,
    require_star_covariance,
    validate_site_masses,
    validate_symmetry_tolerance,
    validate_symprec,
)
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


@dataclass(frozen=True, slots=True)
class HarmonicMeshResult:
    r"""Harmonic frequencies on the irreducible wedge of one q grid.

    Only the representative q points of the star decomposition are stored; the full mesh
    is available through :meth:`expand_frequencies`, which is exact because the
    covariance relation ``D(gq) = U_g(q) D(q) U_g(q)^dagger`` is a unitary similarity, so
    a whole star carries the eigenvalues of its representative.  Nothing here is
    approximate and nothing is weighted: the weights are the star sizes that sum to the
    number of full q points.

    The names are deliberately explicit.  A caller that wants the full mesh has to ask
    for it, so a plain ``qpoints`` cannot be mistaken for either set.
    """

    irreducible_qpoints: np.ndarray
    irreducible_frequencies: np.ndarray
    weights: np.ndarray
    grid: IrreducibleReciprocalGrid
    symprec: float
    time_reversal: bool
    symmetry_tolerance: float | None

    @property
    def n_qpoints(self) -> int:
        """Number of q points of the full grid."""
        return len(self.grid.full.labels)

    @property
    def n_irreducible(self) -> int:
        """Number of irreducible representative q points."""
        return len(self.grid.representatives)

    @property
    def reduction_ratio(self) -> float:
        """How many full q points one representative stands for."""
        return self.n_qpoints / self.n_irreducible

    def full_qpoints(self) -> np.ndarray:
        """Return the q points of the full grid, in full-grid order."""
        return self.grid.full.points

    def expand_frequencies(self) -> np.ndarray:
        """Return the full-grid frequencies, in full-grid order."""
        return expand_star_values(self.irreducible_frequencies, self.grid)


def harmonic_frequencies(
    fc2: ForceConstants,
    interpolation_multiplier: int = 1,
    *,
    symprec: float = 1e-5,
    time_reversal: bool = True,
    symmetry_tolerance: float | None = 1e-6,
) -> HarmonicMeshResult:
    """Return harmonic frequencies on the irreducible wedge of a q grid.

    The dynamical matrices come from :mod:`mlfcs.reciprocal.fourier`, so this path and the
    harmonic sampler share one gauge, and they are diagonalized once per star
    representative.  ``symprec`` is the geometric tolerance used to identify the primitive
    symmetry; it is a distinct quantity from any dynamical-matrix tolerance and is
    recorded in the result rather than hidden in a module constant.
    """
    if 2 not in fc2.orders or fc2.relation is None:
        raise ValueError("fc2 must contain order-2 force constants and a structure relation")
    symprec = validate_symprec(symprec, context="harmonic_frequencies")
    symmetry_tolerance = validate_symmetry_tolerance(
        symmetry_tolerance, context="harmonic_frequencies"
    )
    primitive = fc2.relation.primitive
    masses = np.asarray(primitive.get_masses(), dtype=float)
    lattice = lattice_fc2(fc2)
    terms = fourier_terms(lattice, primitive)
    multiplier = _multiplier(interpolation_multiplier, "interpolation_multiplier")
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=symprec)
    validate_site_masses(symmetry, masses, context="harmonic_frequencies")
    grid = irreducible_reciprocal_grid(
        multiplier * fc2.relation.supercell_matrix, symmetry, time_reversal=time_reversal
    )
    # The star expansion is only legitimate if every member the expansion produces agrees
    # with the matrix built directly at that member's own label; the little group alone does
    # not certify that, so the full star is checked before anything is diagonalized.
    require_star_covariance(
        partial(dynamical_matrices, terms, masses),
        symmetry,
        grid,
        np.asarray(primitive.get_scaled_positions(wrap=False), dtype=float),
        tolerance=symmetry_tolerance,
        context="harmonic_frequencies",
    )
    qpoints = grid.full.points[grid.representatives]
    eigenvalues = np.linalg.eigvalsh(dynamical_matrices(terms, masses, qpoints))
    frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * _OMEGA_TO_THZ
    return HarmonicMeshResult(
        irreducible_qpoints=np.asarray(qpoints, dtype=float),
        irreducible_frequencies=np.asarray(frequencies, dtype=float),
        weights=np.asarray(grid.weights, dtype=np.int64),
        grid=grid,
        symprec=float(symprec),
        time_reversal=bool(time_reversal),
        symmetry_tolerance=None if symmetry_tolerance is None else float(symmetry_tolerance),
    )


def _multiplier(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_relation(fc2: ForceConstants, fc4: ForceConstants) -> None:
    r2, r4 = fc2.relation, fc4.relation
    if r2 is None or r4 is None:
        raise ValueError("fc2 and fc4 must contain explicit StructureRelation objects")
    if (
        len(r2.primitive) != len(r4.primitive)
        or not np.array_equal(r2.primitive.numbers, r4.primitive.numbers)
        or not np.allclose(r2.primitive.cell, r4.primitive.cell, atol=1e-8, rtol=1e-10)
        or not np.allclose(r2.primitive.positions, r4.primitive.positions, atol=1e-8, rtol=1e-10)
    ):
        raise ValueError("fc2 and fc4 primitive structures differ")


def _needed_covariances(
    sparse: SparseOrderForceConstants,
) -> set[tuple[int, int, tuple[int, int, int]]]:
    result: set[tuple[int, int, tuple[int, int, int]]] = set()
    for sites, translations in zip(sparse.sites, sparse.translations, strict=True):
        result.add(
            (
                int(sites[2]),
                int(sites[3]),
                tuple((np.asarray(translations[1]) - np.asarray(translations[2])).tolist()),
            )
        )
    return result


__all__ = ["HarmonicMeshResult", "harmonic_frequencies"]
