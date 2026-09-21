"""Immutable numeric plans for the star expansion and the Fourier kernels.

Everything that depends only on the primitive symmetry, the reciprocal grid and the positional
gauge is built once here and shared: the gate, the covariance and the sampler all consume the
same plan, so an integer inverse, a site permutation or a gauge phase is never recomputed in a
member loop.  A plan is identified by its content -- structure, grid, ``symprec`` and the
time-reversal policy -- never by an object address, and it is immutable, so one consumer cannot
mutate another's geometry.

The structured form is deliberate.  A dense ``3n x 3n`` unitary per member would be the easy
cache, but it is mostly zeros: the operation is a site permutation, one Cartesian rotation and
one global phase, and keeping those three pieces lets a caller assemble the matrix in the same
arithmetic the uncached helpers use while still paying for the integer inverse only once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.reciprocal.fourier import FourierTerm
from mlfcs.reciprocal.grid import IrreducibleReciprocalGrid
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


@dataclass(frozen=True, slots=True)
class FourierPlan:
    """Flattened, fixed-dtype arrays of one exact primitive-lattice FC2 term set.

    The plan is a representation change: it stores what the term tuple stores, with the mass
    weight folded in, so a compiled kernel can consume contiguous arrays instead of Python
    objects.  It is not wired into the hot path until a kernel consumes it, because the
    tuple-based kernel is already vectorized over q points.
    """

    first: np.ndarray
    second: np.ndarray
    images: np.ndarray
    tensors: np.ndarray
    mass_weight: np.ndarray
    n_sites: int

    @classmethod
    def from_terms(cls, terms: tuple[FourierTerm, ...], masses: np.ndarray) -> FourierPlan:
        """Return the flattened plan of a term tuple, with the mass weights folded in."""
        values = tuple(terms)
        weight = np.asarray(masses, dtype=float)
        if weight.ndim != 1 or weight.size == 0:
            raise ValueError("masses must be a non-empty vector")
        if not values:
            raise ValueError("a Fourier plan needs at least one term")
        first = np.asarray([term.first for term in values], dtype=np.int32)
        second = np.asarray([term.second for term in values], dtype=np.int32)
        if first.max(initial=0) >= weight.size or second.max(initial=0) >= weight.size:
            raise ValueError("a term refers to a site outside the mass vector")
        return cls(
            first=first,
            second=second,
            images=np.asarray([term.images for term in values], dtype=float),
            tensors=np.asarray([term.tensor for term in values], dtype=float),
            mass_weight=1.0 / np.sqrt(weight[first] * weight[second]),
            n_sites=int(weight.size),
        )


@dataclass(frozen=True, slots=True)
class ReciprocalExpansionPlan:
    """Cached geometry of every star member of one reciprocal grid."""

    grid: IrreducibleReciprocalGrid
    positions: np.ndarray
    stars: np.ndarray
    representatives: np.ndarray
    operations: np.ndarray
    antiunitary: np.ndarray
    site_permutations: np.ndarray
    cartesian_rotations: np.ndarray
    operation_phases: np.ndarray
    member_gauges: np.ndarray

    @classmethod
    def from_grid(
        cls,
        symmetry: PrimitiveSymmetryOperations,
        grid: IrreducibleReciprocalGrid,
        primitive_positions: np.ndarray,
    ) -> ReciprocalExpansionPlan:
        """Build the plan of one grid, computing each integer inverse and gauge exactly once.

        The pieces are read from the same helpers the uncached path uses, so assembling a member
        matrix from the plan is bit-identical to calling them directly.
        """
        from mlfcs.reciprocal.grid import _integer_inverse
        from mlfcs.reciprocal.symmetry import star_member_gauge, star_member_operator

        positions = np.asarray(primitive_positions, dtype=float)
        n_primitive = int(np.asarray(symmetry.site_permutations).shape[1])
        if positions.shape != (n_primitive, 3):
            raise ValueError(
                f"expected positions of shape ({n_primitive}, 3), got {positions.shape}"
            )
        members = len(grid.full.labels)
        operations = np.asarray(grid.full_operations, dtype=np.int64)
        antiunitary = np.asarray(grid.full_antiunitary, dtype=bool)
        cartesian = np.asarray(symmetry.cartesian_rotations, dtype=float)
        permutations = np.asarray(symmetry.site_permutations, dtype=np.int64)
        gauges = np.empty((members, 3 * n_primitive), dtype=complex)
        phases = np.empty(members, dtype=complex)
        translations = np.asarray(symmetry.translations, dtype=float)
        denominator = grid.full.denominator
        for member in range(members):
            _, flag = star_member_operator(symmetry, grid, member)
            gauges[member] = star_member_gauge(symmetry, grid, member, positions)
            # The operator is one global phase times the site permutation and rotation.  The
            # phase is the same expression displacement_representation evaluates, so assembling
            # from the plan is bit-identical to calling that helper.
            operation = int(operations[member])
            representative = int(grid.representatives[int(grid.full_to_irreducible[member])])
            label = np.asarray(grid.full.labels[representative], dtype=np.int64)
            rotated = label @ _integer_inverse(np.asarray(symmetry.rotations[operation], dtype=np.int64))
            phases[member] = np.exp(
                -2j * np.pi * float(np.asarray(rotated, dtype=float) @ translations[operation])
                / denominator
            )
            if bool(flag) != bool(antiunitary[member]):
                raise RuntimeError(f"member {member} disagrees about its antiunitary flag")
        return cls(
            grid=grid,
            positions=positions,
            stars=np.asarray(grid.full_to_irreducible, dtype=np.int64),
            representatives=np.asarray(grid.representatives, dtype=np.int64),
            operations=operations,
            antiunitary=antiunitary,
            site_permutations=permutations[operations],
            cartesian_rotations=cartesian[operations],
            operation_phases=phases,
            member_gauges=gauges,
        )

    def unitary(self, member: int) -> np.ndarray:
        """Assemble ``U_g`` of one member from the stored permutation, rotation and phase."""
        n = self.site_permutations.shape[1]
        out = np.zeros((3 * n, 3 * n), dtype=complex)
        rotation = self.cartesian_rotations[member].T
        for site in range(n):
            target = int(self.site_permutations[member, site])
            out[3 * target : 3 * target + 3, 3 * site : 3 * site + 3] = (
                self.operation_phases[member] * rotation
            )
        return out

    def star(self, member: int) -> int:
        """Return the star index, i.e. the row of a per-representative array."""
        return int(self.stars[member])

    def apply_to_matrix(self, member: int, matrix: np.ndarray) -> np.ndarray:
        """Return ``Gamma U conj?[W] U^dagger Gamma^dagger`` of one member."""
        source = np.conjugate(matrix) if self.antiunitary[member] else matrix
        unitary = self.unitary(member)
        expanded = unitary @ source @ unitary.conj().T
        gauge = self.member_gauges[member]
        return gauge[:, None] * expanded * gauge.conj()[None, :]

    def apply_to_vectors(self, member: int, vectors: np.ndarray) -> np.ndarray:
        """Return ``Gamma U conj?[E]`` of one member, applied column by column."""
        source = np.conjugate(vectors) if self.antiunitary[member] else vectors
        return self.member_gauges[member][:, None] * (self.unitary(member) @ source)


__all__ = ["FourierPlan", "ReciprocalExpansionPlan"]
