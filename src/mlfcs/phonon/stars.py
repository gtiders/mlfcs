"""Expand irreducible reciprocal-space matrices in the positional gauge.

Full-grid matrices are reconstructed from irreducible representatives using
the stored space-group route, optional time reversal, and reciprocal-lattice
shift associated with each star member. The plan stores symmetry metadata and
site-dependent gauge phases; Fourier matrices are evaluated only at
irreducible representatives.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs.cluster_space import ClusterSpace
from mlfcs.phonon.grid import QStars


@njit(cache=True, nogil=True)
def _transform_star_matrix(
    source: np.ndarray,
    permutation: np.ndarray,
    rotation: np.ndarray,
    phase: np.ndarray,
    antiunitary: bool,
) -> np.ndarray:
    """Transform one representative matrix to a reciprocal-star member.

    For ``g(i) = permutation[i]``, the transformed block is

        D'[g(i)α,g(j)β] = phase[g(i)]*conj(phase[g(j)])
            * sum_ℓr R[ℓ,α] D[*][iℓ,jr] R[r,β],

    where ``D[*]`` is the complex conjugate when time reversal is used and is
    ``D`` otherwise. No dense symmetry-action matrix is formed.
    """
    count = len(permutation)
    result = np.empty_like(source)
    for first in range(count):
        target_first = permutation[first]
        for second in range(count):
            target_second = permutation[second]
            factor = phase[target_first] * np.conj(phase[target_second])
            for alpha in range(3):
                for beta in range(3):
                    value = 0.0j
                    for left in range(3):
                        for right in range(3):
                            entry = source[3 * first + left, 3 * second + right]
                            if antiunitary:
                                entry = np.conj(entry)
                            value += rotation[left, alpha] * entry * rotation[right, beta]
                    result[3 * target_first + alpha, 3 * target_second + beta] = factor * value
    return result


@dataclass(frozen=True, slots=True, init=False)
class StarPlan:
    """Symmetry plan for expanding irreducible reciprocal-space matrices.

    Scalar star-invariant quantities can be expanded by star membership alone,
    but positional-gauge matrices require the full symmetry route. For each
    full-grid member, this plan combines the primitive space-group operation,
    optional time reversal, and reciprocal-lattice shift into the site-dependent
    phase needed to transform its irreducible representative matrix. Full-grid
    matrices are generated on demand rather than stored by the plan.
    """

    stars: QStars
    space: ClusterSpace
    phases: np.ndarray

    def __init__(self, stars: QStars, space: ClusterSpace) -> None:
        """Prepare positional-gauge phases for every reciprocal-star member.

        For reciprocal-lattice shift ``G`` and fractional motif position ``s_i``,
        the phase associated with site ``i`` is
        ``exp(2*pi*i*G·s_i)``. The supplied stars and cluster space must describe
        the same primitive motif.
        """
        if not isinstance(stars, QStars) or not isinstance(space, ClusterSpace):
            raise TypeError("stars and space must be QStars and a ClusterSpace")
        if stars.symmetry.site_permutations.shape[1] != len(space.atomic_numbers):
            raise ValueError("star symmetry and primitive have different motif sizes")
        shifts = np.asarray(
            [stars.member_shift(member) for member in range(stars.grid.size)],
            dtype=np.float64,
        )
        phases = np.exp(2j * np.pi * (shifts @ space.scaled_positions.T))
        phases.setflags(write=False)
        object.__setattr__(self, "stars", stars)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "phases", phases)

    def matrix(self, member: int, representative_matrix: object) -> np.ndarray:
        """Transform one irreducible representative matrix to a full-grid member.

        The stored route applies complex conjugation when time reversal is
        required, then the Cartesian rotation and motif-site permutation, and
        finally the site-dependent positional-gauge phase from the reciprocal-
        lattice shift. The operation's global translation phase cancels in a
        matrix similarity transform.
        """
        if not 0 <= member < self.stars.grid.size:
            raise IndexError("member is outside the reciprocal grid")
        count = 3 * len(self.space.atomic_numbers)
        source = np.asarray(representative_matrix, dtype=np.complex128)
        if source.shape != (count, count):
            raise ValueError(f"representative matrix must have shape ({count}, {count})")
        if not np.all(np.isfinite(source)):
            raise ValueError("representative matrix must be finite")
        operation = int(self.stars.operations[member])
        return _transform_star_matrix(
            np.ascontiguousarray(source),
            self.stars.symmetry.site_permutations[operation],
            self.stars.symmetry.cartesian_rotations[operation],
            self.phases[member],
            bool(self.stars.antiunitary[member]),
        )

    def iter_matrices(self, representatives: object) -> Iterator[tuple[int, np.ndarray]]:
        """Yield full-grid matrices in grid order without materializing the stack."""
        values = np.asarray(representatives)
        count = 3 * len(self.space.atomic_numbers)
        expected = (len(self.stars.representatives), count, count)
        if values.shape != expected:
            raise ValueError(f"representative matrices must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("representative matrices must be finite")
        for member in range(self.stars.grid.size):
            yield member, self.matrix(member, values[self.stars.star_of[member]])


__all__ = ["StarPlan"]
