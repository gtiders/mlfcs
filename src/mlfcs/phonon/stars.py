"""Positional-gauge matrix action of irreducible reciprocal stars.

The plan stores only site phases and operation indices.  A member matrix is
assembled from its representative; no Fourier matrix is built at a member.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs.cluster_space import ClusterSpace
from mlfcs.phonon.grid import QStars


@njit(cache=True, nogil=True)
def _transform(
    source: np.ndarray,
    permutation: np.ndarray,
    rotation: np.ndarray,
    phase: np.ndarray,
    antiunitary: bool,
) -> np.ndarray:
    """Return the rotated/permuted positional-gauge matrix for one star member.

    source is complex128 (3*N, 3*N), permutation (N,), Cartesian row rotation
    (3, 3), and phase (N,). Conjugate source first if antiunitary; rotate site
    blocks, scatter by permutation, and apply phase_i*conj(phase_j).
    No dense action matrix is built and inputs are preserved.
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
    """Stream positional-gauge matrices from irreducible representatives.

    stars and space are referenced domain objects. phases is complex128
    (grid.size, n_atoms), readonly and prepared during direct initialization.
    matrix creates one member;
    iter_matrices streams grid order without retaining another full-grid stack.
    """

    stars: QStars
    space: ClusterSpace
    phases: np.ndarray

    def __init__(self, stars: QStars, space: ClusterSpace) -> None:
        """Prepare readonly site phases for expanding star matrices in positional gauge.

        Reference QStars and ClusterSpace are retained, not copied. phases has
        shape (grid.size, n_atoms), using exp(2*pi*i*G.dot(fractional_position))
        from each member_shift. Wrong types or motif sizes raise an error;
        full-grid Fourier matrices are not constructed.
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
        """Map one representative matrix onto one stored full-grid label.

        Time reversal conjugates first; the space-group operation then rotates
        and permutes Cartesian site blocks.  Finally the reciprocal-lattice
        shift applies its site-dependent positional phase.  The operation's
        global translation phase cancels in a matrix similarity transform.
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
        return _transform(
            np.ascontiguousarray(source),
            self.stars.symmetry.site_permutations[operation],
            self.stars.symmetry.cartesian_rotations[operation],
            self.phases[member],
            bool(self.stars.antiunitary[member]),
        )

    def iter_matrices(self, representatives: object) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(member, matrix)`` without retaining the full-grid stack."""
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
