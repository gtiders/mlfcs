"""Positional-gauge matrix action of irreducible reciprocal stars.

The plan stores only site phases and operation indices.  A member matrix is
assembled from its representative; no Fourier matrix is built at a member.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs.core import PrimitiveCell
from mlfcs.reciprocal.grid import QStars


@njit(cache=True, nogil=True)
def _transform(
    source: np.ndarray,
    permutation: np.ndarray,
    rotation: np.ndarray,
    phase: np.ndarray,
    antiunitary: bool,
) -> np.ndarray:
    """Rotate and scatter 3-by-3 site blocks without a dense representation."""
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


@dataclass(frozen=True, slots=True)
class StarPlan:
    """Expand matrices from star representatives in the positional gauge.

    ``matrix`` builds one requested member at a time. ``iter_matrices`` streams
    members in grid order, so a consumer can accumulate Fourier blocks without
    retaining a second full-grid stack of matrices.
    """

    stars: QStars
    primitive: PrimitiveCell
    phases: np.ndarray

    @classmethod
    def from_stars(cls, stars: QStars, primitive: PrimitiveCell) -> StarPlan:
        if not isinstance(stars, QStars) or not isinstance(primitive, PrimitiveCell):
            raise TypeError("stars and primitive must be QStars and PrimitiveCell")
        if stars.symmetry.site_permutations.shape[1] != primitive.size:
            raise ValueError("star symmetry and primitive have different motif sizes")
        shifts = np.asarray(
            [stars.member_shift(member) for member in range(stars.grid.size)],
            dtype=np.float64,
        )
        phases = np.exp(2j * np.pi * (shifts @ primitive.scaled_positions.T))
        phases.setflags(write=False)
        return cls(stars, primitive, phases)

    def matrix(self, member: int, representative_matrix: object) -> np.ndarray:
        """Map one representative matrix onto one stored full-grid label.

        Time reversal conjugates first; the space-group operation then rotates
        and permutes Cartesian site blocks.  Finally the reciprocal-lattice
        shift applies its site-dependent positional phase.  The operation's
        global translation phase cancels in a matrix similarity transform.
        """
        if not 0 <= member < self.stars.grid.size:
            raise IndexError("member is outside the reciprocal grid")
        count = 3 * self.primitive.size
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
        count = 3 * self.primitive.size
        expected = (len(self.stars.representatives), count, count)
        if values.shape != expected:
            raise ValueError(f"representative matrices must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("representative matrices must be finite")
        for member in range(self.stars.grid.size):
            yield member, self.matrix(member, values[self.stars.star_of[member]])


__all__ = ["StarPlan"]
